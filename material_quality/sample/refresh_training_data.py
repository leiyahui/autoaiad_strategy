"""只读导出投放明细，在本地构造可断点恢复的训练样本。"""

import argparse
import datetime
import hashlib
import json
import os
import shlex
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from sqlalchemy import create_engine, text

from plan_logger import log_info, set_process_name


def load_database_url(config_path: str) -> str:
    """只读取指定变量，避免执行配置文件中的任意命令。"""
    value = os.getenv("MATERIAL_QUALITY_DATABASE_URL", "")
    if not value:
        for line in Path(config_path).read_text().splitlines():
            normalized = line.strip()
            if normalized.startswith("export "):
                normalized = normalized[7:]
            if normalized.startswith("MATERIAL_QUALITY_DATABASE_URL="):
                parts = shlex.split(normalized.split("=", 1)[1])
                value = parts[0] if parts else ""
    if not value:
        raise ValueError("未配置训练数据只读连接")
    return value


def atomic_json(path: Path, value: dict) -> None:
    """原子写入进度，避免监控读取不完整结果。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)


def initialize_local(path: Path) -> sqlite3.Connection:
    """只将聚合指标写入本地，任务主键防止重试重复计数。"""
    connection = sqlite3.connect(str(path), timeout=60)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS baseline (
            material_id TEXT PRIMARY KEY, first_stat_date TEXT, asset_id INTEGER,
            label_status TEXT, label_gmv_7d REAL, delivery_day_count INTEGER,
            primary_industry TEXT
        );
        CREATE TABLE IF NOT EXISTS daily (
            plan_id INTEGER, material_id TEXT, stat_date TEXT,
            gmv REAL, cost REAL, orders REAL, send REAL, click REAL,
            PRIMARY KEY(plan_id, material_id, stat_date)
        );
        CREATE INDEX IF NOT EXISTS daily_material ON daily(material_id, stat_date);
        CREATE TABLE IF NOT EXISTS batches (
            lower_id INTEGER PRIMARY KEY, upper_id INTEGER, source_rows INTEGER
        );
        CREATE TABLE IF NOT EXISTS plans (id INTEGER PRIMARY KEY, industry TEXT);
        CREATE TABLE IF NOT EXISTS first_seen (material_id TEXT PRIMARY KEY, first_day TEXT);
    """)
    return connection


def build_candidates(connection: sqlite3.Connection, as_of: str) -> List[dict]:
    """保留历史首次日期，避免近期再次投放的老素材被误标为新素材。"""
    rows = connection.execute("""
        WITH first_dates AS (
            SELECT d.material_id,
                MIN(COALESCE(b.first_stat_date, d.stat_date),
                    COALESCE(h.first_day, d.stat_date), MIN(d.stat_date)) AS first_day
            FROM daily d LEFT JOIN baseline b ON b.material_id=d.material_id
            LEFT JOIN first_seen h ON h.material_id=d.material_id
            GROUP BY d.material_id
        )
        SELECT d.material_id, f.first_day, MAX(b.asset_id) AS asset_id,
            COUNT(DISTINCT d.stat_date) AS delivery_days,
            SUM(d.gmv) AS gmv, SUM(d.cost) AS cost, SUM(d.orders) AS orders,
            SUM(d.send) AS send, SUM(d.click) AS click,
            GROUP_CONCAT(DISTINCT d.plan_id) AS plan_ids
        FROM daily d INNER JOIN first_dates f ON f.material_id=d.material_id
        LEFT JOIN baseline b ON b.material_id=d.material_id
        WHERE d.stat_date>=f.first_day AND d.stat_date<date(f.first_day, '+7 days')
            AND f.first_day>='2026-08-12'
        GROUP BY d.material_id, f.first_day
    """).fetchall()
    industries = {row["id"]: row["industry"] for row in connection.execute("SELECT * FROM plans")}
    result = []
    for row in rows:
        item = dict(row)
        first = datetime.datetime.strptime(item["first_day"], "%Y-%m-%d").date()
        end = first + datetime.timedelta(days=7)
        status = "ready"
        if end.isoformat() > as_of:
            status = "window_open"
        elif item["gmv"] < 0:
            status = "invalid_negative_gmv"
        elif item["delivery_days"] < 2:
            status = "insufficient_delivery_days"
        plan_ids = sorted(int(value) for value in item["plan_ids"].split(","))
        dominant = connection.execute("""
            SELECT plan_id FROM daily WHERE material_id=? AND stat_date>=?
                AND stat_date<? GROUP BY plan_id
            ORDER BY SUM(gmv) DESC, SUM(cost) DESC, plan_id LIMIT 1
        """, (item["material_id"], item["first_day"], end.isoformat())).fetchone()
        result.append({
            "asset_id": item["asset_id"] or (1000000000 + len(result)),
            "material_id": item["material_id"], "cohort": "all_industries_v1",
            "first_stat_date": item["first_day"], "label_status": status,
            "label_gmv_7d": item["gmv"],
            "label_gmv_per_delivery_day": item["gmv"] / item["delivery_days"],
            "delivery_day_count": item["delivery_days"],
            "cost_7d": item["cost"], "orders_7d": item["orders"],
            "send_7d": item["send"], "click_7d": item["click"],
            "primary_industry": industries.get(dominant[0], "未知行业") or "未知行业",
            "plan_ids": plan_ids,
        })
    return result


def run(args: argparse.Namespace) -> None:
    """以有限并发读取固定主键范围，所有样本和状态均落在本地。"""
    from collections import Counter
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    connection = initialize_local(root / "source.sqlite3")
    engine = create_engine(load_database_url(args.config), pool_size=args.workers,
                           max_overflow=0, connect_args={"read_timeout": 45, "connect_timeout": 10})

    def query(sql: str, params=None):
        with engine.connect() as remote:
            remote.execute(text("SET SESSION MAX_EXECUTION_TIME=30000"))
            remote.execute(text("START TRANSACTION READ ONLY"))
            try:
                return [dict(row) for row in remote.execute(text(sql), params or {}).mappings()]
            finally:
                remote.rollback()

    manifest_path = root / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["as_of"] != args.as_of:
            raise ValueError("断点目录的标签截止日期不一致")
    else:
        maximum = query("SELECT MAX(id) AS value FROM materialhourstats")[0]["value"]
        manifest = {"as_of": args.as_of, "maximum_source_id": maximum,
                    "created_at": datetime.datetime.now().isoformat(), "workers": args.workers,
                    "recent_from": "2026-08-12", "complete": False}
        atomic_json(manifest_path, manifest)
    baseline = query("""
        SELECT s.material_id, s.first_stat_date, s.asset_id, s.label_status,
            s.label_gmv_7d, s.delivery_day_count, c.primary_industry
        FROM material_quality_sample s
        LEFT JOIN material_quality_sample_context c ON c.sample_id=s.id
        WHERE s.cohort='all_industries_v1'
    """)
    with connection:
        connection.executemany("INSERT OR REPLACE INTO baseline VALUES (?,?,?,?,?,?,?)", [
            (r["material_id"], str(r["first_stat_date"]), r["asset_id"], r["label_status"],
             r["label_gmv_7d"], r["delivery_day_count"], r["primary_industry"]) for r in baseline])
        connection.executemany("INSERT OR REPLACE INTO plans VALUES (?,?)", [
            (r["id"], r["industry"]) for r in query("SELECT id,industry FROM accountplan")])
    log_info("历史样本基线已保存，共{}条".format(len(baseline)))
    cursor = int(manifest["maximum_source_id"]) + 1
    tasks = []
    while cursor > 0:
        bound = query("""SELECT MIN(id) AS lower_id,COUNT(*) AS source_rows
            FROM (SELECT id FROM materialhourstats FORCE INDEX(PRIMARY)
                WHERE id<:cursor AND id<=:maximum ORDER BY id DESC LIMIT 25000) t
        """, {"cursor": cursor, "maximum": manifest["maximum_source_id"]})[0]
        if bound["lower_id"] is None:
            break
        low = int(bound["lower_id"])
        tasks.append((low, cursor, int(bound["source_rows"])))
        cursor = low
    completed = {r[0] for r in connection.execute("SELECT lower_id FROM batches")}
    pending = [task for task in tasks if task[0] not in completed]
    log_info("冻结明细范围已切成{}批，待导出{}批，并发{}".format(len(tasks), len(pending), args.workers))

    def export(task):
        low, high, count = task
        for attempt in range(3):
            try:
                rows = query("""
                    SELECT account_plan_id AS plan_id, material_id,
                        IF(date>='2026-08-12',date,'历史') AS stat_date, MIN(date) AS first_day,
                        SUM(COALESCE(gmv,0)) AS gmv,SUM(COALESCE(cost,0)) AS cost,
                        SUM(COALESCE(orders,0)) AS orders,SUM(COALESCE(send,0)) AS send,
                        SUM(COALESCE(click,0)) AS click
                    FROM materialhourstats FORCE INDEX(PRIMARY)
                    WHERE id>=:low AND id<:high AND date<:as_of
                        AND material_type='VIDEO' AND material_id<>''
                        AND (COALESCE(gmv,0)<>0 OR COALESCE(cost,0)<>0
                            OR COALESCE(orders,0)<>0 OR COALESCE(send,0)<>0 OR COALESCE(click,0)<>0)
                    GROUP BY account_plan_id,material_id,IF(date>='2026-08-12',date,'历史')
                """, {"low": low, "high": high, "as_of": args.as_of})
                return task, rows
            except Exception:
                if attempt == 2:
                    raise RuntimeError("明细分段读取失败，范围{}至{}".format(low, high))
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (task, rows) in enumerate(pool.map(export, pending)):
            with connection:
                connection.executemany("""
                    INSERT INTO first_seen VALUES (?,?) ON CONFLICT(material_id)
                    DO UPDATE SET first_day=MIN(first_day,excluded.first_day)
                """, [(r["material_id"], r["first_day"]) for r in rows])
                connection.executemany("""
                    INSERT INTO daily VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(plan_id,material_id,stat_date) DO UPDATE SET
                        gmv=gmv+excluded.gmv,cost=cost+excluded.cost,orders=orders+excluded.orders,
                        send=send+excluded.send,click=click+excluded.click
                """, [(r["plan_id"],r["material_id"],r["stat_date"],r["gmv"],r["cost"],
                       r["orders"],r["send"],r["click"]) for r in rows if r["stat_date"] != "历史"])
                connection.execute("INSERT INTO batches VALUES (?,?,?)", task)
            if index % 20 == 0 or index == len(pending)-1:
                progress = {"phase": "export", "completed_batches": len(completed)+index+1,
                            "total_batches": len(tasks), "updated_at": datetime.datetime.now().isoformat()}
                atomic_json(root / "status.json", progress)
                log_info("明细导出进度：{}/{}批".format(progress["completed_batches"],len(tasks)))
    candidates = build_candidates(connection, args.as_of)
    output = root / "candidates.jsonl"
    with output.open("w") as stream:
        for target in candidates:
            stream.write(json.dumps(target,ensure_ascii=False)+"\n")
    manifest.update({"complete": True,"candidate_count":len(candidates),
                     "statuses":dict(Counter(t["label_status"] for t in candidates)),
                     "months":dict(Counter(t["first_stat_date"][:7] for t in candidates if t["label_status"]=="ready")),
                     "ready_positive":sum(t["label_status"]=="ready" and t["label_gmv_7d"]>0 for t in candidates),
                     "candidate_sha256":hashlib.sha256(output.read_bytes()).hexdigest()})
    atomic_json(manifest_path, manifest)
    atomic_json(root / "status.json", {"phase":"samples_ready",**manifest})
    log_info("本地样本构造完成：{}".format(json.dumps(manifest,ensure_ascii=False)))


def main() -> None:
    """读取本次采集配置并启动可恢复导出。"""
    set_process_name("material_quality_refresh")
    parser = argparse.ArgumentParser(description="只读刷新训练样本")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    datetime.datetime.strptime(args.as_of, "%Y-%m-%d")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("数据库读取并发必须在1至8之间")
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""从生产日汇总只读刷新本地素材档案标签。"""

import argparse
import datetime
import glob
import json
import os
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import bindparam, text

from material_quality.database import engine
from material_quality.feature.local_artifact_store import LocalRawFeatureStore
from material_quality.feature.local_task_state import LocalTaskState
from material_quality.sample.service import build_samples_from_daily_rows
from plan_logger import log_info, log_warn, set_process_name


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """把素材ID分批，控制只读查询参数数量。"""
    for start in range(0, len(values), size):
        yield values[start:start + size]


def effective_label_date(
    requested_date: datetime.date,
    maximum_stat_date: datetime.date,
) -> datetime.date:
    """标签截止日不能超过日汇总已完整覆盖的下一天。"""
    source_limit = maximum_stat_date + datetime.timedelta(days=1)
    return min(requested_date, source_limit)


def _load_asset_references(
    raw_feature_root: str,
    feature_version: str,
    only_window_open: bool,
) -> List[dict]:
    """读取本地素材引用，内容档案和标签引用保持分离。"""
    pattern = os.path.join(
        os.path.realpath(raw_feature_root),
        feature_version,
        "assets",
        "*.json",
    )
    references = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as source:
                reference = json.load(source)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("本地素材引用读取失败：{}，原因：{}".format(
                path,
                exc,
            ))
        if not isinstance(reference, dict):
            raise ValueError("本地素材引用必须是JSON对象：{}".format(path))
        if only_window_open and reference.get("label_status") != "window_open":
            continue
        references.append(reference)
    return references


def _load_daily_rows(
    cohort: str,
    material_ids: Sequence[str],
    batch_size: int,
) -> List[dict]:
    """利用范围表的素材索引分批读取日汇总，不扫描小时明细表。"""
    query = text("""
        SELECT
            account_plan_id,
            material_id,
            stat_date,
            gmv,
            cost,
            orders,
            send,
            click
        FROM material_quality_daily_stat
        WHERE cohort = :cohort
          AND material_id IN :material_ids
    """).bindparams(bindparam("material_ids", expanding=True))
    rows = []
    with engine.connect() as connection:
        for batch in _chunks(material_ids, batch_size):
            batch_rows = connection.execute(query, {
                "cohort": cohort,
                "material_ids": list(batch),
            }).mappings().all()
            rows.extend(dict(row) for row in batch_rows)
    return rows


def _load_maximum_stat_date(cohort: str) -> datetime.date:
    """读取日汇总当前完整覆盖日期。"""
    with engine.connect() as connection:
        value = connection.execute(text("""
            SELECT MAX(stat_date)
            FROM material_quality_daily_stat
            WHERE cohort = :cohort
        """), {"cohort": cohort}).scalar()
    if not isinstance(value, datetime.date):
        raise RuntimeError("素材日汇总没有可用日期")
    return value


def refresh_local_labels(args: argparse.Namespace) -> dict:
    """只更新本地素材引用和SQLite状态，不写生产数据库。"""
    references = _load_asset_references(
        args.raw_feature_root,
        args.feature_version,
        args.only_window_open,
    )
    if not references:
        return {
            "reference_count": 0,
            "updated_count": 0,
            "new_ready_count": 0,
        }
    references_by_material = {
        str(reference["material_id"]): reference
        for reference in references
    }
    maximum_stat_date = _load_maximum_stat_date(args.cohort)
    effective_date = effective_label_date(args.as_of_date, maximum_stat_date)
    if effective_date < args.as_of_date:
        log_warn(
            "请求截止日{}超过日汇总覆盖范围，安全收缩到{}".format(
                args.as_of_date.isoformat(),
                effective_date.isoformat(),
            )
        )
    daily_rows = _load_daily_rows(
        args.cohort,
        sorted(references_by_material),
        args.batch_size,
    )
    samples = build_samples_from_daily_rows(
        daily_rows,
        effective_date,
        window_days=7,
    )
    sample_by_material = {
        str(sample["material_id"]): sample for sample in samples
    }
    store = LocalRawFeatureStore(args.raw_feature_root, args.feature_version)
    local_state = LocalTaskState(args.local_state_db)
    updated_targets = []
    new_ready_count = 0
    status_counts: Dict[str, int] = {}
    missing_count = 0
    for material_id, reference in references_by_material.items():
        sample = sample_by_material.get(material_id)
        if sample is None:
            missing_count += 1
            continue
        previous_status = str(reference.get("label_status") or "")
        current_status = str(sample["label_status"])
        if previous_status != "ready" and current_status == "ready":
            new_ready_count += 1
        status_counts[current_status] = status_counts.get(current_status, 0) + 1
        target = dict(reference)
        target.update(sample)
        target["asset_id"] = int(reference["asset_id"])
        target["cohort"] = args.cohort
        target["primary_industry"] = reference.get("primary_industry")
        target["industry_count"] = reference.get("industry_count")
        updated_targets.append(target)
        if not args.dry_run:
            store.save_asset_reference(
                target,
                str(reference["content_sha256"]),
            )
    if updated_targets and not args.dry_run:
        local_state.upsert_targets(
            updated_targets,
            args.feature_version,
            args.cohort,
        )
    result = {
        "reference_count": len(references),
        "daily_row_count": len(daily_rows),
        "updated_count": len(updated_targets),
        "missing_count": missing_count,
        "new_ready_count": new_ready_count,
        "status_counts": status_counts,
        "maximum_stat_date": maximum_stat_date.isoformat(),
        "effective_as_of_date": effective_date.isoformat(),
        "dry_run": bool(args.dry_run),
    }
    log_info("本地素材标签刷新完成：{}".format(result))
    return result


def _parse_date(value: str) -> datetime.date:
    """解析截止日期。"""
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD格式")


def parse_args() -> argparse.Namespace:
    """读取本地标签刷新参数。"""
    parser = argparse.ArgumentParser(description="只读刷新本地素材标签")
    parser.add_argument("--raw-feature-root", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--local-state-db", required=True)
    parser.add_argument("--cohort", default="all_industries_v1")
    parser.add_argument(
        "--as-of-date",
        type=_parse_date,
        default=datetime.date.today(),
    )
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument(
        "--all-label-statuses",
        action="store_true",
        help="刷新所有本地素材，默认只刷新未闭合窗口",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.only_window_open = not args.all_label_statuses
    return args


def main() -> int:
    """启动本地标签刷新。"""
    set_process_name("material_quality_label_refresh")
    args = parse_args()
    if args.batch_size <= 0:
        log_warn("查询批次大小必须大于0")
        return 2
    refresh_local_labels(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

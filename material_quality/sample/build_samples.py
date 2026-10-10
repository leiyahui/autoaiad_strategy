#!/usr/bin/env python3
"""分批构造视频素材的7日GMV训练样本。"""

import argparse
import datetime
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence

from sqlalchemy import text

from material_quality.database import engine
from material_quality.sample.service import (
    choose_asset_metadata,
    is_food_industry,
    validate_cohort_scope,
)
from plan_logger import log_error, log_info, log_warn, set_process_name


COHORT = "food_v1"
LABEL_VERSION = "gmv_per_delivery_day_7d_v2"
WINDOW_DAYS = 7
LONG_QUERY_TIMEOUT_SECONDS = 600


def _chunks(items: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    """按固定条数切分批次，避免单次写入参数过多。"""
    for start in range(0, len(items), size):
        yield items[start:start + size]


@contextmanager
def _long_transaction():
    """仅为离线全量聚合延长当前连接超时，不修改线上数据库连接配置。"""
    with engine.connect() as conn:
        driver_connection = getattr(conn.connection, "driver_connection", None)
        if driver_connection is None:
            driver_connection = getattr(conn.connection, "connection", None)
        if driver_connection is not None:
            driver_connection._read_timeout = LONG_QUERY_TIMEOUT_SECONDS
            driver_connection._write_timeout = LONG_QUERY_TIMEOUT_SECONDS
        with conn.begin():
            yield conn


def _verify_schema() -> None:
    """检查样本表是否已通过SQL迁移创建。"""
    required_tables = {
        "material_quality_daily_stat",
        "material_quality_asset",
        "material_quality_sample",
        "material_quality_sample_context",
    }
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME IN "
            "('material_quality_daily_stat', 'material_quality_asset', "
            "'material_quality_sample', 'material_quality_sample_context')"
        )).fetchall()
    existing = {str(row[0]) for row in rows}
    missing = required_tables - existing
    if missing:
        raise RuntimeError(
            "素材质量样本表尚未创建，请先执行 material_quality/sql/"
            "add_material_quality_sample_tables.sql；缺少：{}".format(
                ", ".join(sorted(missing))
            )
        )


def _load_target_plans(
    explicit_plan_ids: Optional[List[int]],
    include_all_industries: bool,
    plan_limit: int,
    resume_after_plan_id: int,
) -> List[dict]:
    """读取目标计划，默认保留食品行业，也支持独立全行业范围。"""
    sql = "SELECT id, industry FROM accountplan"
    params: Dict[str, object] = {}
    if explicit_plan_ids:
        placeholders = []
        for index, plan_id in enumerate(explicit_plan_ids):
            key = "plan_id_{}".format(index)
            placeholders.append(":" + key)
            params[key] = plan_id
        sql += " WHERE id IN ({})".format(",".join(placeholders))
    sql += " ORDER BY id"

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()

    plans = []
    for row in rows:
        plan_id = int(row["id"])
        industry = str(row.get("industry") or "").strip()
        if plan_id <= resume_after_plan_id:
            continue
        if not include_all_industries and not is_food_industry(industry):
            continue
        plans.append({"id": plan_id, "industry": industry})
        if plan_limit > 0 and len(plans) >= plan_limit:
            break
    return plans


def _refresh_plan_daily_stats(plan_id: int, cohort: str) -> int:
    """按计划重建视频素材日汇总，单计划事务保证断点安全。"""
    insert_sql = text("""
        INSERT INTO material_quality_daily_stat (
            cohort, account_plan_id, material_id, stat_date,
            gmv, cost, orders, send, click, source_row_count, update_time
        )
        SELECT
            :cohort,
            account_plan_id,
            material_id,
            STR_TO_DATE(date, '%Y-%m-%d'),
            SUM(COALESCE(gmv, 0)),
            SUM(COALESCE(cost, 0)),
            SUM(COALESCE(orders, 0)),
            SUM(COALESCE(send, 0)),
            SUM(COALESCE(click, 0)),
            COUNT(*),
            NOW()
        FROM materialhourstats FORCE INDEX (account_plan_id)
        WHERE account_plan_id = :plan_id
          AND material_type = 'VIDEO'
          AND material_id IS NOT NULL
          AND material_id <> ''
          AND (
              COALESCE(gmv, 0) <> 0
              OR COALESCE(cost, 0) <> 0
              OR COALESCE(orders, 0) <> 0
              OR COALESCE(send, 0) <> 0
              OR COALESCE(click, 0) <> 0
          )
        GROUP BY account_plan_id, material_id, date
    """)
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM material_quality_daily_stat "
            "WHERE cohort = :cohort AND account_plan_id = :plan_id"
        ), {"cohort": cohort, "plan_id": plan_id})
        result = conn.execute(insert_sql, {"cohort": cohort, "plan_id": plan_id})
    return int(result.rowcount or 0)


def _ensure_asset_placeholders(cohort: str) -> int:
    """为日汇总中发现的素材补齐媒体资产占位记录。"""
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT IGNORE INTO material_quality_asset (
                material_id, source_status, download_status, feature_status,
                create_time, update_time
            )
            SELECT DISTINCT
                material_id, 'pending_url', 'pending', 'pending', NOW(), NOW()
            FROM material_quality_daily_stat
            WHERE cohort = :cohort
        """), {"cohort": cohort})
    return int(result.rowcount or 0)


def _load_known_asset_metadata(cohort: str) -> Dict[str, dict]:
    """读取样本范围内素材在现有素材库中的最佳媒体记录。"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                m.id, m.user_id, m.material_id, m.name, m.advertiser_id,
                m.upload_time, m.video_url, m.thumbnail_url,
                m.duration, m.file_size
            FROM materialforedit AS m
            INNER JOIN material_quality_asset AS a
                ON a.material_id = m.material_id
            INNER JOIN (
                SELECT DISTINCT material_id
                FROM material_quality_daily_stat
                WHERE cohort = :cohort
            ) AS d ON d.material_id = m.material_id
        """), {"cohort": cohort}).mappings().all()
    return choose_asset_metadata(rows)


def _sync_asset_metadata(metadata: Dict[str, dict], batch_size: int = 500) -> int:
    """将已有URL和媒体信息同步到样本资产表。"""
    if not metadata:
        return 0
    payload = []
    for material_id, row in metadata.items():
        payload.append({
            "material_id": material_id,
            "material_for_edit_id": row.get("id"),
            "user_id": row.get("user_id"),
            "advertiser_id": row.get("advertiser_id"),
            "material_name": str(row.get("name") or "")[:255],
            "video_url": row.get("video_url"),
            "thumbnail_url": row.get("thumbnail_url"),
            "duration": float(row.get("duration") or 0),
            "file_size": int(row.get("file_size") or 0),
            "source_status": "url_ready" if row.get("video_url") else "pending_url",
        })

    update_sql = text("""
        UPDATE material_quality_asset
        SET material_for_edit_id = :material_for_edit_id,
            user_id = :user_id,
            advertiser_id = :advertiser_id,
            material_name = :material_name,
            video_url = :video_url,
            thumbnail_url = :thumbnail_url,
            duration = :duration,
            file_size = :file_size,
            source_status = :source_status,
            update_time = NOW()
        WHERE material_id = :material_id
    """)
    updated = 0
    with engine.begin() as conn:
        for batch in _chunks(payload, batch_size):
            result = conn.execute(update_sql, list(batch))
            updated += max(int(result.rowcount or 0), 0)
    return updated


def _rebuild_samples(cohort: str, as_of_date: datetime.date) -> int:
    """从日汇总表生成一条素材一条记录的7日GMV标签。"""
    sample_sql = text("""
        INSERT INTO material_quality_sample (
            asset_id, cohort, material_id, label_version, window_days,
            first_stat_date, window_end_date,
            label_gmv_7d, label_log_gmv_7d,
            label_gmv_per_delivery_day, label_log_gmv_per_delivery_day,
            cost_7d, orders_7d, send_7d, click_7d,
            plan_count, source_day_count, delivery_day_count,
            is_window_closed, label_status, create_time, update_time
        )
        SELECT
            a.id,
            :cohort,
            d.material_id,
            :label_version,
            7,
            f.first_stat_date,
            DATE_ADD(f.first_stat_date, INTERVAL 7 DAY),
            SUM(d.gmv),
            LN(1 + GREATEST(SUM(d.gmv), 0)),
            SUM(d.gmv) / GREATEST(COUNT(DISTINCT d.stat_date), 1),
            LN(1 + GREATEST(
                SUM(d.gmv) / GREATEST(COUNT(DISTINCT d.stat_date), 1),
                0
            )),
            SUM(d.cost),
            SUM(d.orders),
            SUM(d.send),
            SUM(d.click),
            COUNT(DISTINCT d.account_plan_id),
            COUNT(*),
            COUNT(DISTINCT d.stat_date),
            CASE
                WHEN DATE_ADD(f.first_stat_date, INTERVAL 7 DAY) <= :as_of_date THEN 1
                ELSE 0
            END,
            CASE
                WHEN DATE_ADD(f.first_stat_date, INTERVAL 7 DAY) > :as_of_date THEN 'window_open'
                WHEN SUM(d.gmv) < 0 THEN 'invalid_negative_gmv'
                WHEN COUNT(DISTINCT d.stat_date) < 2 THEN 'insufficient_delivery_days'
                ELSE 'ready'
            END,
            NOW(),
            NOW()
        FROM material_quality_daily_stat AS d
        INNER JOIN (
            SELECT material_id, MIN(stat_date) AS first_stat_date
            FROM material_quality_daily_stat
            WHERE cohort = :cohort
            GROUP BY material_id
        ) AS f ON f.material_id = d.material_id
        INNER JOIN material_quality_asset AS a ON a.material_id = d.material_id
        WHERE d.cohort = :cohort
          AND d.stat_date >= f.first_stat_date
          AND d.stat_date < DATE_ADD(f.first_stat_date, INTERVAL 7 DAY)
        GROUP BY a.id, d.material_id, f.first_stat_date
        ON DUPLICATE KEY UPDATE
            asset_id = VALUES(asset_id),
            window_days = VALUES(window_days),
            first_stat_date = VALUES(first_stat_date),
            window_end_date = VALUES(window_end_date),
            label_gmv_7d = VALUES(label_gmv_7d),
            label_log_gmv_7d = VALUES(label_log_gmv_7d),
            label_gmv_per_delivery_day = VALUES(label_gmv_per_delivery_day),
            label_log_gmv_per_delivery_day = VALUES(label_log_gmv_per_delivery_day),
            cost_7d = VALUES(cost_7d),
            orders_7d = VALUES(orders_7d),
            send_7d = VALUES(send_7d),
            click_7d = VALUES(click_7d),
            plan_count = VALUES(plan_count),
            source_day_count = VALUES(source_day_count),
            delivery_day_count = VALUES(delivery_day_count),
            is_window_closed = VALUES(is_window_closed),
            label_status = VALUES(label_status),
            update_time = NOW()
    """)
    params = {
        "cohort": cohort,
        "label_version": LABEL_VERSION,
        "as_of_date": as_of_date,
    }
    with _long_transaction() as conn:
        result = conn.execute(sample_sql, params)
        conn.execute(text("""
            DELETE s
            FROM material_quality_sample AS s
            LEFT JOIN material_quality_daily_stat AS d
              ON d.cohort = s.cohort AND d.material_id = s.material_id
            WHERE s.cohort = :cohort
              AND s.label_version = :label_version
              AND d.id IS NULL
        """), params)
    return max(int(result.rowcount or 0), 0)


def _rebuild_sample_context(cohort: str) -> int:
    """为每条样本选择贡献最高的主行业，并记录跨行业数量。"""
    context_sql = text("""
        INSERT INTO material_quality_sample_context (
            sample_id, cohort, material_id, label_version,
            primary_industry, industry_count, create_time, update_time
        )
        SELECT
            s.id,
            s.cohort,
            s.material_id,
            s.label_version,
            COALESCE(MAX(CASE WHEN ranked.industry_rank = 1
                THEN ranked.primary_industry END), '未标注'),
            COALESCE(MAX(ranked.industry_count), 0),
            NOW(),
            NOW()
        FROM material_quality_sample AS s
        LEFT JOIN (
            SELECT
                grouped.material_id,
                grouped.primary_industry,
                ROW_NUMBER() OVER (
                    PARTITION BY grouped.material_id
                    ORDER BY
                        grouped.priority_gmv DESC,
                        grouped.priority_cost DESC,
                        grouped.plan_count DESC,
                        grouped.primary_industry
                ) AS industry_rank,
                COUNT(*) OVER (
                    PARTITION BY grouped.material_id
                ) AS industry_count
            FROM (
                SELECT
                    d.material_id,
                    COALESCE(NULLIF(TRIM(p.industry), ''), '未标注') AS primary_industry,
                    SUM(ABS(d.gmv)) AS priority_gmv,
                    SUM(ABS(d.cost)) AS priority_cost,
                    COUNT(DISTINCT d.account_plan_id) AS plan_count
                FROM material_quality_daily_stat AS d
                INNER JOIN accountplan AS p ON p.id = d.account_plan_id
                WHERE d.cohort = :cohort
                GROUP BY
                    d.material_id,
                    COALESCE(NULLIF(TRIM(p.industry), ''), '未标注')
            ) AS grouped
        ) AS ranked ON ranked.material_id = s.material_id
        WHERE s.cohort = :cohort
          AND s.label_version = :label_version
        GROUP BY s.id, s.cohort, s.material_id, s.label_version
        ON DUPLICATE KEY UPDATE
            primary_industry = VALUES(primary_industry),
            industry_count = VALUES(industry_count),
            update_time = NOW()
    """)
    params = {"cohort": cohort, "label_version": LABEL_VERSION}
    with _long_transaction() as conn:
        result = conn.execute(context_sql, params)
        conn.execute(text("""
            DELETE c
            FROM material_quality_sample_context AS c
            LEFT JOIN material_quality_sample AS s ON s.id = c.sample_id
            WHERE c.cohort = :cohort
              AND c.label_version = :label_version
              AND s.id IS NULL
        """), params)
    return max(int(result.rowcount or 0), 0)


def _load_summary(cohort: str) -> dict:
    """读取当前样本数量与媒体覆盖率。"""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT
                COUNT(*) AS sample_count,
                SUM(s.label_status = 'ready') AS ready_count,
                SUM(s.label_status = 'window_open') AS open_count,
                SUM(s.label_status = 'invalid_negative_gmv') AS invalid_negative_gmv_count,
                SUM(s.label_status = 'insufficient_delivery_days') AS insufficient_delivery_days_count,
                SUM(a.source_status = 'url_ready') AS url_ready_count,
                SUM(s.label_gmv_7d > 0) AS positive_gmv_count,
                COALESCE(SUM(s.label_gmv_7d), 0) AS total_gmv
            FROM material_quality_sample AS s
            INNER JOIN material_quality_asset AS a ON a.id = s.asset_id
            WHERE s.cohort = :cohort AND s.label_version = :label_version
        """), {
            "cohort": cohort,
            "label_version": LABEL_VERSION,
        }).mappings().first()
    return dict(row or {})


def build_samples(args) -> dict:
    """执行可断点的样本构造流程。"""
    _verify_schema()
    validate_cohort_scope(args.cohort, args.all_industries)
    plans = _load_target_plans(
        explicit_plan_ids=args.plan_id,
        include_all_industries=args.all_industries,
        plan_limit=args.plan_limit,
        resume_after_plan_id=args.resume_after_plan_id,
    )
    log_info(
        "样本范围确认 | cohort:{} | 计划数:{} | 截止日期:{}".format(
            args.cohort, len(plans), args.as_of_date
        )
    )
    if args.dry_run:
        for plan in plans[:20]:
            log_info("预览计划 | id:{} | 行业:{}".format(plan["id"], plan["industry"]))
        return {"plan_count": len(plans), "dry_run": True}

    if not args.skip_daily_refresh:
        total_daily_rows = 0
        for index, plan in enumerate(plans, 1):
            start_time = time.time()
            try:
                row_count = _refresh_plan_daily_stats(plan["id"], args.cohort)
                total_daily_rows += row_count
                log_info(
                    "计划日汇总完成 | {}/{} | plan_id:{} | 行业:{} | 日记录:{} | 耗时:{:.1f}秒".format(
                        index,
                        len(plans),
                        plan["id"],
                        plan["industry"],
                        row_count,
                        time.time() - start_time,
                    )
                )
            except Exception as exc:
                log_error(
                    "计划日汇总失败 | plan_id:{} | 原因:{}".format(plan["id"], exc)
                )
                if not args.continue_on_error:
                    raise
        log_info("全部计划日汇总完成 | 写入记录:{}".format(total_daily_rows))
    elif not plans:
        log_warn("已跳过日汇总刷新，将直接使用现有中间表生成样本")

    created_assets = _ensure_asset_placeholders(args.cohort)
    metadata = _load_known_asset_metadata(args.cohort)
    updated_assets = _sync_asset_metadata(metadata)
    affected_samples = _rebuild_samples(args.cohort, args.as_of_date)
    affected_contexts = _rebuild_sample_context(args.cohort)
    summary = _load_summary(args.cohort)
    summary.update({
        "created_assets": created_assets,
        "updated_assets": updated_assets,
        "affected_samples": affected_samples,
        "affected_contexts": affected_contexts,
    })
    log_info("素材质量样本构造完成 | {}".format(summary))
    return summary


def _parse_args(argv: Optional[List[str]] = None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="构造素材首次投放后7日GMV训练样本")
    parser.add_argument("--cohort", default=COHORT, help="样本范围版本")
    parser.add_argument("--plan-id", action="append", type=int, help="仅处理指定内部计划ID，可重复传入")
    parser.add_argument("--plan-limit", type=int, default=0, help="最多处理多少个计划，0表示不限制")
    parser.add_argument("--resume-after-plan-id", type=int, default=0, help="跳过不大于该ID的计划")
    parser.add_argument(
        "--all-industries",
        action="store_true",
        help="不限制行业，必须配合独立样本范围版本",
    )
    parser.add_argument("--skip-daily-refresh", action="store_true", help="跳过计划日汇总，直接重建样本")
    parser.add_argument("--continue-on-error", action="store_true", help="单计划失败后继续处理后续计划")
    parser.add_argument("--dry-run", action="store_true", help="只预览目标计划，不写数据库")
    parser.add_argument(
        "--as-of-date",
        type=lambda value: datetime.datetime.strptime(value, "%Y-%m-%d").date(),
        default=datetime.date.today(),
        help="判断7日窗口是否闭合的截止日期，格式YYYY-MM-DD",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    set_process_name("material_quality_samples")
    args = _parse_args(argv)
    try:
        build_samples(args)
        return 0
    except Exception as exc:
        log_error("素材质量样本构造失败，原因：{}".format(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())

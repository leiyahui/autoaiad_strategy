#!/usr/bin/env python3
"""审计素材质量样本规模、行业覆盖、月份分布和上下文完整性。"""

import argparse
import json
from typing import Dict, List

from sqlalchemy import text

from plan_logger import log_info, set_process_name

from material_quality.database import engine


def _rows_to_dicts(rows) -> List[Dict[str, object]]:
    """将数据库映射行转为可序列化字典。"""
    return [dict(row) for row in rows]


def audit_samples(cohort: str, top_industries: int) -> dict:
    """返回指定范围的样本总览和行业、月份分布。"""
    params = {"cohort": cohort, "top_industries": top_industries}
    with engine.connect() as conn:
        summary = conn.execute(text("""
            SELECT
                COUNT(*) AS sample_count,
                SUM(s.label_status = 'ready') AS ready_count,
                SUM(s.label_status = 'window_open') AS open_count,
                SUM(s.label_status = 'invalid_negative_gmv') AS invalid_count,
                SUM(s.label_status = 'insufficient_delivery_days') AS insufficient_delivery_days_count,
                SUM(s.label_status = 'ready' AND s.label_gmv_7d > 0) AS ready_positive_count,
                SUM(s.label_status = 'ready' AND s.label_gmv_7d = 0) AS ready_zero_count,
                COUNT(DISTINCT c.primary_industry) AS industry_count,
                SUM(c.sample_id IS NULL) AS missing_context_count,
                SUM(a.source_status = 'url_ready') AS url_ready_count,
                SUM(a.source_status = 'unavailable') AS url_unavailable_count
            FROM material_quality_sample AS s
            INNER JOIN material_quality_asset AS a ON a.id = s.asset_id
            LEFT JOIN material_quality_sample_context AS c ON c.sample_id = s.id
            WHERE s.cohort = :cohort
        """), params).mappings().first()
        industries = conn.execute(text("""
            SELECT
                c.primary_industry,
                COUNT(*) AS sample_count,
                SUM(s.label_status = 'ready') AS ready_count,
                SUM(s.label_status = 'ready' AND s.label_gmv_7d > 0) AS positive_count,
                AVG(CASE WHEN s.label_status = 'ready'
                    THEN s.label_gmv_per_delivery_day END) AS avg_gmv_per_delivery_day
            FROM material_quality_sample AS s
            INNER JOIN material_quality_sample_context AS c ON c.sample_id = s.id
            WHERE s.cohort = :cohort
            GROUP BY c.primary_industry
            ORDER BY sample_count DESC
            LIMIT :top_industries
        """), params).mappings().all()
        months = conn.execute(text("""
            SELECT
                DATE_FORMAT(s.first_stat_date, '%Y-%m') AS first_stat_month,
                COUNT(*) AS sample_count,
                SUM(s.label_status = 'ready') AS ready_count,
                SUM(s.label_status = 'ready' AND s.label_gmv_7d > 0) AS positive_count,
                AVG(CASE WHEN s.label_status = 'ready'
                    THEN s.label_gmv_per_delivery_day END) AS avg_gmv_per_delivery_day
            FROM material_quality_sample AS s
            WHERE s.cohort = :cohort
            GROUP BY DATE_FORMAT(s.first_stat_date, '%Y-%m')
            ORDER BY first_stat_month
        """), params).mappings().all()
        safeguards = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM material_quality_sample
                 WHERE cohort = 'food_v1') AS food_sample_count,
                (SELECT COUNT(*) FROM material_quality_feature
                 WHERE feature_version = 'media_visual_stats_v1') AS legacy_feature_count,
                (SELECT COUNT(*) FROM material_quality_semantic_feature) AS semantic_feature_count
        """)).mappings().first()
    return {
        "cohort": cohort,
        "summary": dict(summary or {}),
        "top_industries": _rows_to_dicts(industries),
        "months": _rows_to_dicts(months),
        "safeguards": dict(safeguards or {}),
    }


def parse_args() -> argparse.Namespace:
    """读取样本审计参数。"""
    parser = argparse.ArgumentParser(description="审计素材质量样本")
    parser.add_argument("--cohort", default="all_industries_v1", help="样本范围版本")
    parser.add_argument("--top-industries", type=int, default=20, help="展示行业数")
    return parser.parse_args()


def main() -> int:
    """样本审计入口。"""
    set_process_name("material_quality_samples")
    args = parse_args()
    result = audit_samples(args.cohort, args.top_industries)
    log_info("样本审计结果：{}".format(json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

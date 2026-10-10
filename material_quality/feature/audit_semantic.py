#!/usr/bin/env python3
"""审计多模态语义特征的维度、行业覆盖、耗时和任务状态。"""

import argparse
import json

from sqlalchemy import text

from plan_logger import log_info, set_process_name

from material_quality.database import engine


def audit_semantic_features(cohort: str, feature_version: str) -> dict:
    """返回指定版本的特征完整性和跨行业覆盖信息。"""
    params = {"cohort": cohort, "feature_version": feature_version}
    with engine.connect() as conn:
        summary = conn.execute(text("""
            SELECT
                COUNT(*) AS feature_count,
                COUNT(DISTINCT f.content_sha256) AS distinct_content_count,
                MIN(f.visual_vector_dim) AS min_visual_dim,
                MAX(f.visual_vector_dim) AS max_visual_dim,
                MIN(f.visual_temporal_dim) AS min_visual_temporal_dim,
                MAX(f.visual_temporal_dim) AS max_visual_temporal_dim,
                SUM(f.text_vector_dim > 0) AS text_vector_count,
                SUM(CHAR_LENGTH(COALESCE(f.ocr_text, '')) > 0) AS ocr_text_count,
                MIN(f.ocr_stats_dim) AS min_ocr_stats_dim,
                MAX(f.ocr_stats_dim) AS max_ocr_stats_dim,
                SUM(CHAR_LENGTH(COALESCE(f.asr_text, '')) > 0) AS asr_text_count,
                SUM(f.asr_text_vector_dim > 0) AS asr_text_vector_count,
                MIN(f.audio_stats_dim) AS min_audio_stats_dim,
                MAX(f.audio_stats_dim) AS max_audio_stats_dim,
                MIN(f.sample_frame_count) AS min_frame_count,
                MAX(f.sample_frame_count) AS max_frame_count,
                AVG(f.extraction_seconds) AS avg_extraction_seconds,
                MAX(f.extraction_seconds) AS max_extraction_seconds,
                SUM(s.label_gmv_7d > 0) AS positive_gmv_count,
                COUNT(DISTINCT c.primary_industry) AS industry_count
            FROM material_quality_semantic_feature AS f
            INNER JOIN material_quality_sample AS s
              ON s.asset_id = f.asset_id AND s.cohort = :cohort
            INNER JOIN material_quality_sample_context AS c ON c.sample_id = s.id
            WHERE f.feature_version = :feature_version
        """), params).mappings().first()
        industries = conn.execute(text("""
            SELECT
                c.primary_industry,
                COUNT(*) AS feature_count,
                SUM(s.label_gmv_7d > 0) AS positive_gmv_count,
                AVG(f.extraction_seconds) AS avg_extraction_seconds
            FROM material_quality_semantic_feature AS f
            INNER JOIN material_quality_sample AS s
              ON s.asset_id = f.asset_id AND s.cohort = :cohort
            INNER JOIN material_quality_sample_context AS c ON c.sample_id = s.id
            WHERE f.feature_version = :feature_version
            GROUP BY c.primary_industry
            ORDER BY c.primary_industry
        """), params).mappings().all()
        task_statuses = conn.execute(text("""
            SELECT status, COUNT(*) AS task_count
            FROM material_quality_semantic_task
            WHERE feature_version = :feature_version
            GROUP BY status
            ORDER BY status
        """), params).mappings().all()
    return {
        "cohort": cohort,
        "feature_version": feature_version,
        "summary": dict(summary or {}),
        "industries": [dict(row) for row in industries],
        "task_statuses": [dict(row) for row in task_statuses],
    }


def parse_args() -> argparse.Namespace:
    """读取语义特征审计参数。"""
    parser = argparse.ArgumentParser(description="审计素材质量语义特征")
    parser.add_argument("--cohort", default="all_industries_v1", help="样本范围版本")
    parser.add_argument(
        "--feature-version",
        default="multimodal_cnclip_whisper_small_v2",
        help="语义特征版本",
    )
    return parser.parse_args()


def main() -> int:
    """语义特征审计入口。"""
    set_process_name("material_quality_semantic")
    args = parse_args()
    result = audit_semantic_features(args.cohort, args.feature_version)
    log_info("语义特征审计结果：{}".format(json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

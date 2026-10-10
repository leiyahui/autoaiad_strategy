#!/usr/bin/env python3
"""执行素材URL刷新、临时下载、特征落库和文件清理任务。"""

import argparse
import datetime
import json
import os
import tempfile
import time
from typing import Dict, List, Optional

from sqlalchemy import text

from material_quality.database import engine
from plan_logger import log_error, log_info, log_warn, set_process_name

from material_quality.feature.media_features import (
    download_video,
    extract_media_features,
)
from material_quality.feature.selection import build_first_date_filter, parse_date
from material_quality.feature.url_resolver import resolve_material_url


DEFAULT_COHORT = "food_v1"
DEFAULT_FEATURE_VERSION = "media_visual_stats_v1"
DEFAULT_TEMP_ROOT = "/tmp/material_quality_feature"


def _verify_schema() -> None:
    """检查样本表和特征表是否已经创建。"""
    required_tables = {"material_quality_asset", "material_quality_sample", "material_quality_feature"}
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME IN (
                  'material_quality_asset',
                  'material_quality_sample',
                  'material_quality_feature'
              )
        """)).fetchall()
    existing = {str(row[0]) for row in rows}
    missing = required_tables - existing
    if missing:
        raise RuntimeError("素材质量特征表尚未创建，缺少：{}".format(", ".join(sorted(missing))))


def _load_targets(
    cohort: str,
    feature_version: str,
    scan_limit: int,
    material_id: Optional[str],
    retry_unavailable: bool,
    label_group: str,
    retry_cooldown_minutes: int,
    first_date_from: Optional[datetime.date],
    first_date_to: Optional[datetime.date],
) -> List[dict]:
    """按照标签、日期和可重试状态选择完整窗口样本。"""
    params: Dict[str, object] = {
        "cohort": cohort,
        "feature_version": feature_version,
        "scan_limit": scan_limit,
        "retry_unavailable": 1 if retry_unavailable else 0,
        "retry_cooldown_seconds": retry_cooldown_minutes * 60,
    }
    material_filter = ""
    if material_id:
        material_filter = " AND a.material_id = :material_id"
        params["material_id"] = material_id
    label_filter = ""
    if label_group == "positive":
        label_filter = " AND s.label_gmv_7d > 0"
    elif label_group == "zero":
        label_filter = " AND s.label_gmv_7d = 0"
    date_filter, date_params = build_first_date_filter(first_date_from, first_date_to)
    params.update(date_params)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                a.id AS asset_id,
                a.material_id,
                a.source_status,
                a.file_size,
                s.label_gmv_7d
            FROM material_quality_sample AS s
            INNER JOIN material_quality_asset AS a ON a.id = s.asset_id
            LEFT JOIN material_quality_feature AS f
              ON f.asset_id = a.id AND f.feature_version = :feature_version
            WHERE s.cohort = :cohort
              AND s.label_status = 'ready'
              AND f.id IS NULL
              AND (a.source_status <> 'unavailable' OR :retry_unavailable = 1)
              AND (
                  a.last_error IS NULL
                  OR a.last_error NOT LIKE '%接口临时%'
                  OR a.update_time <= FROM_UNIXTIME(
                      UNIX_TIMESTAMP() - :retry_cooldown_seconds
                  )
              )
              {material_filter}
              {label_filter}
              {date_filter}
            ORDER BY
              (a.source_status = 'url_ready') DESC,
              (a.file_size > 0 AND a.file_size <= 536870912) DESC,
              CRC32(CONCAT(a.material_id, :feature_version))
            LIMIT :scan_limit
        """.format(
            material_filter=material_filter,
            label_filter=label_filter,
            date_filter=date_filter,
        )), params).mappings().all()
    return [dict(row) for row in rows]


def _mark_url_success(asset_id: int, detail: dict) -> None:
    """保存刷新后的地址和媒体元数据。"""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE material_quality_asset
            SET user_id = :user_id,
                advertiser_id = :advertiser_id,
                material_name = :material_name,
                video_url = :video_url,
                thumbnail_url = :thumbnail_url,
                duration = :duration,
                file_size = :file_size,
                source_status = 'url_ready',
                last_error = NULL,
                update_time = NOW()
            WHERE id = :asset_id
        """), {
            "asset_id": asset_id,
            "user_id": detail.get("user_id"),
            "advertiser_id": detail.get("advertiser_id"),
            "material_name": str(detail.get("filename") or detail.get("name") or "")[:255],
            "video_url": detail.get("url"),
            "thumbnail_url": detail.get("poster_url"),
            "duration": float(detail.get("duration") or 0),
            "file_size": int(detail.get("size") or 0),
        })


def _mark_processing(asset_id: int) -> None:
    """标记素材已被当前演示任务领取。"""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE material_quality_asset
            SET download_status = 'downloading',
                feature_status = 'processing',
                lease_until = DATE_ADD(NOW(), INTERVAL 30 MINUTE),
                attempt_count = attempt_count + 1,
                last_error = NULL,
                update_time = NOW()
            WHERE id = :asset_id
        """), {"asset_id": asset_id})


def _save_feature(
    target: dict,
    feature_version: str,
    download_info: dict,
    feature_info: dict,
    extraction_seconds: float,
) -> None:
    """幂等保存特征并标记临时视频已经清理。"""
    values = {
        "asset_id": target["asset_id"],
        "feature_version": feature_version,
        "content_sha256": download_info["content_sha256"],
        "duration": feature_info["duration"],
        "width": feature_info["width"],
        "height": feature_info["height"],
        "fps": feature_info["fps"],
        "bit_rate": feature_info["bit_rate"],
        "file_size": download_info["file_size"],
        "has_audio": 1 if feature_info["has_audio"] else 0,
        "sample_frame_count": feature_info["sample_frame_count"],
        "visual_vector": feature_info["visual_vector"],
        "visual_vector_dim": feature_info["visual_vector_dim"],
        "probe_json": json.dumps(feature_info["raw_probe"], ensure_ascii=False),
        "extraction_seconds": extraction_seconds,
    }
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO material_quality_feature (
                asset_id, feature_version, content_sha256,
                duration, width, height, fps, bit_rate, file_size, has_audio,
                sample_frame_count, visual_vector, visual_vector_dim,
                visual_vector_dtype, probe_json, extraction_seconds,
                create_time, update_time
            ) VALUES (
                :asset_id, :feature_version, :content_sha256,
                :duration, :width, :height, :fps, :bit_rate, :file_size, :has_audio,
                :sample_frame_count, :visual_vector, :visual_vector_dim,
                'float32', :probe_json, :extraction_seconds,
                NOW(), NOW()
            )
            ON DUPLICATE KEY UPDATE
                content_sha256 = VALUES(content_sha256),
                duration = VALUES(duration),
                width = VALUES(width),
                height = VALUES(height),
                fps = VALUES(fps),
                bit_rate = VALUES(bit_rate),
                file_size = VALUES(file_size),
                has_audio = VALUES(has_audio),
                sample_frame_count = VALUES(sample_frame_count),
                visual_vector = VALUES(visual_vector),
                visual_vector_dim = VALUES(visual_vector_dim),
                probe_json = VALUES(probe_json),
                extraction_seconds = VALUES(extraction_seconds),
                update_time = NOW()
        """), values)
        conn.execute(text("""
            UPDATE material_quality_asset
            SET download_status = 'cleaned',
                feature_status = 'ready',
                content_sha256 = :content_sha256,
                feature_version = :feature_version,
                duration = :duration,
                file_size = :file_size,
                lease_until = NULL,
                last_error = NULL,
                update_time = NOW()
            WHERE id = :asset_id
        """), values)


def _mark_failure(asset_id: int, phase: str, error: str) -> None:
    """记录可重试错误并释放任务租约。"""
    with engine.begin() as conn:
        if phase == "url":
            retryable_markers = (
                "接口临时限流",
                "接口临时异常",
                "接口异常",
                "token_refresh_failed",
            )
            source_status = (
                "pending_url"
                if any(marker in error for marker in retryable_markers)
                else "unavailable"
            )
            conn.execute(text("""
                UPDATE material_quality_asset
                SET source_status = :source_status,
                    last_error = :last_error,
                    lease_until = NULL,
                    update_time = NOW()
                WHERE id = :asset_id
            """), {
                "asset_id": asset_id,
                "source_status": source_status,
                "last_error": error[-2000:],
            })
        else:
            conn.execute(text("""
                UPDATE material_quality_asset
                SET download_status = 'pending',
                    feature_status = 'retry',
                    last_error = :last_error,
                    lease_until = NULL,
                    update_time = NOW()
                WHERE id = :asset_id
            """), {"asset_id": asset_id, "last_error": error[-2000:]})


def run_worker(args: argparse.Namespace) -> dict:
    """处理小批素材，成功数达到限制后停止。"""
    _verify_schema()
    os.makedirs(args.temp_root, exist_ok=True)
    targets = _load_targets(
        args.cohort,
        args.feature_version,
        args.scan_limit,
        args.material_id,
        args.retry_unavailable,
        args.label_group,
        args.retry_cooldown_minutes,
        args.first_date_from,
        args.first_date_to,
    )
    log_info("素材特征任务读取到{}个候选素材".format(len(targets)))
    succeeded = 0
    failed = 0
    auth_cache = {}
    usable_auth_cache = {}
    for target in targets:
        if succeeded >= args.limit:
            break
        material_id = str(target["material_id"])
        asset_id = int(target["asset_id"])
        with engine.connect() as conn:
            detail, url_error = resolve_material_url(
                conn,
                args.cohort,
                material_id,
                auth_cache=auth_cache,
                usable_auth_cache=usable_auth_cache,
            )
        if not detail:
            failed += 1
            _mark_failure(asset_id, "url", url_error)
            continue

        _mark_url_success(asset_id, detail)
        _mark_processing(asset_id)
        started_at = time.time()
        try:
            with tempfile.TemporaryDirectory(
                prefix="asset_{}_".format(asset_id),
                dir=args.temp_root,
            ) as work_dir:
                video_path = os.path.join(work_dir, "source.mp4")
                download_info = download_video(
                    str(detail["url"]),
                    video_path,
                    max_video_bytes=args.max_video_mb * 1024 * 1024,
                )
                feature_info = extract_media_features(video_path)
                extraction_seconds = time.time() - started_at
            _save_feature(
                target,
                args.feature_version,
                download_info,
                feature_info,
                extraction_seconds,
            )
            succeeded += 1
            log_info(
                "素材{}特征生成成功，帧数{}，维度{}，耗时{:.1f}秒".format(
                    material_id,
                    feature_info["sample_frame_count"],
                    feature_info["visual_vector_dim"],
                    extraction_seconds,
                )
            )
        except Exception as exc:
            failed += 1
            _mark_failure(asset_id, "feature", str(exc))
            log_error("素材{}特征生成失败：{}".format(material_id, exc))

    result = {
        "candidate_count": len(targets),
        "success_count": succeeded,
        "failure_count": failed,
        "feature_version": args.feature_version,
        "checked_auth_count": len(usable_auth_cache),
    }
    log_info("素材特征任务完成：{}".format(result))
    return result


def parse_args() -> argparse.Namespace:
    """读取演示任务命令行参数。"""
    parser = argparse.ArgumentParser(description="执行素材质量特征构造流程")
    parser.add_argument("--cohort", default=DEFAULT_COHORT, help="样本范围版本")
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION, help="特征版本")
    parser.add_argument("--limit", type=int, default=1, help="本次最多成功处理素材数")
    parser.add_argument("--scan-limit", type=int, default=20, help="本次最多尝试候选素材数")
    parser.add_argument("--material-id", help="仅处理指定素材ID")
    parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help="重新尝试此前已确认URL不可用的素材",
    )
    parser.add_argument(
        "--label-group",
        choices=("all", "positive", "zero"),
        default="all",
        help="按标签分层选择全部、正GMV或零GMV样本",
    )
    parser.add_argument(
        "--retry-cooldown-minutes",
        type=int,
        default=30,
        help="瞬时接口失败再次尝试前的冷却分钟数",
    )
    parser.add_argument(
        "--first-date-from",
        type=parse_date,
        help="仅处理首次统计日在该日期及之后的样本",
    )
    parser.add_argument(
        "--first-date-to",
        type=parse_date,
        help="仅处理首次统计日在该日期及之前的样本",
    )
    parser.add_argument("--temp-root", default=DEFAULT_TEMP_ROOT, help="独立临时目录")
    parser.add_argument("--max-video-mb", type=int, default=512, help="单个视频大小上限")
    return parser.parse_args()


def main() -> int:
    """演示任务入口。"""
    set_process_name("material_quality_feature")
    args = parse_args()
    if (
        args.limit <= 0
        or args.scan_limit <= 0
        or args.max_video_mb <= 0
        or args.retry_cooldown_minutes < 0
    ):
        log_warn("处理数量、扫描数量和视频大小上限必须大于0，冷却时间不能小于0")
        return 2
    if (
        args.first_date_from
        and args.first_date_to
        and args.first_date_from > args.first_date_to
    ):
        log_warn("首次统计日的起始日期不能晚于结束日期")
        return 2
    result = run_worker(args)
    return 0 if result["success_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

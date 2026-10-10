#!/usr/bin/env python3
"""执行全行业素材的一次性多模态特征抽取任务。"""

import argparse
import datetime
import json
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional

from sqlalchemy import text

from plan_logger import log_error, log_info, log_warn, set_process_name

from material_quality.database import engine
from material_quality.feature.media_features import download_video
from material_quality.feature.selection import build_first_date_filter, parse_date
from material_quality.feature.semantic_features import SemanticFeatureExtractor
from material_quality.feature.task_state import (
    mark_failure as _mark_failure,
    mark_processing as _mark_processing,
    mark_url_success as _mark_url_success,
)


DEFAULT_COHORT = "all_industries_v1"
DEFAULT_FEATURE_VERSION = "multimodal_cnclip_whisper_small_v2"
DEFAULT_MODEL_NAME = "RN50"
DEFAULT_MODEL_CACHE = "material_quality/.runtime/models/chinese_clip"
DEFAULT_ASR_MODEL_NAME = "small"
DEFAULT_ASR_MODEL_CACHE = "material_quality/.runtime/models/faster_whisper"
DEFAULT_TEMP_ROOT = "/tmp/material_quality_semantic_feature"
RETRYABLE_URL_MARKERS = (
    "接口临时限流",
    "接口临时异常",
    "接口异常",
    "token_refresh_failed",
    "token_expiring",
    "access_token_missing",
)


def _cleanup_stale_temp_dirs(temp_root: str, stale_hours: float) -> int:
    """清理异常中断遗留的独立素材目录，避免长期累积占满磁盘。"""
    if stale_hours <= 0 or not os.path.isdir(temp_root):
        return 0
    root_path = os.path.realpath(temp_root)
    cutoff_time = time.time() - stale_hours * 3600
    removed = 0
    for name in os.listdir(root_path):
        if not name.startswith("semantic_asset_"):
            continue
        path = os.path.realpath(os.path.join(root_path, name))
        if os.path.dirname(path) != root_path or not os.path.isdir(path):
            continue
        if os.path.getmtime(path) > cutoff_time:
            continue
        shutil.rmtree(path)
        removed += 1
    return removed


def _verify_schema() -> None:
    """检查全行业上下文表和多模态特征表是否已经创建。"""
    required_tables = {
        "material_quality_asset",
        "material_quality_sample",
        "material_quality_sample_context",
        "material_quality_semantic_feature",
        "material_quality_semantic_task",
    }
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME IN (
                  'material_quality_asset',
                  'material_quality_sample',
                  'material_quality_sample_context',
                  'material_quality_semantic_feature',
                  'material_quality_semantic_task'
              )
        """)).fetchall()
    existing = {str(row[0]) for row in rows}
    missing = required_tables - existing
    if missing:
        raise RuntimeError("素材质量语义特征表尚未创建，缺少：{}".format(
            ", ".join(sorted(missing))
        ))
    with engine.connect() as conn:
        column_rows = conn.execute(text("""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND (
                  (TABLE_NAME = 'material_quality_sample'
                   AND COLUMN_NAME IN (
                       'delivery_day_count',
                       'label_gmv_per_delivery_day',
                       'label_log_gmv_per_delivery_day'
                   ))
                  OR
                  (TABLE_NAME = 'material_quality_semantic_feature'
                   AND COLUMN_NAME IN (
                       'visual_temporal_vector', 'visual_temporal_dim',
                       'asr_text', 'asr_text_vector', 'audio_stats_vector',
                       'extractor_config_json'
                   ))
              )
        """)).fetchall()
    existing_columns = {
        "{}.{}".format(str(row[0]), str(row[1])) for row in column_rows
    }
    required_columns = {
        "material_quality_sample.delivery_day_count",
        "material_quality_sample.label_gmv_per_delivery_day",
        "material_quality_sample.label_log_gmv_per_delivery_day",
        "material_quality_semantic_feature.visual_temporal_vector",
        "material_quality_semantic_feature.visual_temporal_dim",
        "material_quality_semantic_feature.asr_text",
        "material_quality_semantic_feature.asr_text_vector",
        "material_quality_semantic_feature.audio_stats_vector",
        "material_quality_semantic_feature.extractor_config_json",
    }
    missing_columns = required_columns - existing_columns
    if missing_columns:
        raise RuntimeError(
            "素材质量多模态V2字段尚未迁移，缺少：{}".format(
                ", ".join(sorted(missing_columns))
            )
        )


def _load_targets(
    cohort: str,
    feature_version: str,
    scan_limit: int,
    material_id: Optional[str],
    retry_unavailable: bool,
    retry_failed: bool,
    include_window_open: bool,
    min_delivery_days: int,
    label_group: str,
    first_date_from: Optional[datetime.date],
    first_date_to: Optional[datetime.date],
    shard_count: int,
    shard_index: int,
    retry_cooldown_minutes: int,
) -> List[dict]:
    """按行业分散选择已投放两天以上的成熟或待成熟素材。"""
    params: Dict[str, object] = {
        "cohort": cohort,
        "feature_version": feature_version,
        "scan_limit": scan_limit,
        "retry_unavailable": 1 if retry_unavailable else 0,
        "retry_failed": 1 if retry_failed else 0,
        "include_window_open": 1 if include_window_open else 0,
        "min_delivery_days": min_delivery_days,
        "shard_count": shard_count,
        "shard_index": shard_index,
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
            WITH candidates AS (
                SELECT
                    a.id AS asset_id,
                    a.material_id,
                    a.source_status,
                    a.file_size,
                    s.label_gmv_7d,
                    s.label_gmv_per_delivery_day,
                    s.delivery_day_count,
                    s.label_status,
                    s.first_stat_date,
                    c.primary_industry,
                    c.industry_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY c.primary_industry
                        ORDER BY CRC32(CONCAT(a.material_id, ':', :feature_version))
                    ) AS industry_sequence
                FROM material_quality_sample AS s
                INNER JOIN material_quality_asset AS a ON a.id = s.asset_id
                INNER JOIN material_quality_sample_context AS c
                  ON c.sample_id = s.id AND c.cohort = s.cohort
                LEFT JOIN material_quality_semantic_feature AS f
                  ON f.asset_id = a.id AND f.feature_version = :feature_version
                LEFT JOIN material_quality_semantic_task AS t
                  ON t.asset_id = a.id AND t.feature_version = :feature_version
                WHERE s.cohort = :cohort
                  AND (
                      s.label_status = 'ready'
                      OR (:include_window_open = 1 AND s.label_status = 'window_open')
                  )
                  AND s.delivery_day_count >= :min_delivery_days
                  AND f.id IS NULL
                  AND (
                      a.source_status <> 'unavailable'
                      OR :retry_unavailable = 1
                      OR t.id IS NULL
                  )
                  AND (
                      t.id IS NULL
                      OR (t.status = 'processing' AND t.lease_until < NOW())
                      OR (
                          t.status = 'url_retry'
                          AND t.update_time <= FROM_UNIXTIME(
                              UNIX_TIMESTAMP() - :retry_cooldown_seconds
                          )
                      )
                      OR (
                          :retry_failed = 1
                          AND t.status IN (
                              'url_unavailable', 'url_retry', 'feature_retry'
                          )
                      )
                  )
                  AND MOD(
                      CRC32(CONCAT(a.material_id, ':', :feature_version)),
                      :shard_count
                  ) = :shard_index
                  {material_filter}
                  {label_filter}
                  {date_filter}
            )
            SELECT
                asset_id, material_id, source_status, file_size,
                label_gmv_7d, label_gmv_per_delivery_day,
                delivery_day_count, label_status, first_stat_date,
                primary_industry, industry_count
            FROM candidates
            ORDER BY
                industry_sequence,
                CRC32(CONCAT(material_id, ':', :feature_version))
            LIMIT :scan_limit
        """.format(
            material_filter=material_filter,
            label_filter=label_filter,
            date_filter=date_filter,
        )), params).mappings().all()
    return [dict(row) for row in rows]


def _claim_task(
    asset_id: int,
    feature_version: str,
    retry_failed: bool,
    retry_cooldown_minutes: int,
) -> bool:
    """原子领取版本化任务，阻止并发进程重复处理同一素材。"""
    with engine.begin() as conn:
        inserted = conn.execute(text("""
            INSERT IGNORE INTO material_quality_semantic_task (
                asset_id, feature_version, status, last_phase,
                attempt_count, last_error, lease_until,
                last_attempt_time, create_time, update_time
            ) VALUES (
                :asset_id, :feature_version, 'processing', 'url',
                1, NULL, DATE_ADD(NOW(), INTERVAL 60 MINUTE),
                NOW(), NOW(), NOW()
            )
        """), {"asset_id": asset_id, "feature_version": feature_version})
        if int(inserted.rowcount or 0) == 1:
            return True
        claimed = conn.execute(text("""
            UPDATE material_quality_semantic_task
            SET status = 'processing',
                last_phase = 'url',
                attempt_count = attempt_count + 1,
                last_error = NULL,
                lease_until = DATE_ADD(NOW(), INTERVAL 60 MINUTE),
                last_attempt_time = NOW(),
                update_time = NOW()
            WHERE asset_id = :asset_id
              AND feature_version = :feature_version
              AND (
                  (status = 'processing' AND lease_until < NOW())
                  OR (
                      status = 'url_retry'
                      AND update_time <= FROM_UNIXTIME(
                          UNIX_TIMESTAMP() - :retry_cooldown_seconds
                      )
                  )
                  OR (
                      :retry_failed = 1
                      AND status IN ('url_unavailable', 'feature_retry')
                  )
              )
        """), {
            "asset_id": asset_id,
            "feature_version": feature_version,
            "retry_failed": 1 if retry_failed else 0,
            "retry_cooldown_seconds": retry_cooldown_minutes * 60,
        })
        return int(claimed.rowcount or 0) == 1


def _mark_semantic_failure(
    asset_id: int,
    feature_version: str,
    phase: str,
    error: str,
) -> None:
    """同时记录资产失败和版本化任务失败，避免后续批次反复试错。"""
    _mark_failure(asset_id, phase, error)
    if phase == "url" and _is_retryable_url_error(error):
        task_status = "url_retry"
    elif phase == "url":
        task_status = "url_unavailable"
    else:
        task_status = "feature_retry"
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE material_quality_semantic_task
            SET status = :status,
                last_phase = :phase,
                last_error = :last_error,
                lease_until = NULL,
                update_time = NOW()
            WHERE asset_id = :asset_id AND feature_version = :feature_version
        """), {
            "asset_id": asset_id,
            "feature_version": feature_version,
            "status": task_status,
            "phase": phase,
            "last_error": error[-2000:],
        })


def _is_retryable_url_error(error: str) -> bool:
    """判断URL错误是否适合冷却后重试。"""
    return any(marker in str(error or "") for marker in RETRYABLE_URL_MARKERS)


def _save_feature(
    target: dict,
    feature_version: str,
    model_name: str,
    download_info: dict,
    feature_info: dict,
    extraction_seconds: float,
) -> None:
    """幂等保存多模态特征，并标记临时素材已经清理。"""
    values = {
        "asset_id": target["asset_id"],
        "feature_version": feature_version,
        "content_sha256": download_info["content_sha256"],
        "visual_model_name": model_name,
        "visual_vector": feature_info["visual_vector"],
        "visual_vector_dim": feature_info["visual_vector_dim"],
        "visual_temporal_vector": feature_info["visual_temporal_vector"],
        "visual_temporal_dim": feature_info["visual_temporal_dim"],
        "text_model_name": model_name,
        "text_vector": feature_info["text_vector"],
        "text_vector_dim": feature_info["text_vector_dim"],
        "ocr_engine_name": "RapidOCR-ONNXRuntime",
        "ocr_text": feature_info["ocr_text"],
        "ocr_json": feature_info["ocr_json"],
        "ocr_stats_vector": feature_info["ocr_stats_vector"],
        "ocr_stats_dim": feature_info["ocr_stats_dim"],
        "asr_model_name": feature_info["asr_model_name"],
        "asr_language": feature_info["asr_language"],
        "asr_language_probability": feature_info["asr_language_probability"],
        "asr_text": feature_info["asr_text"],
        "asr_json": feature_info["asr_json"],
        "asr_text_vector": feature_info["asr_text_vector"],
        "asr_text_vector_dim": feature_info["asr_text_vector_dim"],
        "audio_stats_vector": feature_info["audio_stats_vector"],
        "audio_stats_dim": feature_info["audio_stats_dim"],
        "duration": feature_info["duration"],
        "width": feature_info["width"],
        "height": feature_info["height"],
        "fps": feature_info["fps"],
        "bit_rate": feature_info["bit_rate"],
        "file_size": download_info["file_size"],
        "has_audio": 1 if feature_info["has_audio"] else 0,
        "sample_frame_count": feature_info["sample_frame_count"],
        "probe_json": json.dumps(feature_info["raw_probe"], ensure_ascii=False),
        "extractor_config_json": json.dumps(
            feature_info["extractor_config"],
            ensure_ascii=False,
        ),
        "extraction_seconds": extraction_seconds,
    }
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO material_quality_semantic_feature (
                asset_id, feature_version, content_sha256,
                visual_model_name, visual_vector, visual_vector_dim,
                visual_vector_dtype, visual_temporal_vector, visual_temporal_dim,
                text_model_name, text_vector,
                text_vector_dim, text_vector_dtype, ocr_engine_name,
                ocr_text, ocr_json, ocr_stats_vector, ocr_stats_dim,
                asr_model_name, asr_language, asr_language_probability,
                asr_text, asr_json, asr_text_vector,
                asr_text_vector_dim, asr_text_vector_dtype,
                audio_stats_vector, audio_stats_dim,
                duration, width, height, fps, bit_rate, file_size, has_audio,
                sample_frame_count, probe_json, extractor_config_json,
                extraction_seconds,
                create_time, update_time
            ) VALUES (
                :asset_id, :feature_version, :content_sha256,
                :visual_model_name, :visual_vector, :visual_vector_dim,
                'float32', :visual_temporal_vector, :visual_temporal_dim,
                :text_model_name, :text_vector,
                :text_vector_dim, 'float32', :ocr_engine_name,
                :ocr_text, :ocr_json, :ocr_stats_vector, :ocr_stats_dim,
                :asr_model_name, :asr_language, :asr_language_probability,
                :asr_text, :asr_json, :asr_text_vector,
                :asr_text_vector_dim, 'float32',
                :audio_stats_vector, :audio_stats_dim,
                :duration, :width, :height, :fps, :bit_rate, :file_size, :has_audio,
                :sample_frame_count, :probe_json, :extractor_config_json,
                :extraction_seconds,
                NOW(), NOW()
            )
            ON DUPLICATE KEY UPDATE
                content_sha256 = VALUES(content_sha256),
                visual_model_name = VALUES(visual_model_name),
                visual_vector = VALUES(visual_vector),
                visual_vector_dim = VALUES(visual_vector_dim),
                visual_temporal_vector = VALUES(visual_temporal_vector),
                visual_temporal_dim = VALUES(visual_temporal_dim),
                text_model_name = VALUES(text_model_name),
                text_vector = VALUES(text_vector),
                text_vector_dim = VALUES(text_vector_dim),
                ocr_engine_name = VALUES(ocr_engine_name),
                ocr_text = VALUES(ocr_text),
                ocr_json = VALUES(ocr_json),
                ocr_stats_vector = VALUES(ocr_stats_vector),
                ocr_stats_dim = VALUES(ocr_stats_dim),
                asr_model_name = VALUES(asr_model_name),
                asr_language = VALUES(asr_language),
                asr_language_probability = VALUES(asr_language_probability),
                asr_text = VALUES(asr_text),
                asr_json = VALUES(asr_json),
                asr_text_vector = VALUES(asr_text_vector),
                asr_text_vector_dim = VALUES(asr_text_vector_dim),
                audio_stats_vector = VALUES(audio_stats_vector),
                audio_stats_dim = VALUES(audio_stats_dim),
                duration = VALUES(duration),
                width = VALUES(width),
                height = VALUES(height),
                fps = VALUES(fps),
                bit_rate = VALUES(bit_rate),
                file_size = VALUES(file_size),
                has_audio = VALUES(has_audio),
                sample_frame_count = VALUES(sample_frame_count),
                probe_json = VALUES(probe_json),
                extractor_config_json = VALUES(extractor_config_json),
                extraction_seconds = VALUES(extraction_seconds),
                update_time = NOW()
        """), values)
        conn.execute(text("""
            UPDATE material_quality_semantic_task
            SET status = 'ready',
                last_phase = 'complete',
                last_error = NULL,
                lease_until = NULL,
                update_time = NOW()
            WHERE asset_id = :asset_id AND feature_version = :feature_version
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


def run_worker(args: argparse.Namespace) -> dict:
    """复用模型处理小批素材，成功数量达到限制后停止。"""
    _verify_schema()
    os.makedirs(args.temp_root, exist_ok=True)
    removed_temp_count = _cleanup_stale_temp_dirs(
        args.temp_root,
        args.stale_temp_hours,
    )
    if removed_temp_count:
        log_info("已清理{}个异常中断遗留的素材临时目录".format(removed_temp_count))
    targets = _load_targets(
        args.cohort,
        args.feature_version,
        args.scan_limit,
        args.material_id,
        args.retry_unavailable,
        args.retry_failed,
        args.include_window_open,
        args.min_delivery_days,
        args.label_group,
        args.first_date_from,
        args.first_date_to,
        args.shard_count,
        args.shard_index,
        args.retry_cooldown_minutes,
    )
    candidate_count = len(targets)
    for target in targets:
        target["cohort"] = args.cohort
    log_info("语义特征任务读取到{}个候选素材".format(candidate_count))
    if not targets:
        return {
            "candidate_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "feature_version": args.feature_version,
        }
    local_state = None
    artifact_store = None
    if args.state_mode == "local":
        from material_quality.feature.local_task_state import LocalTaskState

        local_state = LocalTaskState(args.local_state_db)
        local_state.upsert_targets(targets, args.feature_version, args.cohort)
        targets = local_state.filter_unfinished_targets(
            targets,
            args.feature_version,
            include_unavailable=args.retry_failed,
        )
        log_info("排除本地已完成任务后剩余{}条候选素材".format(len(targets)))
    if args.raw_feature_root:
        from material_quality.feature.local_artifact_store import (
            LocalRawFeatureStore,
        )

        artifact_store = LocalRawFeatureStore(
            args.raw_feature_root,
            args.feature_version,
        )

    auth_cache = {}
    usable_auth_cache = {}
    material_candidates_cache = None
    account_status_counts = {}
    if args.url_mode == "local_readonly":
        from material_quality.feature.local_url_resolver import (
            preload_material_auth_candidates,
        )

        with engine.connect() as conn:
            auth_snapshot = preload_material_auth_candidates(
                conn,
                args.cohort,
                [str(target["material_id"]) for target in targets],
            )
        auth_cache = auth_snapshot["auth_cache"]
        usable_auth_cache = auth_snapshot["usable_auth_cache"]
        material_candidates_cache = auth_snapshot["material_candidates"]
        account_status_counts = auth_snapshot["account_status_counts"]
        deferred_errors = auth_snapshot["deferred_errors"]
        if deferred_errors:
            deferred_asset_errors = {
                int(target["asset_id"]): deferred_errors[str(target["material_id"])]
                for target in targets
                if str(target["material_id"]) in deferred_errors
            }
            if local_state:
                local_state.defer_assets(
                    deferred_asset_errors,
                    args.feature_version,
                )
            targets = [
                target for target in targets
                if str(target["material_id"]) not in deferred_errors
            ]
            log_info(
                "账户授权前置过滤跳过{}条素材，保留{}条可调用素材，账户状态{}".format(
                    len(deferred_asset_errors),
                    len(targets),
                    account_status_counts,
                )
            )
        if local_state:
            local_state.reactivate_assets(
                [int(target["asset_id"]) for target in targets],
                args.feature_version,
            )
    if not targets:
        result = {
            "candidate_count": candidate_count,
            "eligible_candidate_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "feature_version": args.feature_version,
            "account_status_counts": account_status_counts,
        }
        if local_state:
            result["local_task_statuses"] = local_state.summary(
                args.feature_version
            )
        log_info("语义特征任务完成：{}".format(result))
        return result
    if args.preflight_only:
        result = {
            "candidate_count": candidate_count,
            "eligible_candidate_count": len(targets),
            "success_count": 0,
            "failure_count": 0,
            "feature_version": args.feature_version,
            "checked_auth_count": len(usable_auth_cache),
            "account_status_counts": account_status_counts,
        }
        if local_state:
            result["local_task_statuses"] = local_state.summary(
                args.feature_version
            )
        log_info("语义特征预检完成：{}".format(result))
        return result
    extractor = SemanticFeatureExtractor(
        args.model_name,
        args.model_cache,
        args.torch_threads,
        use_modelscope=args.model_source == "modelscope",
        asr_model_name=args.asr_model_name,
        asr_model_cache=args.asr_model_cache,
        asr_compute_type=args.asr_compute_type,
        asr_cpu_threads=args.asr_cpu_threads,
        ocr_cpu_threads=args.ocr_cpu_threads,
        ocr_worker_count=args.ocr_worker_count,
        asr_num_workers=args.asr_num_workers,
        asr_split_min_seconds=args.asr_split_min_seconds,
        parallel_slot_dir=args.parallel_slot_dir,
        asr_extra_slots=args.asr_extra_slots,
    )
    succeeded = 0
    failed = 0
    if args.url_mode == "local_readonly":
        from material_quality.feature.local_url_resolver import (
            resolve_material_url_local as resolve_url,
        )
    else:
        from material_quality.feature.url_resolver import (
            resolve_material_url as resolve_url,
        )
    for target in targets:
        if succeeded >= args.limit:
            break
        if args.stop_file and os.path.exists(args.stop_file):
            log_info("检测到安全停止文件，将在当前素材边界结束任务")
            break
        free_disk_bytes = shutil.disk_usage(args.temp_root).free
        if free_disk_bytes < args.min_free_disk_gb * 1024 * 1024 * 1024:
            log_warn("本地可用磁盘低于{}GB，停止领取新素材".format(
                args.min_free_disk_gb
            ))
            break
        material_id = str(target["material_id"])
        asset_id = int(target["asset_id"])
        if local_state:
            if not local_state.claim(asset_id, args.feature_version):
                continue
        else:
            if not _claim_task(
                asset_id,
                args.feature_version,
                args.retry_failed,
                args.retry_cooldown_minutes,
            ):
                continue
        with engine.connect() as conn:
            resolve_kwargs = {
                "auth_cache": auth_cache,
                "usable_auth_cache": usable_auth_cache,
            }
            if args.url_mode == "local_readonly":
                resolve_kwargs["material_candidates_cache"] = (
                    material_candidates_cache
                )
            detail, url_error = resolve_url(
                conn,
                args.cohort,
                material_id,
                **resolve_kwargs
            )
        if not detail:
            failed += 1
            if local_state:
                local_state.mark_failure(
                    asset_id,
                    args.feature_version,
                    "url",
                    url_error,
                    _is_retryable_url_error(url_error),
                    args.retry_cooldown_minutes * 60,
                )
            else:
                _mark_semantic_failure(
                    asset_id,
                    args.feature_version,
                    "url",
                    url_error,
                )
            continue

        if not local_state:
            _mark_url_success(asset_id, detail)
            _mark_processing(asset_id)
        started_at = time.time()
        try:
            with tempfile.TemporaryDirectory(
                prefix="semantic_asset_{}_".format(asset_id),
                dir=args.temp_root,
            ) as work_dir:
                video_path = os.path.join(work_dir, "source.mp4")
                download_info = download_video(
                    str(detail["url"]),
                    video_path,
                    max_video_bytes=args.max_video_mb * 1024 * 1024,
                )
                if (
                    local_state
                    and artifact_store
                    and artifact_store.has_content(download_info["content_sha256"])
                ):
                    artifact_store.save_asset_reference(
                        target,
                        download_info["content_sha256"],
                    )
                    artifact_path = artifact_store.content_directory(
                        download_info["content_sha256"]
                    )
                    local_state.mark_ready(
                        asset_id,
                        args.feature_version,
                        download_info["content_sha256"],
                        artifact_path,
                    )
                    succeeded += 1
                    log_info("素材{}命中相同视频内容，复用已有原始特征".format(
                        material_id
                    ))
                    continue
                feature_info = extractor.extract(
                    video_path,
                    work_dir,
                    keyframe_count=args.keyframe_count,
                    frame_long_side=args.frame_long_side,
                )
                feature_info["extractor_config"] = {
                    "keyframe_count": args.keyframe_count,
                    "frame_long_side": args.frame_long_side,
                    "frame_sampling_strategy": "opening_scene_timeline_v1",
                    "opening_seconds": 5.0,
                    "opening_interval_seconds": 0.5,
                    "timeline_interval_seconds": 2.0,
                    "scene_threshold": 0.30,
                    "max_scene_changes": 12,
                    "visual_model_name": args.model_name,
                    "asr_model_name": args.asr_model_name,
                    "asr_compute_type": args.asr_compute_type,
                    "ocr_cpu_threads": args.ocr_cpu_threads,
                    "ocr_worker_count": args.ocr_worker_count,
                    "asr_cpu_threads": args.asr_cpu_threads,
                    "asr_num_workers": args.asr_num_workers,
                    "asr_split_min_seconds": args.asr_split_min_seconds,
                    "asr_extra_slots": args.asr_extra_slots,
                    "torch_threads": args.torch_threads,
                }
                feature_info["visual_model_name"] = args.model_name
                artifact_path = ""
                if artifact_store:
                    artifact_path = artifact_store.save(
                        target,
                        download_info["content_sha256"],
                        feature_info,
                        source_video_path=video_path,
                    )
                extraction_seconds = time.time() - started_at
            if local_state:
                local_state.mark_ready(
                    asset_id,
                    args.feature_version,
                    download_info["content_sha256"],
                    artifact_path,
                )
            else:
                _save_feature(
                    target,
                    args.feature_version,
                    args.model_name,
                    download_info,
                    feature_info,
                    extraction_seconds,
                )
            succeeded += 1
            log_info(
                "素材{}多模态特征生成成功，行业{}，帧数{}，视觉维度{}，OCR字符{}，ASR字符{}，耗时{:.1f}秒，阶段耗时{}".format(
                    material_id,
                    target.get("primary_industry") or "未知",
                    feature_info["sample_frame_count"],
                    feature_info["visual_vector_dim"],
                    len(feature_info["ocr_text"]),
                    len(feature_info["asr_text"]),
                    extraction_seconds,
                    feature_info.get("stage_seconds") or {},
                )
            )
        except Exception as exc:
            failed += 1
            if local_state:
                local_state.mark_failure(
                    asset_id,
                    args.feature_version,
                    "feature",
                    str(exc),
                    True,
                    args.retry_cooldown_minutes * 60,
                )
            else:
                _mark_semantic_failure(
                    asset_id,
                    args.feature_version,
                    "feature",
                    str(exc),
                )
            log_error("素材{}语义特征生成失败：{}".format(material_id, exc))

    result = {
        "candidate_count": candidate_count,
        "eligible_candidate_count": len(targets),
        "success_count": succeeded,
        "failure_count": failed,
        "feature_version": args.feature_version,
        "checked_auth_count": len(usable_auth_cache),
        "account_status_counts": account_status_counts,
    }
    if local_state:
        result["local_task_statuses"] = local_state.summary(args.feature_version)
    log_info("语义特征任务完成：{}".format(result))
    return result


def parse_args() -> argparse.Namespace:
    """读取全行业语义抽取任务参数。"""
    parser = argparse.ArgumentParser(description="执行素材质量多模态语义特征构造流程")
    parser.add_argument("--cohort", default=DEFAULT_COHORT, help="样本范围版本")
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION, help="特征版本")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="中文CLIP官方模型名")
    parser.add_argument("--model-cache", default=DEFAULT_MODEL_CACHE, help="模型权重缓存目录")
    parser.add_argument("--asr-model-name", default=DEFAULT_ASR_MODEL_NAME, help="ASR模型名称")
    parser.add_argument("--asr-model-cache", default=DEFAULT_ASR_MODEL_CACHE, help="ASR模型缓存目录")
    parser.add_argument(
        "--asr-compute-type",
        default="int8",
        help="ASR模型CPU计算精度",
    )
    parser.add_argument(
        "--model-source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
        help="模型下载源，国内服务器默认使用ModelScope",
    )
    parser.add_argument("--limit", type=int, default=1, help="本次最多成功处理素材数")
    parser.add_argument("--scan-limit", type=int, default=50, help="本次最多尝试候选素材数")
    parser.add_argument("--material-id", help="仅处理指定素材ID")
    parser.add_argument(
        "--url-mode",
        choices=("production", "local_readonly"),
        default="production",
        help="生产统一刷新或本地只读Token解析模式",
    )
    parser.add_argument(
        "--state-mode",
        choices=("database", "local"),
        default="database",
        help="任务和特征状态写生产数据库或仅写本地",
    )
    parser.add_argument(
        "--local-state-db",
        default="/tmp/material_quality_local_tasks.sqlite3",
        help="本地任务状态SQLite路径",
    )
    parser.add_argument(
        "--raw-feature-root",
        default="",
        help="原始逐帧和逐块Embedding的本地版本化目录",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只读取候选和授权状态，不加载模型、下载视频或写生产数据库",
    )
    parser.add_argument("--retry-unavailable", action="store_true", help="重试旧范围中URL不可用的素材")
    parser.add_argument("--retry-failed", action="store_true", help="重试当前语义版本已经失败的素材")
    parser.add_argument(
        "--include-window-open",
        action="store_true",
        help="允许标签窗口未闭合但已投放两天以上的素材先抽特征",
    )
    parser.add_argument(
        "--min-delivery-days",
        type=int,
        default=2,
        help="进入下载候选所需的最少实际投放天数",
    )
    parser.add_argument(
        "--label-group",
        choices=("all", "positive", "zero"),
        default="all",
        help="按标签分层选择全部、正GMV或零GMV样本",
    )
    parser.add_argument("--first-date-from", type=parse_date, help="首次统计日起始日期")
    parser.add_argument("--first-date-to", type=parse_date, help="首次统计日结束日期")
    parser.add_argument("--keyframe-count", type=int, default=48, help="每个视频抽取的关键帧上限")
    parser.add_argument("--frame-long-side", type=int, default=960, help="OCR关键帧最长边像素")
    parser.add_argument("--torch-threads", type=int, default=4, help="单任务PyTorch CPU线程数")
    parser.add_argument("--ocr-cpu-threads", type=int, default=2, help="单任务OCR CPU线程数")
    parser.add_argument("--ocr-worker-count", type=int, default=2, help="单素材OCR并行引擎数")
    parser.add_argument("--asr-cpu-threads", type=int, default=2, help="单任务ASR CPU线程数")
    parser.add_argument("--asr-num-workers", type=int, default=2, help="单模型ASR并行推理工作数")
    parser.add_argument(
        "--asr-split-min-seconds",
        type=float,
        default=45.0,
        help="启用长音频动态分段的最小秒数",
    )
    parser.add_argument("--parallel-slot-dir", default="", help="跨进程额外并发槽位目录")
    parser.add_argument("--asr-extra-slots", type=int, default=0, help="全机长音频额外并发槽位数")
    parser.add_argument("--shard-count", type=int, default=1, help="并发任务分片总数")
    parser.add_argument("--shard-index", type=int, default=0, help="当前任务分片编号，从0开始")
    parser.add_argument(
        "--retry-cooldown-minutes",
        type=int,
        default=30,
        help="URL临时失败后的自动重试冷却分钟数",
    )
    parser.add_argument("--temp-root", default=DEFAULT_TEMP_ROOT, help="独立临时目录")
    parser.add_argument("--stop-file", help="存在时在素材边界安全停止的本地文件")
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=20.0,
        help="可用磁盘低于该值时停止领取新素材",
    )
    parser.add_argument(
        "--stale-temp-hours",
        type=float,
        default=6.0,
        help="清理异常中断临时目录的最小小时数",
    )
    parser.add_argument("--max-video-mb", type=int, default=512, help="单个视频大小上限")
    return parser.parse_args()


def main() -> int:
    """语义抽取任务入口。"""
    set_process_name("material_quality_semantic")
    args = parse_args()
    positive_values = (
        args.limit,
        args.scan_limit,
        args.keyframe_count,
        args.frame_long_side,
        args.torch_threads,
        args.ocr_cpu_threads,
        args.ocr_worker_count,
        args.asr_cpu_threads,
        args.asr_num_workers,
        args.asr_split_min_seconds,
        args.min_delivery_days,
        args.shard_count,
        args.retry_cooldown_minutes,
        args.max_video_mb,
        args.stale_temp_hours,
        args.min_free_disk_gb,
    )
    if any(value <= 0 for value in positive_values):
        log_warn("处理数量、抽帧参数、线程数和视频大小上限必须大于0")
        return 2
    if args.asr_extra_slots < 0:
        log_warn("ASR额外并发槽位数不能为负数")
        return 2
    if args.state_mode == "local" and not args.raw_feature_root:
        log_warn("本地状态模式必须指定原始特征目录")
        return 2
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        log_warn("任务分片编号必须小于分片总数且不能为负数")
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

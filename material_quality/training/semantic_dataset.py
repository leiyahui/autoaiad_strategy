"""多模态V2训练数据读取、解码和快照工具。"""

import datetime
import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from material_quality.inference.feature_schema import (
    ASR_SEMANTIC_DIM,
    AUDIO_STATS_DIM,
    FEATURE_GROUP_NAMES,
    FEATURE_GROUP_RANGES,
    FEATURE_NAMES,
    MEDIA_FEATURE_NAMES,
    OCR_SEMANTIC_DIM,
    OCR_STATS_DIM,
    VISUAL_SEMANTIC_DIM,
    VISUAL_TEMPORAL_DIM,
    build_semantic_feature_row,
)


LOCAL_LABEL_VERSION = "gmv_per_delivery_day_7d_v2"


def _dataset_digest(dataset: Dict[str, np.ndarray]) -> str:
    """计算快照摘要，保证训练和评估使用同一批数据。"""
    digest = hashlib.sha256()
    for key in ("sample_ids", "asset_ids", "date_ordinals", "labels", "features"):
        values = np.ascontiguousarray(dataset[key])
        digest.update(key.encode("utf-8"))
        digest.update(values.tobytes())
    for key in ("material_ids", "content_sha256", "primary_industries"):
        if key not in dataset:
            continue
        digest.update(key.encode("utf-8"))
        digest.update("\n".join(dataset[key].tolist()).encode("utf-8"))
    return digest.hexdigest()


def _read_json(path: str, description: str) -> dict:
    """读取本地档案JSON，并把损坏文件转换为清晰错误。"""
    try:
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("{}读取失败：{}".format(description, exc))
    if not isinstance(payload, dict):
        raise ValueError("{}必须是JSON对象".format(description))
    return payload


def _parse_local_date(value: object) -> datetime.date:
    """解析本地素材引用中的首次统计日期。"""
    try:
        return datetime.datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        raise ValueError("首次统计日期格式无效：{}".format(value))


def _local_vector(
    vectors: object,
    key: str,
    expected_dim: int,
    allow_missing: bool = False,
) -> Tuple[Optional[bytes], int]:
    """读取本地NPZ中的聚合向量并校验固定维度。"""
    if key not in vectors:
        if allow_missing:
            return None, 0
        raise ValueError("本地向量文件缺少{}".format(key))
    array = np.asarray(vectors[key], dtype="<f4")
    if allow_missing and array.size == 0:
        return None, 0
    if array.ndim != 1 or array.size != expected_dim:
        raise ValueError("{}维度异常，期望{}，实际{}".format(
            key,
            expected_dim,
            array.shape,
        ))
    if not np.isfinite(array).all():
        raise ValueError("{}包含非有限数值".format(key))
    return np.ascontiguousarray(array).tobytes(), int(array.size)


def _build_local_semantic_feature_row(vectors: object, manifest: dict) -> np.ndarray:
    """把V4本地原始档案转换为现有多模态训练特征。"""
    visual, visual_dim = _local_vector(
        vectors,
        "visual_aggregate",
        VISUAL_SEMANTIC_DIM,
    )
    temporal, temporal_dim = _local_vector(
        vectors,
        "visual_temporal",
        VISUAL_TEMPORAL_DIM,
    )
    ocr, ocr_dim = _local_vector(
        vectors,
        "ocr_aggregate",
        OCR_SEMANTIC_DIM,
        allow_missing=True,
    )
    ocr_stats, ocr_stats_dim = _local_vector(
        vectors,
        "ocr_stats",
        OCR_STATS_DIM,
    )
    asr, asr_dim = _local_vector(
        vectors,
        "asr_aggregate",
        ASR_SEMANTIC_DIM,
        allow_missing=True,
    )
    audio_stats, audio_stats_dim = _local_vector(
        vectors,
        "audio_stats",
        AUDIO_STATS_DIM,
    )
    probe = manifest.get("probe") or {}
    format_info = probe.get("format") or {}
    row = {
        "visual_vector": visual,
        "visual_vector_dim": visual_dim,
        "visual_temporal_vector": temporal,
        "visual_temporal_dim": temporal_dim,
        "text_vector": ocr,
        "text_vector_dim": ocr_dim,
        "ocr_stats_vector": ocr_stats,
        "ocr_stats_dim": ocr_stats_dim,
        "asr_text_vector": asr,
        "asr_text_vector_dim": asr_dim,
        "audio_stats_vector": audio_stats,
        "audio_stats_dim": audio_stats_dim,
        "duration": manifest.get("duration"),
        "file_size": format_info.get("size"),
        "bit_rate": manifest.get("bit_rate") or format_info.get("bit_rate"),
        "fps": manifest.get("fps"),
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "has_audio": 1 if manifest.get("has_audio") else 0,
    }
    return build_semantic_feature_row(row)


def load_local_semantic_feature(
    raw_feature_root: str,
    feature_version: str,
    content_sha256: str,
) -> Tuple[np.ndarray, dict]:
    """按内容哈希读取一条本地V4特征，供训练和在线推理共用。"""
    normalized_sha = str(content_sha256 or "").lower()
    if len(normalized_sha) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_sha
    ):
        raise ValueError("视频内容哈希不是有效SHA256")
    version_root = os.path.realpath(os.path.join(raw_feature_root, feature_version))
    expected_parent = os.path.realpath(raw_feature_root)
    if os.path.commonpath([version_root, expected_parent]) != expected_parent:
        raise ValueError("特征版本路径超出本地原始特征目录")
    content_base = os.path.join(
        version_root,
        "content",
        normalized_sha[:2],
        normalized_sha,
    )
    manifest = _read_json(
        content_base + ".json",
        "内容档案{}".format(normalized_sha),
    )
    if manifest.get("content_sha256") != normalized_sha:
        raise ValueError("内容档案哈希与请求哈希不一致")
    try:
        with np.load(content_base + ".npz", allow_pickle=False) as vectors:
            feature = _build_local_semantic_feature_row(vectors, manifest)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("本地向量读取失败，原因：{}".format(exc))
    return feature, manifest


def load_local_semantic_dataset(
    raw_feature_root: str,
    cohort: str,
    feature_version: str,
    first_date_from: Optional[datetime.date] = None,
    first_date_to: Optional[datetime.date] = None,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """从本地V4档案冻结成熟样本，不访问或修改生产数据库。"""
    version_root = os.path.realpath(os.path.join(raw_feature_root, feature_version))
    expected_parent = os.path.realpath(raw_feature_root)
    if os.path.commonpath([version_root, expected_parent]) != expected_parent:
        raise ValueError("特征版本路径超出本地原始特征目录")
    asset_root = os.path.join(version_root, "assets")
    content_root = os.path.join(version_root, "content")
    if not os.path.isdir(asset_root) or not os.path.isdir(content_root):
        raise RuntimeError("本地V4素材引用或内容目录不存在")

    records = []
    for filename in sorted(os.listdir(asset_root)):
        if not filename.endswith(".json"):
            continue
        reference = _read_json(
            os.path.join(asset_root, filename),
            "素材引用{}".format(filename),
        )
        if reference.get("feature_version") != feature_version:
            continue
        if reference.get("cohort") != cohort:
            continue
        if reference.get("label_status") != "ready":
            continue
        if int(reference.get("delivery_day_count") or 0) < 2:
            continue
        first_date = _parse_local_date(reference.get("first_stat_date"))
        if first_date_from is not None and first_date < first_date_from:
            continue
        if first_date_to is not None and first_date > first_date_to:
            continue
        label = float(reference.get("label_gmv_per_delivery_day") or 0.0)
        if label < 0 or not math.isfinite(label):
            raise ValueError("素材{}的日均GMV标签无效".format(filename))

        content_sha256 = str(reference.get("content_sha256") or "").lower()
        if len(content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in content_sha256
        ):
            raise ValueError("素材{}的内容哈希无效".format(filename))
        content_base = os.path.join(
            content_root,
            content_sha256[:2],
            content_sha256,
        )
        content_manifest = _read_json(
            content_base + ".json",
            "内容档案{}".format(content_sha256),
        )
        if content_manifest.get("content_sha256") != content_sha256:
            raise ValueError("内容档案哈希与素材引用不一致")
        vector_path = content_base + ".npz"
        try:
            with np.load(vector_path, allow_pickle=False) as vectors:
                features = _build_local_semantic_feature_row(
                    vectors,
                    content_manifest,
                )
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("素材{}的本地向量读取失败：{}".format(filename, exc))
        records.append({
            "sample_id": int(reference["asset_id"]),
            "asset_id": int(reference["asset_id"]),
            "material_id": str(reference.get("material_id") or ""),
            "content_sha256": content_sha256,
            "first_date": first_date,
            "delivery_day_count": int(reference.get("delivery_day_count") or 0),
            "label": label,
            "primary_industry": str(reference.get("primary_industry") or "未知行业"),
            "features": features,
        })
    if not records:
        raise RuntimeError("没有可用于训练的成熟本地V4样本")

    records.sort(key=lambda item: (item["first_date"], item["material_id"]))
    content_hashes = [item["content_sha256"] for item in records]
    if len(set(content_hashes)) != len(content_hashes):
        raise RuntimeError("本地V4训练数据包含重复视频内容哈希")
    labels = np.asarray([item["label"] for item in records], dtype=np.float64)
    dataset = {
        "sample_ids": np.asarray([item["sample_id"] for item in records], dtype=np.int64),
        "asset_ids": np.asarray([item["asset_id"] for item in records], dtype=np.int64),
        "material_ids": np.asarray([item["material_id"] for item in records]),
        "content_sha256": np.asarray(content_hashes),
        "primary_industries": np.asarray([
            item["primary_industry"] for item in records
        ]),
        "date_ordinals": np.asarray([
            item["first_date"].toordinal() for item in records
        ], dtype=np.int32),
        "delivery_day_counts": np.asarray([
            item["delivery_day_count"] for item in records
        ], dtype=np.int16),
        "labels": labels,
        "features": np.vstack([
            item["features"] for item in records
        ]).astype(np.float32),
    }
    industry_counts: Dict[str, int] = {}
    for item in records:
        industry = item["primary_industry"]
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    manifest = {
        "cohort": cohort,
        "feature_version": feature_version,
        "label_version": LOCAL_LABEL_VERSION,
        "label_name": "label_gmv_per_delivery_day",
        "sample_count": int(labels.size),
        "positive_count": int((labels > 0).sum()),
        "zero_count": int((labels == 0).sum()),
        "feature_dim": int(dataset["features"].shape[1]),
        "feature_group_ranges": {
            key: [value[0], value[1]] for key, value in FEATURE_GROUP_RANGES.items()
        },
        "min_first_stat_date": records[0]["first_date"].isoformat(),
        "max_first_stat_date": records[-1]["first_date"].isoformat(),
        "primary_industry_counts": industry_counts,
        "source_type": "local_raw_feature_archive",
        "sample_id_source": "asset_id",
        "dataset_sha256": _dataset_digest(dataset),
    }
    return dataset, manifest


def load_semantic_dataset(
    cohort: str,
    feature_version: str,
    first_date_from: Optional[datetime.date] = None,
    first_date_to: Optional[datetime.date] = None,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """读取成熟日均GMV标签和对应的多模态V2特征。"""
    from sqlalchemy import text

    from material_quality.database import engine

    date_filter = ""
    params: Dict[str, object] = {
        "cohort": cohort,
        "feature_version": feature_version,
        "ready_status": "ready",
    }
    if first_date_from is not None:
        date_filter += " AND s.first_stat_date >= :first_date_from"
        params["first_date_from"] = first_date_from
    if first_date_to is not None:
        date_filter += " AND s.first_stat_date <= :first_date_to"
        params["first_date_to"] = first_date_to

    with engine.connect() as conn:
        rows = list(conn.execute(text("""
            SELECT
                s.id AS sample_id,
                s.asset_id,
                s.material_id,
                s.label_version,
                s.first_stat_date,
                s.delivery_day_count,
                s.label_gmv_per_delivery_day,
                f.content_sha256,
                f.duration,
                f.width,
                f.height,
                f.fps,
                f.bit_rate,
                f.file_size,
                f.has_audio,
                f.visual_vector,
                f.visual_vector_dim,
                f.visual_temporal_vector,
                f.visual_temporal_dim,
                f.text_vector,
                f.text_vector_dim,
                f.ocr_stats_vector,
                f.ocr_stats_dim,
                f.asr_text_vector,
                f.asr_text_vector_dim,
                f.audio_stats_vector,
                f.audio_stats_dim
            FROM material_quality_semantic_feature AS f
            INNER JOIN material_quality_sample AS s ON s.asset_id = f.asset_id
            WHERE f.feature_version = :feature_version
              AND s.cohort = :cohort
              AND s.label_status = :ready_status
              AND s.is_window_closed = 1
              AND s.delivery_day_count >= 2
              {date_filter}
            ORDER BY s.first_stat_date, s.material_id
        """.format(date_filter=date_filter)), params).mappings())
    if not rows:
        raise RuntimeError("没有可用于训练的成熟多模态样本")

    feature_rows = [build_semantic_feature_row(dict(row)) for row in rows]
    content_hashes = [str(row["content_sha256"]) for row in rows]
    if len(set(content_hashes)) != len(content_hashes):
        raise RuntimeError("多模态训练数据包含重复视频内容哈希")
    label_versions = sorted({str(row["label_version"]) for row in rows})
    if len(label_versions) != 1:
        raise RuntimeError("多模态训练数据包含多个标签版本")

    labels = np.asarray([
        float(row["label_gmv_per_delivery_day"]) for row in rows
    ], dtype=np.float64)
    if (labels < 0).any() or not np.isfinite(labels).all():
        raise RuntimeError("日均GMV标签包含负数或非有限数值")

    dataset = {
        "sample_ids": np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64),
        "asset_ids": np.asarray([int(row["asset_id"]) for row in rows], dtype=np.int64),
        "material_ids": np.asarray([str(row["material_id"]) for row in rows]),
        "content_sha256": np.asarray(content_hashes),
        "date_ordinals": np.asarray([
            row["first_stat_date"].toordinal() for row in rows
        ], dtype=np.int32),
        "delivery_day_counts": np.asarray([
            int(row["delivery_day_count"]) for row in rows
        ], dtype=np.int16),
        "labels": labels,
        "features": np.vstack(feature_rows).astype(np.float32),
    }
    manifest = {
        "cohort": cohort,
        "feature_version": feature_version,
        "label_version": label_versions[0],
        "label_name": "label_gmv_per_delivery_day",
        "sample_count": int(labels.size),
        "positive_count": int((labels > 0).sum()),
        "zero_count": int((labels == 0).sum()),
        "feature_dim": int(dataset["features"].shape[1]),
        "feature_group_ranges": {
            key: [value[0], value[1]] for key, value in FEATURE_GROUP_RANGES.items()
        },
        "min_first_stat_date": datetime.date.fromordinal(
            int(dataset["date_ordinals"].min())
        ).isoformat(),
        "max_first_stat_date": datetime.date.fromordinal(
            int(dataset["date_ordinals"].max())
        ).isoformat(),
        "dataset_sha256": _dataset_digest(dataset),
    }
    return dataset, manifest


def save_semantic_snapshot(
    output_dir: str,
    dataset: Dict[str, np.ndarray],
    manifest: dict,
) -> None:
    """保存不可变多模态训练快照。"""
    os.makedirs(output_dir, exist_ok=False)
    np.savez_compressed(os.path.join(output_dir, "dataset.npz"), **dataset)
    payload = dict(manifest)
    payload["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(
        os.path.join(output_dir, "dataset_manifest.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)


def load_semantic_snapshot(snapshot_dir: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """读取已冻结的多模态训练快照并校验摘要。"""
    dataset_path = os.path.join(snapshot_dir, "dataset.npz")
    manifest_path = os.path.join(snapshot_dir, "dataset_manifest.json")
    with np.load(dataset_path, allow_pickle=False) as source:
        dataset = {key: source[key] for key in source.files}
    with open(manifest_path, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    actual_digest = _dataset_digest(dataset)
    if actual_digest != manifest.get("dataset_sha256"):
        raise RuntimeError("多模态训练快照摘要校验失败")
    if int(dataset["features"].shape[1]) != len(FEATURE_NAMES):
        raise RuntimeError("多模态训练快照特征维度异常")
    return dataset, manifest

"""素材质量训练数据读取、特征构造和快照工具。"""

import datetime
import hashlib
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


VISUAL_BASE_FEATURE_NAMES = [
    "红色均值",
    "绿色均值",
    "蓝色均值",
    "红色标准差",
    "绿色标准差",
    "蓝色标准差",
    "亮度均值",
    "亮度标准差",
    "饱和度均值",
    "边缘强度",
] + ["亮度直方图{:02d}".format(index) for index in range(16)]

VISUAL_FEATURE_NAMES = []
for prefix in ("全帧均值", "全帧标准差", "首帧"):
    VISUAL_FEATURE_NAMES.extend(
        ["{}_{}".format(prefix, name) for name in VISUAL_BASE_FEATURE_NAMES]
    )
VISUAL_FEATURE_NAMES.extend(["帧间变化均值", "帧间变化标准差"])

MEDIA_FEATURE_NAMES = [
    "视频时长对数",
    "文件大小对数",
    "码率对数",
    "帧率",
    "宽高比",
    "百万像素数",
    "是否竖屏",
    "是否方屏",
]

FEATURE_NAMES = VISUAL_FEATURE_NAMES + MEDIA_FEATURE_NAMES
VISUAL_FEATURE_DIM = len(VISUAL_FEATURE_NAMES)


def _build_media_features(row: dict) -> List[float]:
    """构造只依赖素材本身的媒体属性，不引入投放结果。"""
    duration = max(float(row.get("duration") or 0), 0.0)
    file_size = max(float(row.get("file_size") or 0), 0.0)
    bit_rate = max(float(row.get("bit_rate") or 0), 0.0)
    fps = max(float(row.get("fps") or 0), 0.0)
    width = max(int(row.get("width") or 0), 0)
    height = max(int(row.get("height") or 0), 0)
    aspect_ratio = float(width) / float(height) if height > 0 else 0.0
    megapixels = float(width * height) / 1000000.0
    is_portrait = 1.0 if height > width else 0.0
    is_square = 1.0 if width > 0 and height > 0 and abs(width - height) <= 2 else 0.0
    return [
        math.log1p(duration),
        math.log1p(file_size),
        math.log1p(bit_rate),
        fps,
        aspect_ratio,
        megapixels,
        is_portrait,
        is_square,
    ]


def build_feature_row(row: dict) -> np.ndarray:
    """解码80维视觉向量并拼接媒体属性。"""
    vector_dim = int(row.get("visual_vector_dim") or 0)
    raw_vector = bytes(row.get("visual_vector") or b"")
    if vector_dim != VISUAL_FEATURE_DIM:
        raise ValueError("视觉向量维度异常，期望{}，实际{}".format(
            VISUAL_FEATURE_DIM,
            vector_dim,
        ))
    visual = np.frombuffer(raw_vector, dtype="<f4")
    if visual.size != VISUAL_FEATURE_DIM:
        raise ValueError("视觉向量二进制长度与维度不一致")
    combined = np.concatenate((
        visual.astype(np.float64),
        np.asarray(_build_media_features(row), dtype=np.float64),
    ))
    if not np.isfinite(combined).all():
        raise ValueError("训练特征包含非有限数值")
    return combined


def _dataset_digest(dataset: Dict[str, np.ndarray]) -> str:
    """计算训练快照摘要，保证后续评估可追溯。"""
    digest = hashlib.sha256()
    for key in ("sample_ids", "asset_ids", "date_ordinals", "labels", "features"):
        values = np.ascontiguousarray(dataset[key])
        digest.update(key.encode("utf-8"))
        digest.update(values.tobytes())
    for key in ("material_ids", "content_sha256"):
        digest.update(key.encode("utf-8"))
        digest.update("\n".join(dataset[key].tolist()).encode("utf-8"))
    return digest.hexdigest()


def load_feature_dataset(
    cohort: str,
    feature_version: str,
) -> Tuple[Dict[str, np.ndarray], Dict[Tuple[str, int], int], dict]:
    """从数据库读取冻结训练所需字段和真实标签分层数量。"""
    from sqlalchemy import text

    from material_quality.database import engine

    with engine.connect() as conn:
        rows = list(conn.execute(text("""
            SELECT
                s.id AS sample_id,
                s.asset_id,
                s.material_id,
                s.label_version,
                s.first_stat_date,
                s.window_end_date,
                s.label_gmv_7d,
                f.content_sha256,
                f.duration,
                f.width,
                f.height,
                f.fps,
                f.bit_rate,
                f.file_size,
                f.visual_vector,
                f.visual_vector_dim
            FROM material_quality_feature AS f
            INNER JOIN material_quality_sample AS s ON s.asset_id = f.asset_id
            WHERE f.feature_version = :feature_version
              AND s.cohort = :cohort
              AND s.label_status = 'ready'
              AND s.is_window_closed = 1
            ORDER BY s.first_stat_date, s.material_id
        """), {
            "cohort": cohort,
            "feature_version": feature_version,
        }).mappings())
        population_rows = list(conn.execute(text("""
            SELECT
                DATE_FORMAT(first_stat_date, '%Y-%m') AS month_key,
                IF(label_gmv_7d > 0, 1, 0) AS is_positive,
                COUNT(*) AS sample_count
            FROM material_quality_sample
            WHERE cohort = :cohort
              AND label_status = 'ready'
              AND is_window_closed = 1
            GROUP BY month_key, is_positive
            ORDER BY month_key, is_positive
        """), {"cohort": cohort}).mappings())
    if not rows:
        raise RuntimeError("没有可用于训练的完整窗口特征样本")

    feature_rows = [build_feature_row(dict(row)) for row in rows]
    content_hashes = [str(row["content_sha256"]) for row in rows]
    if len(set(content_hashes)) != len(content_hashes):
        raise RuntimeError("训练数据包含重复视频内容哈希")
    label_versions = sorted({str(row["label_version"]) for row in rows})
    if len(label_versions) != 1:
        raise RuntimeError("训练数据包含多个标签版本")

    dataset = {
        "sample_ids": np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64),
        "asset_ids": np.asarray([int(row["asset_id"]) for row in rows], dtype=np.int64),
        "material_ids": np.asarray([str(row["material_id"]) for row in rows]),
        "content_sha256": np.asarray(content_hashes),
        "date_ordinals": np.asarray([
            row["first_stat_date"].toordinal() for row in rows
        ], dtype=np.int32),
        "labels": np.asarray([float(row["label_gmv_7d"]) for row in rows], dtype=np.float64),
        "features": np.vstack(feature_rows).astype(np.float64),
    }
    population_counts = {
        (str(row["month_key"]), int(row["is_positive"])): int(row["sample_count"])
        for row in population_rows
    }
    manifest = {
        "cohort": cohort,
        "feature_version": feature_version,
        "label_version": label_versions[0],
        "sample_count": int(dataset["labels"].size),
        "positive_count": int((dataset["labels"] > 0).sum()),
        "zero_count": int((dataset["labels"] == 0).sum()),
        "feature_dim": int(dataset["features"].shape[1]),
        "feature_names": FEATURE_NAMES,
        "min_first_stat_date": datetime.date.fromordinal(
            int(dataset["date_ordinals"].min())
        ).isoformat(),
        "max_first_stat_date": datetime.date.fromordinal(
            int(dataset["date_ordinals"].max())
        ).isoformat(),
        "dataset_sha256": _dataset_digest(dataset),
    }
    return dataset, population_counts, manifest


def build_population_weights(
    date_ordinals: np.ndarray,
    labels: np.ndarray,
    population_counts: Dict[Tuple[str, int], int],
) -> np.ndarray:
    """按月份和标签还原完整历史样本的真实占比。"""
    strata = [
        (
            datetime.date.fromordinal(int(date_value)).strftime("%Y-%m"),
            1 if float(label) > 0 else 0,
        )
        for date_value, label in zip(date_ordinals, labels)
    ]
    feature_counts = Counter(strata)
    weights = np.asarray([
        float(population_counts.get(stratum, feature_counts[stratum]))
        / float(feature_counts[stratum])
        for stratum in strata
    ], dtype=np.float64)
    mean_weight = float(weights.mean()) if weights.size else 0.0
    if mean_weight <= 0:
        raise ValueError("训练样本权重均值必须大于0")
    return weights / mean_weight


def save_dataset_snapshot(
    output_dir: str,
    dataset: Dict[str, np.ndarray],
    population_counts: Dict[Tuple[str, int], int],
    manifest: dict,
) -> None:
    """保存不可变数据快照和可读清单。"""
    os.makedirs(output_dir, exist_ok=False)
    np.savez_compressed(os.path.join(output_dir, "dataset.npz"), **dataset)
    manifest_payload = dict(manifest)
    manifest_payload["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest_payload["population_strata"] = [
        {"month": key[0], "is_positive": key[1], "sample_count": value}
        for key, value in sorted(population_counts.items())
    ]
    with open(os.path.join(output_dir, "dataset_manifest.json"), "w", encoding="utf-8") as output:
        json.dump(manifest_payload, output, ensure_ascii=False, indent=2)

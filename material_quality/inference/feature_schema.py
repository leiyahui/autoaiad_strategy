"""线上推理使用的固定多模态特征结构和内容档案读取逻辑。"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


VISUAL_SEMANTIC_DIM = 2048
VISUAL_TEMPORAL_DIM = 12
OCR_SEMANTIC_DIM = 2048
OCR_STATS_DIM = 16
ASR_SEMANTIC_DIM = 2048
AUDIO_STATS_DIM = 13

MEDIA_FEATURE_NAMES = [
    "视频时长对数",
    "文件大小对数",
    "码率对数",
    "帧率",
    "宽高比",
    "百万像素数",
    "是否竖屏",
    "是否方屏",
    "是否有音轨",
    "是否有OCR文本",
    "是否有ASR文本",
]

FEATURE_GROUP_NAMES = {
    "visual": (
        ["画面语义{:04d}".format(index) for index in range(VISUAL_SEMANTIC_DIM)]
        + ["画面时序{:02d}".format(index) for index in range(VISUAL_TEMPORAL_DIM)]
    ),
    "ocr": (
        ["OCR语义{:04d}".format(index) for index in range(OCR_SEMANTIC_DIM)]
        + ["OCR统计{:02d}".format(index) for index in range(OCR_STATS_DIM)]
    ),
    "asr": (
        ["ASR语义{:04d}".format(index) for index in range(ASR_SEMANTIC_DIM)]
        + ["音频统计{:02d}".format(index) for index in range(AUDIO_STATS_DIM)]
    ),
    "media": MEDIA_FEATURE_NAMES,
}

FEATURE_NAMES: List[str] = []
FEATURE_GROUP_RANGES: Dict[str, Tuple[int, int]] = {}
for group_name in ("visual", "ocr", "asr", "media"):
    group_start = len(FEATURE_NAMES)
    FEATURE_NAMES.extend(FEATURE_GROUP_NAMES[group_name])
    FEATURE_GROUP_RANGES[group_name] = (group_start, len(FEATURE_NAMES))


def _decode_vector(
    raw_value: object,
    actual_dim: int,
    expected_dim: int,
    field_name: str,
    allow_missing: bool = False,
) -> np.ndarray:
    """按固定维度解码浮点向量，缺失模态使用零向量占位。"""
    if allow_missing and actual_dim == 0:
        return np.zeros(expected_dim, dtype=np.float32)
    if actual_dim != expected_dim:
        raise ValueError("{}维度异常，期望{}，实际{}".format(
            field_name,
            expected_dim,
            actual_dim,
        ))
    raw_vector = bytes(raw_value or b"")
    vector = np.frombuffer(raw_vector, dtype="<f4")
    if vector.size != expected_dim:
        raise ValueError("{}二进制长度与维度不一致".format(field_name))
    if not np.isfinite(vector).all():
        raise ValueError("{}包含非有限数值".format(field_name))
    return vector.astype(np.float32, copy=False)


def _build_media_features(row: dict) -> np.ndarray:
    """构造仅依赖视频文件本身的媒体属性。"""
    duration = max(float(row.get("duration") or 0), 0.0)
    file_size = max(float(row.get("file_size") or 0), 0.0)
    bit_rate = max(float(row.get("bit_rate") or 0), 0.0)
    fps = max(float(row.get("fps") or 0), 0.0)
    width = max(int(row.get("width") or 0), 0)
    height = max(int(row.get("height") or 0), 0)
    aspect_ratio = float(width) / float(height) if height > 0 else 0.0
    megapixels = float(width * height) / 1000000.0
    values = np.asarray([
        math.log1p(duration),
        math.log1p(file_size),
        math.log1p(bit_rate),
        fps,
        aspect_ratio,
        megapixels,
        1.0 if height > width else 0.0,
        1.0 if width > 0 and height > 0 and abs(width - height) <= 2 else 0.0,
        1.0 if int(row.get("has_audio") or 0) else 0.0,
        1.0 if int(row.get("text_vector_dim") or 0) > 0 else 0.0,
        1.0 if int(row.get("asr_text_vector_dim") or 0) > 0 else 0.0,
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("媒体属性包含非有限数值")
    return values


def build_semantic_feature_row(row: dict) -> np.ndarray:
    """按模型锁定顺序拼接画面、OCR、ASR和媒体属性。"""
    parts = [
        _decode_vector(
            row.get("visual_vector"),
            int(row.get("visual_vector_dim") or 0),
            VISUAL_SEMANTIC_DIM,
            "画面语义向量",
        ),
        _decode_vector(
            row.get("visual_temporal_vector"),
            int(row.get("visual_temporal_dim") or 0),
            VISUAL_TEMPORAL_DIM,
            "画面时序向量",
        ),
        _decode_vector(
            row.get("text_vector"),
            int(row.get("text_vector_dim") or 0),
            OCR_SEMANTIC_DIM,
            "OCR语义向量",
            allow_missing=True,
        ),
        _decode_vector(
            row.get("ocr_stats_vector"),
            int(row.get("ocr_stats_dim") or 0),
            OCR_STATS_DIM,
            "OCR统计向量",
        ),
        _decode_vector(
            row.get("asr_text_vector"),
            int(row.get("asr_text_vector_dim") or 0),
            ASR_SEMANTIC_DIM,
            "ASR语义向量",
            allow_missing=True,
        ),
        _decode_vector(
            row.get("audio_stats_vector"),
            int(row.get("audio_stats_dim") or 0),
            AUDIO_STATS_DIM,
            "音频统计向量",
        ),
        _build_media_features(row),
    ]
    combined = np.concatenate(parts).astype(np.float32, copy=False)
    if combined.size != len(FEATURE_NAMES):
        raise ValueError("多模态推理特征总维度异常")
    return combined


def _read_json(path: str, description: str) -> dict:
    """读取内容档案，并把损坏文件转换为清晰错误。"""
    try:
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("{}读取失败：{}".format(description, exc))
    if not isinstance(payload, dict):
        raise ValueError("{}必须是JSON对象".format(description))
    return payload


def _local_vector(
    vectors: object,
    key: str,
    expected_dim: int,
    allow_missing: bool = False,
) -> Tuple[Optional[bytes], int]:
    """读取内容级NPZ向量并校验固定维度。"""
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


def _build_local_semantic_feature_row(
    vectors: object,
    manifest: dict,
) -> np.ndarray:
    """把内容级原始档案转换为正式模型需要的固定特征。"""
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
    """按内容哈希读取一条线上多模态特征。"""
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

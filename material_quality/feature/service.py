"""素材质量特征构造的纯计算逻辑。"""

import struct
from typing import List, Optional, Tuple

import numpy as np


FRAME_WIDTH = 64
FRAME_HEIGHT = 64
FRAME_CHANNELS = 3
FRAME_FEATURE_DIM = 26
VISUAL_VECTOR_DIM = FRAME_FEATURE_DIM * 3 + 2

def parse_fraction(value: Optional[str]) -> float:
    """解析ffprobe返回的分数字符串。"""
    normalized = str(value or "0").strip()
    if not normalized:
        return 0.0
    if "/" not in normalized:
        try:
            return float(normalized)
        except ValueError:
            return 0.0
    numerator, denominator = normalized.split("/", 1)
    try:
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    except ValueError:
        return 0.0


def _frame_features(frame: np.ndarray) -> np.ndarray:
    """从缩略帧提取颜色、亮度、饱和度和边缘统计。"""
    normalized = frame.astype(np.float32) / 255.0
    channel_mean = normalized.mean(axis=(0, 1))
    channel_std = normalized.std(axis=(0, 1))
    luminance = (
        normalized[:, :, 0] * 0.299
        + normalized[:, :, 1] * 0.587
        + normalized[:, :, 2] * 0.114
    )
    saturation = normalized.max(axis=2) - normalized.min(axis=2)
    horizontal_edge = np.abs(np.diff(luminance, axis=1)).mean()
    vertical_edge = np.abs(np.diff(luminance, axis=0)).mean()
    histogram, _ = np.histogram(luminance, bins=16, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32) / float(luminance.size)
    return np.concatenate((
        channel_mean,
        channel_std,
        np.asarray([
            luminance.mean(),
            luminance.std(),
            saturation.mean(),
            (horizontal_edge + vertical_edge) / 2.0,
        ], dtype=np.float32),
        histogram,
    )).astype(np.float32)


def build_visual_vector(
    raw_frames: bytes,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> Tuple[List[float], int]:
    """把ffmpeg输出的连续RGB缩略帧聚合成固定长度低阶视觉向量。"""
    frame_size = width * height * FRAME_CHANNELS
    if not raw_frames or len(raw_frames) % frame_size != 0:
        raise ValueError("抽帧数据为空或长度不完整")
    frames = np.frombuffer(raw_frames, dtype=np.uint8).reshape(
        (-1, height, width, FRAME_CHANNELS)
    )
    per_frame = np.stack([_frame_features(frame) for frame in frames])
    if per_frame.shape[1] != FRAME_FEATURE_DIM:
        raise ValueError("单帧视觉特征维度异常")

    if frames.shape[0] > 1:
        normalized = frames.astype(np.float32) / 255.0
        motion = np.abs(np.diff(normalized, axis=0)).mean(axis=(1, 2, 3))
        motion_mean = float(motion.mean())
        motion_std = float(motion.std())
    else:
        motion_mean = 0.0
        motion_std = 0.0

    vector = np.concatenate((
        per_frame.mean(axis=0),
        per_frame.std(axis=0),
        per_frame[0],
        np.asarray([motion_mean, motion_std], dtype=np.float32),
    )).astype(np.float32)
    if vector.size != VISUAL_VECTOR_DIM:
        raise ValueError("聚合视觉特征维度异常")
    return vector.tolist(), int(frames.shape[0])


def pack_float32(values: List[float]) -> bytes:
    """将向量编码为紧凑且跨平台的float32二进制。"""
    if not values:
        raise ValueError("待编码向量不能为空")
    return struct.pack("<{}f".format(len(values)), *values)

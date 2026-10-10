"""素材视频的高分辨率关键帧、中文语义向量和OCR特征提取。"""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from material_quality.feature.service import pack_float32


DEFAULT_KEYFRAME_COUNT = 8
DEFAULT_FRAME_LONG_SIDE = 960
OCR_STATS_DIM = 16
VISUAL_TEMPORAL_DIM = 12
AUDIO_STATS_DIM = 13
DEFAULT_TEXT_CHUNK_CHARS = 40
DEFAULT_MAX_TEXT_CHUNKS = 32
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 180
DEFAULT_OPENING_SECONDS = 5.0
DEFAULT_OPENING_INTERVAL = 0.5
DEFAULT_UNIFORM_INTERVAL = 2.0
DEFAULT_SCENE_THRESHOLD = 0.30
DEFAULT_MAX_SCENE_CHANGES = 12
DEFAULT_ASR_SPLIT_MIN_SECONDS = 45.0
DEFAULT_ASR_SPLIT_CHUNK_MIN_SECONDS = 20.0


class FileSlotLease:
    """持有一个跨进程文件槽位，用完后释放锁和文件句柄。"""

    def __init__(self, handle: object):
        self.handle = handle

    def release(self) -> None:
        """释放槽位，允许其他素材使用额外并发资源。"""
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None

    def __enter__(self) -> "FileSlotLease":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()


class FileSlotPool:
    """通过文件锁限制本机多进程共享的额外并发数。"""

    def __init__(self, directory: str, prefix: str, slot_count: int):
        self.paths = []
        if directory and slot_count > 0:
            os.makedirs(directory, exist_ok=True)
            self.paths = [
                os.path.join(directory, "{}_{}.slot".format(prefix, index))
                for index in range(slot_count)
            ]

    def acquire(self) -> Optional[FileSlotLease]:
        """非阻塞申请一个槽位，全部占用时返回空值。"""
        for path in self.paths:
            handle = open(path, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return FileSlotLease(handle)
            except BlockingIOError:
                handle.close()
        return None


def find_audio_split_sample(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    minimum_chunk_seconds: float = DEFAULT_ASR_SPLIT_CHUNK_MIN_SECONDS,
    search_seconds: float = 8.0,
    window_seconds: float = 0.4,
) -> Optional[int]:
    """在音频中点附近选择能量最低的切分位置，减少截断话语。"""
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    minimum_samples = int(max(minimum_chunk_seconds, 0.1) * sample_rate)
    if values.size < minimum_samples * 2:
        return None
    midpoint = values.size // 2
    search_samples = int(max(search_seconds, 0.1) * sample_rate)
    window_samples = max(int(max(window_seconds, 0.05) * sample_rate), 1)
    start = max(minimum_samples, midpoint - search_samples)
    end = min(values.size - minimum_samples, midpoint + search_samples)
    if start >= end:
        return None
    half_window = max(window_samples // 2, 1)
    best_sample = midpoint
    best_energy = float("inf")
    for sample_index in range(start, end + 1, window_samples):
        window_start = max(sample_index - half_window, 0)
        window_end = min(sample_index + half_window, values.size)
        current = values[window_start:window_end]
        if current.size <= 0:
            continue
        energy = float(np.mean(np.square(current)))
        if energy < best_energy:
            best_energy = energy
            best_sample = sample_index
    return best_sample


def merge_asr_results(
    results: Sequence[Tuple[float, float, Dict[str, object]]],
) -> Dict[str, object]:
    """按时间偏移合并分段ASR结果，保留完整时间轴。"""
    text_parts = []
    segments = []
    language = ""
    weighted_probability = 0.0
    total_duration = 0.0
    for offset, duration, result in results:
        text_value = str(result.get("text") or "").strip()
        if text_value:
            text_parts.append(text_value)
        if not language:
            language = str(result.get("language") or "")
        weight = max(float(duration), 0.0)
        weighted_probability += float(
            result.get("language_probability") or 0.0
        ) * weight
        total_duration += weight
        for segment in result.get("segments") or []:
            record = dict(segment)
            record["start"] = float(record.get("start") or 0.0) + offset
            record["end"] = float(record.get("end") or 0.0) + offset
            segments.append(record)
    segments.sort(key=lambda item: float(item.get("start") or 0.0))
    return {
        "text": "\n".join(text_parts),
        "segments": segments,
        "language": language,
        "language_probability": (
            weighted_probability / total_duration if total_duration > 0 else 0.0
        ),
    }


def normalize_ocr_text(value: object) -> str:
    """归一化OCR文本，用于跨帧去重但保留中文和业务符号。"""
    return re.sub(r"\s+", "", str(value or "")).strip().casefold()


def aggregate_frame_embeddings(embeddings: np.ndarray) -> List[float]:
    """将逐帧语义向量聚合为均值和标准差，兼顾主体与画面变化。"""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("逐帧语义向量维度异常")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    aggregated = np.concatenate((normalized.mean(axis=0), normalized.std(axis=0)))
    return aggregated.astype(np.float32).tolist()


def split_text_chunks(
    text_value: str,
    chunk_chars: int = DEFAULT_TEXT_CHUNK_CHARS,
    max_chunks: int = DEFAULT_MAX_TEXT_CHUNKS,
) -> List[str]:
    """按中文CLIP上下文长度切分长文本，避免只保留开头内容。"""
    if chunk_chars <= 0 or max_chunks <= 0:
        raise ValueError("文本分块参数必须大于0")
    normalized = re.sub(r"\s+", " ", str(text_value or "")).strip()
    if not normalized:
        return []
    chunks = []
    for start in range(0, len(normalized), chunk_chars):
        chunk = normalized[start:start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    return chunks


def build_visual_temporal_features(
    embeddings: np.ndarray,
    frame_paths: Sequence[str],
) -> List[float]:
    """聚合相邻帧语义变化和像素变化，补充静态均值无法表达的节奏信息。"""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] <= 0:
        raise ValueError("逐帧语义向量维度异常")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    if normalized.shape[0] > 1:
        adjacent = 1.0 - np.sum(normalized[:-1] * normalized[1:], axis=1)
        first_last = float(1.0 - np.sum(normalized[0] * normalized[-1]))
    else:
        adjacent = np.zeros(1, dtype=np.float32)
        first_last = 0.0
    center = normalized.mean(axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm > 0:
        center = center / center_norm
    center_distances = 1.0 - np.sum(normalized * center, axis=1)

    grayscale_frames = []
    if frame_paths:
        from PIL import Image
        for path in frame_paths:
            with Image.open(path) as image:
                grayscale_frames.append(np.asarray(
                    image.convert("L").resize((96, 96)),
                    dtype=np.float32,
                ) / 255.0)
    if len(grayscale_frames) > 1:
        pixel_changes = np.asarray([
            float(np.mean(np.abs(current - previous)))
            for previous, current in zip(grayscale_frames[:-1], grayscale_frames[1:])
        ], dtype=np.float32)
    else:
        pixel_changes = np.zeros(1, dtype=np.float32)
    scene_change_ratio = float((pixel_changes >= 0.18).mean())
    features = [
        float(adjacent.mean()),
        float(adjacent.std()),
        float(adjacent.min()),
        float(adjacent.max()),
        first_last,
        float(center_distances.mean()),
        float(center_distances.std()),
        float(pixel_changes.mean()),
        float(pixel_changes.std()),
        float(pixel_changes.min()),
        float(pixel_changes.max()),
        scene_change_ratio,
    ]
    if len(features) != VISUAL_TEMPORAL_DIM or not all(
        math.isfinite(value) for value in features
    ):
        raise ValueError("画面变化特征维度或数值异常")
    return features


def extract_audio_waveform(
    video_path: str,
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
    max_audio_seconds: int = DEFAULT_MAX_AUDIO_SECONDS,
    timeout_seconds: int = 300,
) -> np.ndarray:
    """通过ffmpeg在内存中解码单声道音频，避免残留额外音频文件。"""
    if sample_rate <= 0 or max_audio_seconds <= 0:
        raise ValueError("音频采样率和最长时长必须大于0")
    command = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-t", str(max_audio_seconds), "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError("视频音频解码失败：{}".format(detail))
    waveform = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
    if waveform.size == 0:
        raise RuntimeError("视频音轨没有解码出有效波形")
    return waveform


def build_audio_stats(
    waveform: np.ndarray,
    sample_rate: int,
    asr_segments: Sequence[Dict[str, object]],
    asr_text: str,
) -> List[float]:
    """构造响度、停顿和语速等固定维音频统计。"""
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1 or values.size <= 0 or sample_rate <= 0:
        raise ValueError("音频波形或采样率异常")
    duration = values.size / float(sample_rate)
    absolute = np.abs(values)
    rms = float(np.sqrt(np.mean(values * values)))
    zero_crossing_rate = float(np.mean(values[:-1] * values[1:] < 0)) if values.size > 1 else 0.0
    frame_size = max(int(sample_rate * 0.03), 1)
    frame_count = values.size // frame_size
    if frame_count > 0:
        framed = values[:frame_count * frame_size].reshape(frame_count, frame_size)
        frame_rms = np.sqrt(np.mean(framed * framed, axis=1))
        silence_ratio = float((frame_rms < 0.015).mean())
        low_energy = max(float(np.quantile(frame_rms, 0.1)), 1e-6)
        high_energy = max(float(np.quantile(frame_rms, 0.9)), low_energy)
        dynamic_range_db = min(20.0 * math.log10(high_energy / low_energy), 80.0)
    else:
        silence_ratio = 0.0
        dynamic_range_db = 0.0
    speech_seconds = sum(max(
        float(segment.get("end") or 0) - float(segment.get("start") or 0),
        0.0,
    ) for segment in asr_segments)
    speech_ratio = min(speech_seconds / max(duration, 1e-6), 1.0)
    character_count = len(re.sub(r"\s+", "", str(asr_text or "")))
    log_probabilities = np.asarray([
        float(segment.get("avg_logprob") or 0) for segment in asr_segments
    ], dtype=np.float32)
    no_speech_probabilities = np.asarray([
        float(segment.get("no_speech_prob") or 0) for segment in asr_segments
    ], dtype=np.float32)
    features = [
        duration,
        rms,
        float(absolute.mean()),
        float(absolute.max()),
        float((absolute >= 0.99).mean()),
        zero_crossing_rate,
        silence_ratio,
        dynamic_range_db,
        speech_ratio,
        character_count / max(speech_seconds, 1e-6) if character_count else 0.0,
        len(asr_segments) * 60.0 / max(duration, 1e-6),
        float(log_probabilities.mean()) if log_probabilities.size else 0.0,
        float(no_speech_probabilities.mean()) if no_speech_probabilities.size else 0.0,
    ]
    if len(features) != AUDIO_STATS_DIM or not all(
        math.isfinite(value) for value in features
    ):
        raise ValueError("音频统计特征维度或数值异常")
    return features


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    """计算OCR四边形面积，兼容轻微倾斜的字幕框。"""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += float(point[0]) * float(next_point[1])
        area -= float(next_point[0]) * float(point[1])
    return abs(area) / 2.0


def _build_ocr_record(
    box: object,
    text_value: object,
    score_value: object,
    frame_index: int,
    width: int,
    height: int,
) -> Optional[Dict[str, object]]:
    """将不同RapidOCR版本的单行输出整理为稳定结构。"""
    normalized_text = normalize_ocr_text(text_value)
    if not normalized_text:
        return None
    try:
        points = [[float(point[0]), float(point[1])] for point in box]
    except (TypeError, ValueError, IndexError):
        points = []
    try:
        score = float(score_value)
    except (TypeError, ValueError):
        score = 0.0
    safe_width = max(int(width), 1)
    safe_height = max(int(height), 1)
    area_ratio = min(
        _polygon_area(points) / float(safe_width * safe_height),
        1.0,
    )
    if points:
        center_x = sum(point[0] for point in points) / len(points) / safe_width
        center_y = sum(point[1] for point in points) / len(points) / safe_height
    else:
        center_x = 0.5
        center_y = 0.5
    return {
        "frame_index": int(frame_index),
        "text": str(text_value or "").strip(),
        "normalized_text": normalized_text,
        "score": score,
        "box": points,
        "area_ratio": area_ratio,
        "center_x": min(max(center_x, 0.0), 1.0),
        "center_y": min(max(center_y, 0.0), 1.0),
    }


def parse_rapidocr_output(
    output: object,
    frame_index: int,
    width: int,
    height: int,
) -> List[Dict[str, object]]:
    """兼容RapidOCR新旧返回类型并输出逐行识别结果。"""
    if isinstance(output, tuple) and output:
        output = output[0]

    records = []
    if hasattr(output, "txts"):
        raw_boxes = getattr(output, "boxes", None)
        raw_texts = getattr(output, "txts", None)
        raw_scores = getattr(output, "scores", None)
        boxes = list(raw_boxes) if raw_boxes is not None else []
        texts = list(raw_texts) if raw_texts is not None else []
        scores = list(raw_scores) if raw_scores is not None else []
        raw_rows = zip(boxes, texts, scores)
    elif isinstance(output, dict):
        raw_boxes = output.get("boxes")
        raw_texts = output.get("txts")
        if raw_texts is None:
            raw_texts = output.get("texts")
        raw_scores = output.get("scores")
        boxes = list(raw_boxes) if raw_boxes is not None else []
        texts = list(raw_texts) if raw_texts is not None else []
        scores = list(raw_scores) if raw_scores is not None else []
        raw_rows = zip(boxes, texts, scores)
    else:
        raw_rows = []
        for row in output or []:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                raw_rows.append((row[0], row[1], row[2]))

    for box, text_value, score_value in raw_rows:
        record = _build_ocr_record(
            box,
            text_value,
            score_value,
            frame_index,
            width,
            height,
        )
        if record:
            records.append(record)
    return records


def build_ocr_features(
    records: Sequence[Dict[str, object]],
    frame_count: int,
) -> Tuple[str, List[float], List[Dict[str, object]]]:
    """聚合去重OCR文本、置信度和画面版式统计。"""
    ordered_unique = []
    seen_text = set()
    for record in records:
        normalized_text = str(record.get("normalized_text") or "")
        if normalized_text and normalized_text not in seen_text:
            seen_text.add(normalized_text)
            ordered_unique.append(str(record.get("text") or normalized_text))

    total_lines = len(records)
    unique_lines = len(ordered_unique)
    total_chars = sum(len(str(record.get("normalized_text") or "")) for record in records)
    unique_chars = sum(len(normalize_ocr_text(text_value)) for text_value in ordered_unique)
    text_frames = len({int(record.get("frame_index") or 0) for record in records})
    scores = np.asarray([float(record.get("score") or 0) for record in records])
    areas = np.asarray([float(record.get("area_ratio") or 0) for record in records])
    centers_x = np.asarray([float(record.get("center_x") or 0.5) for record in records])
    centers_y = np.asarray([float(record.get("center_y") or 0.5) for record in records])

    def safe_stat(values: np.ndarray, operation: str) -> float:
        if values.size == 0:
            return 0.0
        return float(getattr(values, operation)())

    top_ratio = float((centers_y < 0.33).mean()) if centers_y.size else 0.0
    bottom_ratio = float((centers_y > 0.67).mean()) if centers_y.size else 0.0
    duplicate_ratio = 1.0 - unique_lines / float(max(total_lines, 1))
    stats = [
        float(total_lines),
        float(unique_lines),
        float(total_chars),
        float(unique_chars),
        text_frames / float(max(frame_count, 1)),
        safe_stat(scores, "mean"),
        safe_stat(scores, "std"),
        safe_stat(scores, "max"),
        safe_stat(areas, "mean"),
        safe_stat(areas, "std"),
        safe_stat(areas, "max"),
        safe_stat(centers_x, "mean"),
        safe_stat(centers_y, "mean"),
        top_ratio,
        bottom_ratio,
        duplicate_ratio,
    ]
    if len(stats) != OCR_STATS_DIM or not all(math.isfinite(value) for value in stats):
        raise ValueError("OCR统计特征维度或数值异常")
    return "\n".join(ordered_unique), stats, list(records)


def build_keyframe_timestamps(
    duration: float,
    frame_count: int = DEFAULT_KEYFRAME_COUNT,
    scene_timestamps: Optional[Sequence[float]] = None,
) -> List[float]:
    """合并高频开场、场景边界和全程均匀点，控制总帧数上限。"""
    if frame_count <= 0:
        raise ValueError("关键帧数量必须大于0")
    safe_duration = max(float(duration or 0), 0.0)
    if safe_duration <= 0:
        return [0.0]

    # 避开视频文件声明时长的最末端，减少尾帧不存在导致的空输出。
    tail_margin = min(0.1, max(safe_duration * 0.01, 0.02))
    max_timestamp = max(safe_duration - tail_margin, 0.0)
    opening_end = min(DEFAULT_OPENING_SECONDS, max_timestamp)
    opening_count = int(math.floor(
        opening_end / DEFAULT_OPENING_INTERVAL
    )) + 1
    selected = [
        round(index * DEFAULT_OPENING_INTERVAL, 6)
        for index in range(opening_count)
    ][:frame_count]

    # 场景切换点前后各留一帧，后续可以直接判断转场两侧画面。
    for timestamp in list(scene_timestamps or [])[:DEFAULT_MAX_SCENE_CHANGES]:
        for candidate in (float(timestamp) - 0.2, float(timestamp)):
            bounded = min(max(candidate, 0.0), max_timestamp)
            if all(abs(bounded - current) > 0.08 for current in selected):
                selected.append(round(bounded, 6))
            if len(selected) >= frame_count:
                return sorted(selected[:frame_count])

    candidates = np.arange(
        0.0,
        max_timestamp + DEFAULT_UNIFORM_INTERVAL,
        DEFAULT_UNIFORM_INTERVAL,
    ).clip(0.0, max_timestamp).tolist()
    candidates.append(max_timestamp)
    while len(selected) < frame_count and candidates:
        available = [
            value for value in candidates
            if all(abs(value - current) > 1e-6 for current in selected)
        ]
        if not available:
            break
        # 每次选择离已有时间点最远的位置，使剩余帧均匀覆盖完整视频。
        next_value = max(
            available,
            key=lambda value: min(abs(value - current) for current in selected),
        )
        selected.append(next_value)
        candidates = available
    return sorted(round(value, 6) for value in selected)


def detect_scene_change_timestamps(
    video_path: str,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    max_scene_changes: int = DEFAULT_MAX_SCENE_CHANGES,
    timeout_seconds: int = 240,
) -> List[float]:
    """使用低分辨率FFmpeg场景分数检测硬切和明显转场时间。"""
    if threshold <= 0 or threshold >= 1 or max_scene_changes <= 0:
        raise ValueError("场景检测阈值和数量上限不合法")
    video_filter = "scale=160:-2,select=gt(scene\\,{:.4f}),showinfo".format(
        threshold
    )
    command = [
        "ffmpeg", "-nostats", "-loglevel", "info", "-i", video_path,
        "-an", "-vf", video_filter, "-frames:v", str(max_scene_changes),
        "-f", "null", "-",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError("视频场景变化检测失败：{}".format(detail))
    output = result.stderr.decode("utf-8", errors="replace")
    timestamps = []
    for match in re.finditer(r"pts_time:\s*([0-9.]+)", output):
        value = float(match.group(1))
        if not timestamps or abs(value - timestamps[-1]) >= 0.5:
            timestamps.append(value)
        if len(timestamps) >= max_scene_changes:
            break
    return timestamps


def build_frame_reasons(
    timestamps: Sequence[float],
    scene_timestamps: Sequence[float],
) -> List[List[str]]:
    """为每张帧记录采样原因，便于后续按开场、转场或全程筛选。"""
    records = []
    for timestamp in timestamps:
        reasons = []
        if timestamp <= DEFAULT_OPENING_SECONDS + 1e-6:
            reasons.append("opening")
        if any(abs(timestamp - scene_time) <= 0.08 for scene_time in scene_timestamps):
            reasons.append("scene_after")
        if any(abs(timestamp - (scene_time - 0.2)) <= 0.08 for scene_time in scene_timestamps):
            reasons.append("scene_before")
        if not reasons:
            reasons.append("timeline")
        records.append(reasons)
    return records


def extract_keyframes_with_timestamps(
    video_path: str,
    output_dir: str,
    duration: float,
    frame_count: int = DEFAULT_KEYFRAME_COUNT,
    long_side: int = DEFAULT_FRAME_LONG_SIDE,
    timeout_seconds: int = 240,
) -> Tuple[List[str], List[float], List[List[str]], List[float]]:
    """按开场、场景和全程策略抽帧，返回时间点和采样原因。"""
    if frame_count <= 0 or long_side <= 0:
        raise ValueError("关键帧数量和长边尺寸必须大于0")
    os.makedirs(output_dir, exist_ok=True)
    scene_timestamps = detect_scene_change_timestamps(video_path)
    requested_timestamps = build_keyframe_timestamps(
        duration,
        frame_count,
        scene_timestamps=scene_timestamps,
    )
    frame_reasons = build_frame_reasons(requested_timestamps, scene_timestamps)
    video_filter = "scale={}:{}:force_original_aspect_ratio=decrease".format(
        long_side,
        long_side,
    )
    paths = []
    actual_timestamps = []
    actual_reasons = []
    errors = []
    per_frame_timeout = max(
        30,
        int(timeout_seconds / max(len(requested_timestamps), 1)),
    )
    for frame_index, timestamp in enumerate(requested_timestamps):
        output_path = os.path.join(output_dir, "frame_{:03d}.jpg".format(frame_index))
        command = [
            "ffmpeg", "-v", "error", "-ss", "{:.6f}".format(timestamp),
            "-i", video_path, "-an", "-frames:v", "1", "-vf", video_filter,
            "-q:v", "3", "-y", output_path,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=per_frame_timeout,
            check=False,
        )
        if result.returncode == 0 and os.path.isfile(output_path):
            paths.append(output_path)
            actual_timestamps.append(timestamp)
            actual_reasons.append(frame_reasons[frame_index])
            continue
        errors.append(result.stderr.decode("utf-8", errors="replace")[-300:])
    if not paths:
        detail = errors[-1] if errors else "未产生图片文件"
        raise RuntimeError("视频高分辨率抽帧没有产生有效画面：{}".format(detail))
    return paths, actual_timestamps, actual_reasons, scene_timestamps


def extract_keyframes(
    video_path: str,
    output_dir: str,
    duration: float,
    frame_count: int = DEFAULT_KEYFRAME_COUNT,
    long_side: int = DEFAULT_FRAME_LONG_SIDE,
    timeout_seconds: int = 240,
) -> List[str]:
    """兼容旧调用，仅返回关键帧路径。"""
    paths, _, _, _ = extract_keyframes_with_timestamps(
        video_path,
        output_dir,
        duration,
        frame_count=frame_count,
        long_side=long_side,
        timeout_seconds=timeout_seconds,
    )
    return paths


class SemanticFeatureExtractor:
    """复用模型实例提取视觉、OCR、ASR和音频特征。"""

    def __init__(
        self,
        model_name: str,
        model_cache: str,
        torch_threads: int = 4,
        use_modelscope: bool = True,
        asr_model_name: str = "small",
        asr_model_cache: Optional[str] = None,
        asr_compute_type: str = "int8",
        asr_cpu_threads: int = 2,
        ocr_cpu_threads: int = 2,
        ocr_worker_count: int = 2,
        asr_num_workers: int = 2,
        asr_split_min_seconds: float = DEFAULT_ASR_SPLIT_MIN_SECONDS,
        parallel_slot_dir: str = "",
        asr_extra_slots: int = 0,
    ):
        try:
            import torch
            import cn_clip.clip as clip
            from rapidocr import RapidOCR
            from cn_clip.clip import load_from_name
        except ImportError as exc:
            raise RuntimeError("语义特征依赖尚未安装：{}".format(exc))
        torch.set_num_threads(max(int(torch_threads), 1))
        self.torch = torch
        self.clip = clip
        self.model_name = model_name
        os.makedirs(model_cache, exist_ok=True)
        self.model, self.preprocess = load_from_name(
            model_name,
            device="cpu",
            download_root=model_cache,
            use_modelscope=use_modelscope,
        )
        self.model.eval()
        self.ocr_cpu_threads = max(int(ocr_cpu_threads), 1)
        self.ocr_worker_count = max(int(ocr_worker_count), 1)
        ocr_engine_threads = max(
            self.ocr_cpu_threads // self.ocr_worker_count,
            1,
        )
        # ONNXRuntime默认会尝试使用较多CPU线程，多素材并发时反而会互相抢占。
        self.ocr_engines = [
            RapidOCR(params={
                "EngineConfig.onnxruntime.intra_op_num_threads": ocr_engine_threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
            for _ in range(self.ocr_worker_count)
        ]
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("ASR特征依赖尚未安装：{}".format(exc))
        self.asr_model_name = asr_model_name
        self.asr_num_workers = max(int(asr_num_workers), 1)
        self.asr_split_min_seconds = max(float(asr_split_min_seconds), 1.0)
        self.asr_slot_pool = FileSlotPool(
            parallel_slot_dir,
            "asr_extra",
            max(int(asr_extra_slots), 0),
        )
        local_asr_path = (
            os.path.join(asr_model_cache, asr_model_name)
            if asr_model_cache else ""
        )
        asr_model_source = (
            local_asr_path
            if local_asr_path and os.path.isfile(
                os.path.join(local_asr_path, "model.bin")
            )
            else asr_model_name
        )
        self.asr_model = WhisperModel(
            asr_model_source,
            device="cpu",
            compute_type=asr_compute_type,
            cpu_threads=max(int(asr_cpu_threads), 1),
            num_workers=self.asr_num_workers,
            download_root=asr_model_cache,
        )

    def _embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        """批量计算逐帧中文CLIP视觉侧向量。"""
        from PIL import Image

        images = []
        try:
            for path in image_paths:
                images.append(Image.open(path).convert("RGB"))
            inputs = self.torch.stack([
                self.preprocess(image) for image in images
            ])
            with self.torch.no_grad():
                embeddings = self.model.encode_image(inputs)
            return embeddings.detach().cpu().numpy().astype(np.float32)
        finally:
            for image in images:
                image.close()

    def _embed_text_details(self, text_value: str) -> Dict[str, object]:
        """保留逐块文本向量并生成兼容V2的聚合向量。"""
        chunks = split_text_chunks(text_value)
        if not chunks:
            return {
                "chunks": [],
                "embeddings": np.empty((0, 0), dtype=np.float32),
                "aggregated": None,
            }
        inputs = self.clip.tokenize(chunks)
        with self.torch.no_grad():
            embeddings = self.model.encode_text(inputs)
        values = embeddings.detach().cpu().numpy().astype(np.float32)
        return {
            "chunks": chunks,
            "embeddings": values,
            "aggregated": aggregate_frame_embeddings(values),
        }

    def _embed_text(self, text_value: str) -> Optional[List[float]]:
        """兼容旧调用方，仅返回长文本的聚合向量。"""
        return self._embed_text_details(text_value)["aggregated"]

    def embed_script_text(self, text_value: str) -> Tuple[np.ndarray, List[str]]:
        """复用已加载的中文CLIP文本侧生成正式模型所需脚本向量。"""
        details = self._embed_text_details(text_value)
        aggregated = details.get("aggregated")
        chunks = list(details.get("chunks") or [])
        if aggregated is None or not chunks:
            raise ValueError("脚本规范文本为空")
        vector = np.asarray(aggregated, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("脚本聚合向量无效")
        return vector, chunks

    def _transcribe_audio(self, waveform: np.ndarray) -> Dict[str, object]:
        """使用中文ASR生成逐段文本和置信度信息。"""
        segment_iterator, info = self.asr_model.transcribe(
            waveform,
            language="zh",
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=False,
        )
        records = []
        text_parts = []
        for segment in segment_iterator:
            text_value = str(segment.text or "").strip()
            if text_value:
                text_parts.append(text_value)
            records.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text_value,
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
            })
        return {
            "text": "\n".join(text_parts),
            "segments": records,
            "language": str(getattr(info, "language", "zh") or "zh"),
            "language_probability": float(
                getattr(info, "language_probability", 0.0) or 0.0
            ),
        }

    def _extract_ocr_partition(
        self,
        engine: object,
        frame_items: Sequence[Tuple[int, str]],
        frame_timestamps: Sequence[float],
    ) -> List[Dict[str, object]]:
        """使用独立OCR引擎处理一组帧，避免跨线程共享ONNX会话。"""
        from PIL import Image

        records = []
        for frame_index, frame_path in frame_items:
            with Image.open(frame_path) as image:
                width, height = image.size
            output = engine(frame_path)
            frame_records = parse_rapidocr_output(
                output,
                frame_index,
                width,
                height,
            )
            for record in frame_records:
                record["timestamp"] = frame_timestamps[frame_index]
            records.extend(frame_records)
        return records

    def _extract_ocr_records(
        self,
        frame_paths: Sequence[str],
        frame_timestamps: Sequence[float],
    ) -> List[Dict[str, object]]:
        """将48帧均匀分配给多个OCR引擎，并行识别后恢复时间顺序。"""
        partitions = [[] for _ in self.ocr_engines]
        for frame_index, frame_path in enumerate(frame_paths):
            partitions[frame_index % len(partitions)].append(
                (frame_index, frame_path)
            )
        active_items = [
            (engine, items)
            for engine, items in zip(self.ocr_engines, partitions)
            if items
        ]
        if len(active_items) == 1:
            return self._extract_ocr_partition(
                active_items[0][0],
                active_items[0][1],
                frame_timestamps,
            )
        records = []
        with ThreadPoolExecutor(max_workers=len(active_items)) as executor:
            futures = [
                executor.submit(
                    self._extract_ocr_partition,
                    engine,
                    items,
                    frame_timestamps,
                )
                for engine, items in active_items
            ]
            for future in futures:
                records.extend(future.result())
        records.sort(key=lambda item: int(item.get("frame_index") or 0))
        return records

    def _transcribe_audio_with_dynamic_parallelism(
        self,
        waveform: np.ndarray,
    ) -> Tuple[Dict[str, object], int]:
        """只在长音频且全局有空闲槽位时分段并行，避免CPU过量竞争。"""
        duration = len(waveform) / float(DEFAULT_AUDIO_SAMPLE_RATE)
        if duration < self.asr_split_min_seconds or self.asr_num_workers < 2:
            return self._transcribe_audio(waveform), 1
        lease = self.asr_slot_pool.acquire()
        if lease is None:
            return self._transcribe_audio(waveform), 1
        try:
            split_sample = find_audio_split_sample(waveform)
            if split_sample is None:
                return self._transcribe_audio(waveform), 1
            chunks = [waveform[:split_sample], waveform[split_sample:]]
            offsets = [0.0, split_sample / float(DEFAULT_AUDIO_SAMPLE_RATE)]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(self._transcribe_audio, chunk)
                    for chunk in chunks
                ]
                results = [
                    (
                        offset,
                        len(chunk) / float(DEFAULT_AUDIO_SAMPLE_RATE),
                        future.result(),
                    )
                    for offset, chunk, future in zip(offsets, chunks, futures)
                ]
            return merge_asr_results(results), 2
        finally:
            lease.release()

    def _extract_and_transcribe_audio(
        self,
        video_path: str,
    ) -> Tuple[np.ndarray, Dict[str, object], Dict[str, float]]:
        """在独立线程中解码和识别音频，使其与视觉和OCR阶段重叠。"""
        audio_started_at = time.monotonic()
        waveform = extract_audio_waveform(video_path)
        audio_seconds = time.monotonic() - audio_started_at
        asr_started_at = time.monotonic()
        result, chunk_count = self._transcribe_audio_with_dynamic_parallelism(
            waveform
        )
        return waveform, result, {
            "audio": audio_seconds,
            "asr": time.monotonic() - asr_started_at,
            "asr_chunks": float(chunk_count),
        }

    def extract(
        self,
        video_path: str,
        work_dir: str,
        keyframe_count: int = DEFAULT_KEYFRAME_COUNT,
        frame_long_side: int = DEFAULT_FRAME_LONG_SIDE,
    ) -> Dict[str, object]:
        """一次提取单个视频的完整多模态原始特征。"""
        from material_quality.feature.media_features import probe_video

        total_started_at = time.monotonic()
        stage_seconds = {}
        stage_started_at = time.monotonic()
        media_info = probe_video(video_path)
        stage_seconds["probe"] = time.monotonic() - stage_started_at
        frame_dir = os.path.join(work_dir, "frames")
        stage_started_at = time.monotonic()
        (
            frame_paths,
            frame_timestamps,
            frame_reasons,
            scene_change_timestamps,
        ) = extract_keyframes_with_timestamps(
            video_path,
            frame_dir,
            float(media_info["duration"]),
            frame_count=keyframe_count,
            long_side=frame_long_side,
        )
        stage_seconds["keyframes"] = time.monotonic() - stage_started_at

        # ASR与后续视觉、OCR使用不同模型，可安全重叠以缩短单条素材墙钟时间。
        executor = None
        asr_future = None
        if media_info["has_audio"]:
            executor = ThreadPoolExecutor(max_workers=1)
            asr_future = executor.submit(
                self._extract_and_transcribe_audio,
                video_path,
            )
        try:
            stage_started_at = time.monotonic()
            image_embeddings = self._embed_images(frame_paths)
            visual_vector = aggregate_frame_embeddings(image_embeddings)
            visual_temporal = build_visual_temporal_features(
                image_embeddings,
                frame_paths,
            )
            stage_seconds["visual"] = time.monotonic() - stage_started_at

            stage_started_at = time.monotonic()
            ocr_records = self._extract_ocr_records(
                frame_paths,
                frame_timestamps,
            )
            stage_seconds["ocr"] = time.monotonic() - stage_started_at

            asr_result = {
                "text": "",
                "segments": [],
                "language": "",
                "language_probability": 0.0,
            }
            waveform = None
            if asr_future is not None:
                waveform, asr_result, audio_stage_seconds = asr_future.result()
                stage_seconds.update(audio_stage_seconds)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        stage_started_at = time.monotonic()
        ocr_text, ocr_stats, normalized_records = build_ocr_features(
            ocr_records,
            len(frame_paths),
        )
        ocr_embedding_details = self._embed_text_details(ocr_text)
        text_embedding = ocr_embedding_details["aggregated"]
        stage_seconds["ocr_text"] = time.monotonic() - stage_started_at
        asr_embedding_details = {
            "chunks": [],
            "embeddings": np.empty((0, 0), dtype=np.float32),
            "aggregated": None,
        }
        asr_text_embedding = None
        audio_stats = [0.0] * AUDIO_STATS_DIM
        if waveform is not None:
            stage_started_at = time.monotonic()
            asr_embedding_details = self._embed_text_details(
                str(asr_result["text"])
            )
            asr_text_embedding = asr_embedding_details["aggregated"]
            audio_stats = build_audio_stats(
                waveform,
                DEFAULT_AUDIO_SAMPLE_RATE,
                asr_result["segments"],
                str(asr_result["text"]),
            )
            stage_seconds["asr_text"] = time.monotonic() - stage_started_at
        stage_seconds["total"] = time.monotonic() - total_started_at
        media_info.update({
            "sample_frame_count": len(frame_paths),
            "visual_vector": pack_float32(visual_vector),
            "visual_vector_dim": len(visual_vector),
            "visual_temporal_vector": pack_float32(visual_temporal),
            "visual_temporal_dim": len(visual_temporal),
            "ocr_text": ocr_text,
            "ocr_json": json.dumps(normalized_records, ensure_ascii=False),
            "ocr_stats_vector": pack_float32(ocr_stats),
            "ocr_stats_dim": len(ocr_stats),
            "text_vector": pack_float32(text_embedding) if text_embedding else None,
            "text_vector_dim": len(text_embedding) if text_embedding else 0,
            "asr_model_name": self.asr_model_name,
            "asr_language": asr_result["language"],
            "asr_language_probability": asr_result["language_probability"],
            "asr_text": asr_result["text"],
            "asr_json": json.dumps(asr_result["segments"], ensure_ascii=False),
            "asr_text_vector": (
                pack_float32(asr_text_embedding) if asr_text_embedding else None
            ),
            "asr_text_vector_dim": (
                len(asr_text_embedding) if asr_text_embedding else 0
            ),
            "audio_stats_vector": pack_float32(audio_stats),
            "audio_stats_dim": len(audio_stats),
            # 原始向量只进入本地特征库，不写入生产数据库的大字段。
            "raw_visual_embeddings": image_embeddings,
            "raw_frame_paths": list(frame_paths),
            "raw_frame_timestamps": frame_timestamps,
            "raw_frame_reasons": frame_reasons,
            "raw_scene_change_timestamps": scene_change_timestamps,
            "raw_ocr_chunks": list(ocr_embedding_details["chunks"]),
            "raw_ocr_chunk_embeddings": ocr_embedding_details["embeddings"],
            "raw_asr_chunks": list(asr_embedding_details["chunks"]),
            "raw_asr_chunk_embeddings": asr_embedding_details["embeddings"],
            "stage_seconds": {
                key: round(value, 3) for key, value in stage_seconds.items()
            },
        })
        return media_info

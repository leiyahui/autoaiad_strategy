"""素材视频的临时下载、媒体探测和低阶视觉特征提取。"""

import hashlib
import json
import os
import subprocess
from typing import Dict

import requests

from material_quality.feature.service import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    build_visual_vector,
    pack_float32,
    parse_fraction,
)


DEFAULT_MAX_VIDEO_BYTES = 512 * 1024 * 1024
DEFAULT_SAMPLE_FRAME_COUNT = 12


def download_video(
    url: str,
    destination: str,
    timeout_seconds: int = 300,
    max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
) -> Dict[str, object]:
    """流式下载视频，并在超过空间上限或下载不完整时删除临时文件。"""
    if not url:
        raise ValueError("视频下载地址不能为空")
    partial_path = destination + ".part"
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, timeout_seconds),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as response:
            response.raise_for_status()
            expected_size = int(response.headers.get("content-length") or 0)
            if expected_size > max_video_bytes:
                raise RuntimeError("视频大小超过单文件处理上限")
            with open(partial_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_video_bytes:
                        raise RuntimeError("视频下载过程中超过单文件处理上限")
                    output.write(chunk)
                    digest.update(chunk)
            if expected_size > 0 and downloaded != expected_size:
                raise RuntimeError("视频下载不完整")
        os.replace(partial_path, destination)
        return {"file_size": downloaded, "content_sha256": digest.hexdigest()}
    except Exception:
        for path in (partial_path, destination):
            if os.path.exists(path):
                os.remove(path)
        raise


def probe_video(video_path: str, timeout_seconds: int = 60) -> Dict[str, object]:
    """使用ffprobe读取视频和音轨信息。"""
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,width,height,"
        "r_frame_rate,avg_frame_rate,bit_rate,sample_rate,channels",
        "-of", "json", video_path,
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
        raise RuntimeError("视频媒体信息读取失败：{}".format(detail))
    try:
        raw_probe = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("视频媒体信息解析失败：{}".format(exc))

    streams = raw_probe.get("streams") or []
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        raise RuntimeError("下载内容中没有视频轨道")
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    format_info = raw_probe.get("format") or {}
    fps = parse_fraction(
        video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    )
    return {
        "duration": float(format_info.get("duration") or 0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "fps": fps,
        "bit_rate": int(
            format_info.get("bit_rate") or video_stream.get("bit_rate") or 0
        ),
        "file_size": int(format_info.get("size") or 0),
        "has_audio": audio_stream is not None,
        "raw_probe": raw_probe,
    }


def extract_raw_frames(
    video_path: str,
    duration: float,
    sample_frame_count: int = DEFAULT_SAMPLE_FRAME_COUNT,
    timeout_seconds: int = 180,
) -> bytes:
    """均匀抽取小尺寸RGB帧，避免在磁盘产生大量图片文件。"""
    safe_duration = max(float(duration or 0), 1.0)
    frame_rate = max(float(sample_frame_count) / safe_duration, 0.05)
    video_filter = (
        "fps={:.8f},scale={}:{}:force_original_aspect_ratio=decrease,"
        "pad={}:{}:(ow-iw)/2:(oh-ih)/2:black"
    ).format(
        frame_rate,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        FRAME_HEIGHT,
    )
    command = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-an", "-vf", video_filter,
        "-frames:v", str(sample_frame_count),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
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
        raise RuntimeError("视频抽帧失败：{}".format(detail))
    if not result.stdout:
        raise RuntimeError("视频抽帧没有产生有效画面")
    return result.stdout


def extract_media_features(video_path: str) -> Dict[str, object]:
    """提取媒体属性和固定长度视觉统计向量。"""
    media_info = probe_video(video_path)
    raw_frames = extract_raw_frames(video_path, float(media_info["duration"]))
    vector, frame_count = build_visual_vector(raw_frames)
    media_info["sample_frame_count"] = frame_count
    media_info["visual_vector"] = pack_float32(vector)
    media_info["visual_vector_dim"] = len(vector)
    return media_info

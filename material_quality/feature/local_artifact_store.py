"""素材质量原始Embedding和可复用分镜资料的本地版本化存储。"""

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Sequence

import numpy as np


_SAFE_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _json_default(value: object) -> object:
    """把日期和NumPy标量转换为稳定的JSON值。"""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("本地特征清单包含无法序列化的值：{}".format(type(value).__name__))


def _atomic_write_json(path: str, payload: Dict[str, object]) -> None:
    """同目录原子写入JSON，避免任务中断留下半文件。"""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".artifact_",
        suffix=".json.tmp",
        dir=directory,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(
                payload,
                output,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _packed_float32(value: object, expected_dim: int) -> np.ndarray:
    """把数据库兼容二进制向量恢复为本地float32数组。"""
    if value is None or expected_dim <= 0:
        return np.empty((0,), dtype=np.float32)
    array = np.frombuffer(value, dtype="<f4").copy()
    if array.size != expected_dim:
        raise ValueError("聚合向量实际维度与声明维度不一致")
    return array


def _validate_array(name: str, value: object) -> np.ndarray:
    """统一原始向量类型并拒绝非有限数值。"""
    array = np.asarray(value, dtype=np.float32)
    if array.size and not np.isfinite(array).all():
        raise ValueError("{}包含非有限数值".format(name))
    return array


def _file_sha256(path: str) -> str:
    """计算派生媒体文件哈希，便于后续校验本地档案完整性。"""
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class LocalRawFeatureStore:
    """按内容哈希去重保存原始向量，并按素材保存标签引用。"""

    def __init__(self, root_dir: str, feature_version: str):
        resolved_root = os.path.realpath(root_dir)
        if not resolved_root:
            raise ValueError("本地原始特征目录不能为空")
        if not _SAFE_VERSION_PATTERN.match(str(feature_version or "")):
            raise ValueError("特征版本包含不安全字符")
        self.root_dir = resolved_root
        self.feature_version = feature_version
        self.version_root = os.path.join(resolved_root, feature_version)
        self.content_root = os.path.join(self.version_root, "content")
        self.asset_root = os.path.join(self.version_root, "assets")
        os.makedirs(self.content_root, exist_ok=True)
        os.makedirs(self.asset_root, exist_ok=True)

    def _content_paths(self, content_sha256: str) -> Dict[str, str]:
        """返回内容级音频、向量、清单和关键帧路径。"""
        normalized_sha = str(content_sha256 or "").lower()
        if not _SHA256_PATTERN.match(normalized_sha):
            raise ValueError("视频内容哈希不是有效SHA256")
        directory = os.path.join(self.content_root, normalized_sha[:2])
        basename = os.path.join(directory, normalized_sha)
        return {
            "directory": directory,
            "vectors": basename + ".npz",
            "manifest": basename + ".json",
            "contact_sheet": basename + ".jpg",
            "audio_archive": basename + ".audio.opus",
            "keyframes": basename + ".frames",
        }

    def has_content(self, content_sha256: str) -> bool:
        """检查相同内容和特征版本是否已经完整保存。"""
        paths = self._content_paths(content_sha256)
        if not all(os.path.isfile(paths[name]) for name in ("vectors", "manifest")):
            return False
        if not os.path.isdir(paths["keyframes"]):
            return False
        try:
            with open(paths["manifest"], "r", encoding="utf-8") as input_file:
                manifest = json.load(input_file)
        except (OSError, ValueError, TypeError):
            return False
        if manifest.get("has_audio") and not os.path.isfile(paths["audio_archive"]):
            return False
        keyframe_files = manifest.get("keyframe_files") or []
        valid_keyframes = [
            item for item in keyframe_files
            if isinstance(item, dict) and item.get("file")
        ]
        return (
            len(valid_keyframes) == len(keyframe_files)
            and bool(valid_keyframes)
            and all(os.path.isfile(os.path.join(
                paths["keyframes"],
                str(item["file"]),
            )) for item in valid_keyframes)
        )

    def has_model_inputs(self, content_sha256: str) -> bool:
        """只检查聚合评分所需清单和向量，允许重媒体已经按策略清理。"""
        paths = self._content_paths(content_sha256)
        if not all(os.path.isfile(paths[name]) for name in ("vectors", "manifest")):
            return False
        try:
            with open(paths["manifest"], "r", encoding="utf-8") as input_file:
                manifest = json.load(input_file)
            with np.load(paths["vectors"], allow_pickle=False) as vectors:
                required_vectors = {
                    "visual_aggregate",
                    "visual_temporal",
                    "ocr_stats",
                    "audio_stats",
                }
                return (
                    manifest.get("content_sha256") == content_sha256
                    and required_vectors.issubset(set(vectors.files))
                )
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def content_manifest_path(self, content_sha256: str) -> str:
        """返回内容清单路径，供脚本理解阶段读取同一份证据。"""
        return self._content_paths(content_sha256)["manifest"]

    def prune_replay_media(self, content_sha256: str) -> int:
        """评分完成后删除关键帧和音频，仅保留原始向量与结构化文本。"""
        paths = self._content_paths(content_sha256)
        removed_bytes = 0
        for name in ("contact_sheet", "audio_archive"):
            path = paths[name]
            if os.path.isfile(path):
                removed_bytes += os.path.getsize(path)
                os.remove(path)
        keyframe_directory = paths["keyframes"]
        if os.path.isdir(keyframe_directory):
            for directory, _, filenames in os.walk(keyframe_directory):
                for filename in filenames:
                    path = os.path.join(directory, filename)
                    if os.path.isfile(path):
                        removed_bytes += os.path.getsize(path)
            shutil.rmtree(keyframe_directory)
        try:
            with open(paths["manifest"], "r", encoding="utf-8") as input_file:
                manifest = json.load(input_file)
        except (OSError, ValueError, TypeError):
            return removed_bytes
        manifest["archive_policy"] = "vectors_text_only_v1"
        manifest["media_pruned_at"] = datetime.datetime.now().isoformat(
            timespec="seconds"
        )
        manifest["media_pruned_bytes"] = int(removed_bytes)
        manifest["audio_archive_file"] = None
        manifest["contact_sheet_file"] = None
        _atomic_write_json(paths["manifest"], manifest)
        return removed_bytes

    def content_directory(self, content_sha256: str) -> str:
        """返回内容级特征目录，不向调用方暴露内部路径规则。"""
        return self._content_paths(content_sha256)["directory"]

    def _save_vectors(self, path: str, feature_info: Dict[str, object]) -> None:
        """压缩保存原始及V2聚合向量，后续训练可自由重组。"""
        arrays = {
            "visual_frames": _validate_array(
                "逐帧视觉向量",
                feature_info.get("raw_visual_embeddings", []),
            ),
            "ocr_chunks": _validate_array(
                "OCR分块向量",
                feature_info.get("raw_ocr_chunk_embeddings", []),
            ),
            "asr_chunks": _validate_array(
                "ASR分块向量",
                feature_info.get("raw_asr_chunk_embeddings", []),
            ),
            "visual_aggregate": _packed_float32(
                feature_info.get("visual_vector"),
                int(feature_info.get("visual_vector_dim") or 0),
            ),
            "visual_temporal": _packed_float32(
                feature_info.get("visual_temporal_vector"),
                int(feature_info.get("visual_temporal_dim") or 0),
            ),
            "ocr_aggregate": _packed_float32(
                feature_info.get("text_vector"),
                int(feature_info.get("text_vector_dim") or 0),
            ),
            "ocr_stats": _packed_float32(
                feature_info.get("ocr_stats_vector"),
                int(feature_info.get("ocr_stats_dim") or 0),
            ),
            "asr_aggregate": _packed_float32(
                feature_info.get("asr_text_vector"),
                int(feature_info.get("asr_text_vector_dim") or 0),
            ),
            "audio_stats": _packed_float32(
                feature_info.get("audio_stats_vector"),
                int(feature_info.get("audio_stats_dim") or 0),
            ),
            "frame_timestamps": _validate_array(
                "关键帧时间",
                feature_info.get("raw_frame_timestamps", []),
            ),
        }
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".vectors_",
            suffix=".npz.tmp",
            dir=directory,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as output:
                np.savez_compressed(output, **arrays)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def _save_contact_sheet(self, path: str, frame_paths: Sequence[str]) -> None:
        """保存按时间顺序排列的低分辨率关键帧拼图，供故事线理解复用。"""
        if not frame_paths:
            return
        from PIL import Image, ImageOps

        tile_size = 160 if len(frame_paths) > 24 else 240
        column_count = min(8 if len(frame_paths) > 24 else 4, len(frame_paths))
        row_count = (len(frame_paths) + column_count - 1) // column_count
        sheet = Image.new(
            "RGB",
            (column_count * tile_size, row_count * tile_size),
            color=(0, 0, 0),
        )
        try:
            for frame_index, frame_path in enumerate(frame_paths):
                with Image.open(frame_path) as image:
                    tile = ImageOps.contain(
                        image.convert("RGB"),
                        (tile_size, tile_size),
                    )
                    left = (frame_index % column_count) * tile_size
                    top = (frame_index // column_count) * tile_size
                    offset_x = left + (tile_size - tile.width) // 2
                    offset_y = top + (tile_size - tile.height) // 2
                    sheet.paste(tile, (offset_x, offset_y))
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix=".contact_",
                suffix=".jpg.tmp",
                dir=directory,
            )
            os.close(file_descriptor)
            try:
                sheet.save(temporary_path, format="JPEG", quality=82, optimize=True)
                os.replace(temporary_path, path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
        finally:
            sheet.close()

    def _save_audio_archive(
        self,
        path: str,
        source_video_path: str,
        timeout_seconds: int = 600,
    ) -> int:
        """保存40kbps单声道Opus，允许后续更换ASR和声音模型。"""
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".audio_",
            suffix=".opus.tmp",
            dir=directory,
        )
        os.close(file_descriptor)
        try:
            command = [
                "ffmpeg", "-v", "error", "-i", source_video_path,
                "-vn", "-ac", "1", "-ar", "24000", "-c:a", "libopus",
                "-b:a", "40k", "-vbr", "on", "-compression_level", "10",
                "-application", "audio", "-f", "opus", "-y", temporary_path,
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
                raise RuntimeError("音频档案生成失败：{}".format(detail))
            if os.path.getsize(temporary_path) <= 0:
                raise ValueError("音频档案文件为空")
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return os.path.getsize(path)

    def _save_keyframes(
        self,
        directory: str,
        frame_paths: Sequence[str],
        timestamps: Sequence[float],
        reasons: Optional[Sequence[Sequence[str]]] = None,
    ) -> Sequence[Dict[str, object]]:
        """压缩保存逐张有序WebP关键帧，供多模态模型直接读取。"""
        if not frame_paths or len(frame_paths) != len(timestamps):
            raise ValueError("关键帧路径和时间戳必须存在且数量一致")
        parent_directory = os.path.dirname(directory)
        os.makedirs(parent_directory, exist_ok=True)
        temporary_directory = tempfile.mkdtemp(
            prefix=".keyframes_",
            dir=parent_directory,
        )
        records = []
        try:
            for frame_index, (frame_path, timestamp) in enumerate(zip(
                frame_paths,
                timestamps,
            )):
                if not os.path.isfile(frame_path):
                    raise ValueError("关键帧文件不存在：{}".format(frame_index))
                filename = "frame_{:03d}_{:010.3f}.webp".format(
                    frame_index,
                    float(timestamp),
                )
                from PIL import Image
                with Image.open(frame_path) as image:
                    converted = image.convert("RGB")
                    converted.thumbnail((640, 640))
                    converted.save(
                        os.path.join(temporary_directory, filename),
                        format="WEBP",
                        quality=75,
                        method=4,
                    )
                records.append({
                    "index": frame_index,
                    "timestamp": float(timestamp),
                    "file": filename,
                    "reasons": list(reasons[frame_index]) if reasons else [],
                })
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            os.replace(temporary_directory, directory)
        finally:
            if os.path.isdir(temporary_directory):
                shutil.rmtree(temporary_directory)
        return records

    def save(
        self,
        target: Dict[str, object],
        content_sha256: str,
        feature_info: Dict[str, object],
        source_video_path: Optional[str] = None,
    ) -> str:
        """原子保存帧序列、音频、特征和素材级标签引用，返回内容目录。"""
        paths = self._content_paths(content_sha256)
        if not self.has_content(content_sha256):
            self._save_vectors(paths["vectors"], feature_info)
            audio_archive_bytes = 0
            if bool(feature_info.get("has_audio")):
                audio_archive_bytes = self._save_audio_archive(
                    paths["audio_archive"],
                    str(source_video_path or ""),
                )
            keyframe_files = self._save_keyframes(
                paths["keyframes"],
                feature_info.get("raw_frame_paths", []),
                feature_info.get("raw_frame_timestamps", []),
                feature_info.get("raw_frame_reasons", []),
            )
            self._save_contact_sheet(
                paths["contact_sheet"],
                feature_info.get("raw_frame_paths", []),
            )
            content_manifest = {
                "schema_version": "material_quality_raw_v2",
                "feature_version": self.feature_version,
                "content_sha256": content_sha256,
                "visual_model_name": feature_info.get("visual_model_name"),
                "asr_model_name": feature_info.get("asr_model_name"),
                "asr_language": feature_info.get("asr_language"),
                "asr_language_probability": feature_info.get(
                    "asr_language_probability"
                ),
                "duration": feature_info.get("duration"),
                "width": feature_info.get("width"),
                "height": feature_info.get("height"),
                "fps": feature_info.get("fps"),
                "bit_rate": feature_info.get("bit_rate"),
                "has_audio": bool(feature_info.get("has_audio")),
                "sample_frame_count": feature_info.get("sample_frame_count"),
                "frame_timestamps": feature_info.get("raw_frame_timestamps", []),
                "frame_reasons": feature_info.get("raw_frame_reasons", []),
                "scene_change_timestamps": feature_info.get(
                    "raw_scene_change_timestamps", []
                ),
                "ocr_text": feature_info.get("ocr_text") or "",
                "ocr_records": json.loads(feature_info.get("ocr_json") or "[]"),
                "ocr_chunks": feature_info.get("raw_ocr_chunks", []),
                "asr_text": feature_info.get("asr_text") or "",
                "asr_segments": json.loads(feature_info.get("asr_json") or "[]"),
                "asr_chunks": feature_info.get("raw_asr_chunks", []),
                "probe": feature_info.get("raw_probe") or {},
                "extractor_config": feature_info.get("extractor_config") or {},
                "stage_seconds": feature_info.get("stage_seconds") or {},
                "vectors_file": os.path.basename(paths["vectors"]),
                "archive_policy": "event_frames_opus_v1",
                "audio_archive_file": (
                    os.path.basename(paths["audio_archive"])
                    if audio_archive_bytes > 0 else None
                ),
                "audio_archive_bytes": audio_archive_bytes,
                "audio_archive_sha256": (
                    _file_sha256(paths["audio_archive"])
                    if audio_archive_bytes > 0 else None
                ),
                "audio_archive_config": (
                    {"codec": "opus", "bit_rate": 40000, "channels": 1}
                    if audio_archive_bytes > 0 else None
                ),
                "keyframe_config": {
                    "format": "webp",
                    "max_long_side": 640,
                    "quality": 75,
                },
                "keyframe_directory": os.path.basename(paths["keyframes"]),
                "keyframe_files": keyframe_files,
                "contact_sheet_file": (
                    os.path.basename(paths["contact_sheet"])
                    if os.path.isfile(paths["contact_sheet"])
                    else None
                ),
            }
            _atomic_write_json(paths["manifest"], content_manifest)
        self.save_asset_reference(target, content_sha256)
        return paths["directory"]

    def save_asset_reference(
        self,
        target: Dict[str, object],
        content_sha256: str,
    ) -> None:
        """为同内容的不同素材保存独立标签和审计上下文。"""
        asset_id = int(target["asset_id"])
        reference = {
            "schema_version": "material_quality_asset_ref_v1",
            "asset_id": asset_id,
            "material_id": str(target.get("material_id") or ""),
            "cohort": str(target.get("cohort") or "all_industries_v1"),
            "feature_version": self.feature_version,
            "content_sha256": content_sha256,
            "label_status": target.get("label_status"),
            "label_gmv_7d": target.get("label_gmv_7d"),
            "label_gmv_per_delivery_day": target.get(
                "label_gmv_per_delivery_day"
            ),
            "delivery_day_count": target.get("delivery_day_count"),
            "first_stat_date": target.get("first_stat_date"),
            "primary_industry": target.get("primary_industry"),
            "industry_count": target.get("industry_count"),
        }
        _atomic_write_json(
            os.path.join(self.asset_root, "{}.json".format(asset_id)),
            reference,
        )

#!/usr/bin/env python3
"""使用Codex登录态持续调度GPT-5.6 Luna抽取本地素材脚本。"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BACKEND_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from material_quality.feature.local_agent_script_features import (
    ChineseCLIPScriptEmbedder,
    DEFAULT_MODEL_NAME,
    DEFAULT_SCRIPT_FEATURE_VERSION,
    save_local_script_feature,
)
from material_quality.feature.script_agent_features import (
    SCRIPT_ANALYSIS_SCHEMA,
    SYSTEM_INSTRUCTIONS,
    _validate_script_analysis,
    has_script_feature,
)
from plan_logger import (
    log_error,
    log_info,
    log_warn,
    set_process_name,
)


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_SOURCE_VERSION = "multimodal_cnclip_whisper_replayable_v4"
DEFAULT_PROMPT_VERSION = "codex_exec_material_script_v2"
DEFAULT_RUNTIME_DIRECTORY = "material_quality/.runtime/agent_script_auto"
DEFAULT_SNAPSHOT_PATH = (
    "material_quality/.runtime/training/"
    "mq_multimodal_v4_20260820_1334/snapshot/dataset.npz"
)


def read_codex_usage(path: str) -> dict:
    """只读取结构化用量事件，不把素材原文写入进度摘要。"""
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "turn.completed":
                    for key, value in (event.get("usage") or {}).items():
                        if isinstance(value, (int, float)):
                            result[key] = result.get(key, 0) + value
    except OSError:
        return {}
    return result


FORBIDDEN_SOURCE_FIELDS = {
    "asset_id",
    "delivery_day_counts",
    "delivery_days",
    "gmv",
    "label",
    "labels",
    "label_gmv_per_delivery_day",
    "sample_id",
    "training_data",
}


def _now_text() -> str:
    """返回便于状态文件和日志阅读的本地时间。"""
    return datetime.datetime.now().isoformat(timespec="seconds")


def _file_sha256(path: str) -> str:
    """计算文件摘要，保证每次输入和输出均可审计。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    """计算文本摘要。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: str) -> dict:
    """读取JSON对象并拒绝非对象根节点。"""
    with open(path, "r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("JSON根节点必须是对象：{}".format(path))
    return payload


def _atomic_write_json(path: str, payload: object) -> None:
    """原子保存JSON，避免状态读取方看到半截内容。"""
    directory = os.path.dirname(os.path.realpath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".json_",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _atomic_write_text(path: str, value: str) -> None:
    """原子保存文本。"""
    directory = os.path.dirname(os.path.realpath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".text_",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _normalize_path(repo_root: str, path: str) -> str:
    """把配置路径统一为绝对路径。"""
    if os.path.isabs(path):
        return os.path.realpath(path)
    return os.path.realpath(os.path.join(repo_root, path))


def load_snapshot_hashes(snapshot_path: str) -> List[str]:
    """只从冻结快照读取内容哈希，不加载GMV标签进入Agent输入。"""
    with np.load(snapshot_path, allow_pickle=False) as dataset:
        if "content_sha256" not in dataset.files:
            raise ValueError("冻结快照缺少content_sha256")
        hashes = [str(value).lower() for value in dataset["content_sha256"]]
    invalid = [value for value in hashes if len(value) != 64]
    if invalid:
        raise ValueError("冻结快照包含无效内容哈希")
    if len(set(hashes)) != len(hashes):
        raise ValueError("冻结快照包含重复内容哈希")
    return hashes


def _trim_text(value: object, maximum: int) -> str:
    """限制异常长文本，同时保留正常素材的完整OCR和ASR。"""
    text_value = str(value or "").strip()
    if len(text_value) <= maximum:
        return text_value
    return text_value[:maximum] + "\n[后续内容因安全长度限制省略]"


def _safe_ocr_records(records: object) -> List[dict]:
    """仅保留脚本理解需要的OCR时间、文本和位置证据。"""
    safe_records = []
    for item in list(records or [])[:240]:
        if not isinstance(item, dict):
            continue
        safe_records.append({
            "timestamp": item.get("timestamp"),
            "text": _trim_text(item.get("text"), 300),
            "score": item.get("score"),
            "center_x": item.get("center_x"),
            "center_y": item.get("center_y"),
            "area_ratio": item.get("area_ratio"),
        })
    return safe_records


def _safe_asr_segments(records: object) -> List[dict]:
    """仅保留脚本理解需要的ASR时间和文本证据。"""
    safe_records = []
    for item in list(records or [])[:240]:
        if not isinstance(item, dict):
            continue
        safe_records.append({
            "start": item.get("start"),
            "end": item.get("end"),
            "text": _trim_text(item.get("text"), 500),
            "no_speech_prob": item.get("no_speech_prob"),
        })
    return safe_records


def build_safe_evidence(manifest: dict, contact_sheet_path: str) -> dict:
    """通过字段白名单构造发给Luna的无标签证据。"""
    evidence = {
        "content_sha256": str(manifest.get("content_sha256") or "").lower(),
        "media": {
            "duration_seconds": manifest.get("duration"),
            "width": manifest.get("width"),
            "height": manifest.get("height"),
            "fps": manifest.get("fps"),
        },
        "timeline": {
            "frame_timestamps": list(manifest.get("frame_timestamps") or [])[:360],
            "scene_change_timestamps": list(
                manifest.get("scene_change_timestamps") or []
            )[:360],
        },
        "ocr": {
            "text": _trim_text(manifest.get("ocr_text"), 16000),
            "chunks": [
                _trim_text(value, 1200)
                for value in list(manifest.get("ocr_chunks") or [])[:80]
            ],
            "records": _safe_ocr_records(manifest.get("ocr_records")),
        },
        "asr": {
            "text": _trim_text(manifest.get("asr_text"), 20000),
            "chunks": [
                _trim_text(value, 1600)
                for value in list(manifest.get("asr_chunks") or [])[:100]
            ],
            "segments": _safe_asr_segments(manifest.get("asr_segments")),
        },
        "contact_sheet": {
            "sha256": _file_sha256(contact_sheet_path),
            "description": "全时间轴关键帧拼图，按从左到右、从上到下排列",
        },
    }
    if len(evidence["content_sha256"]) != 64:
        raise ValueError("内容档案哈希无效")
    return evidence


def build_output_schema(expected_hashes: Sequence[str] = ()) -> dict:
    """构造一个批次的严格结构化输出约束。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "content_sha256": ({"type":"string","enum":list(expected_hashes)}
                                           if expected_hashes else {"type":"string"}),
                        "analysis": SCRIPT_ANALYSIS_SCHEMA,
                    },
                    "required": ["content_sha256", "analysis"],
                },
            },
        },
        "required": ["items"],
    }


def _prompt_contract_sha256() -> str:
    """记录不随具体素材变化的提示词约束摘要。"""
    payload = {
        "version": DEFAULT_PROMPT_VERSION,
        "system": SYSTEM_INSTRUCTIONS,
        "schema": build_output_schema(),
        "rules": [
            "只使用提示内证据和附件拼图",
            "禁止读取其他本地文件",
            "禁止评价质量或预测商业表现",
            "禁止接收GMV、标签、投放天数和训练数据",
        ],
    }
    return _text_sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def build_agent_prompt(evidence_items: Sequence[dict]) -> str:
    """把安全证据直接放入提示，避免Agent自行读取源目录。"""
    expected_hashes = [item["content_sha256"] for item in evidence_items]
    sections = [
        SYSTEM_INSTRUCTIONS,
        "本批次共有{}条素材，附件图片与下方证据按相同顺序一一对应。".format(
            len(evidence_items)
        ),
        "全部必要证据已在本提示和附件中。不要调用终端、不要调用任何工具、"
        "不要读取其他本地文件，也不要联网搜索。",
        "输入不含且不得推断GMV、质量标签、投放天数、训练数据或商业表现。",
        "请为每个哈希返回一条items记录，顺序必须与以下列表一致，不可遗漏、"
        "重复或增加：{}".format(json.dumps(expected_hashes, ensure_ascii=False)),
    ]
    for index, evidence in enumerate(evidence_items, 1):
        sections.append("素材{}安全证据：\n{}".format(
            index,
            json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        ))
    sections.append("只返回符合输出结构约束的JSON对象，不输出额外解释。")
    return "\n\n".join(sections)


def parse_job_output(path: str, expected_hashes: Sequence[str]) -> List[Tuple[str, dict]]:
    """严格核对批次哈希和脚本结构，拒绝静默遗漏或串样。"""
    payload = _read_json(path)
    if set(payload) != {"items"} or not isinstance(payload.get("items"), list):
        raise ValueError("Codex批次输出根结构无效")
    normalized_expected = list(expected_hashes)
    if len(payload["items"]) != len(normalized_expected):
        raise ValueError("Codex批次输出数量与输入不一致")
    records = []
    seen = set()
    for index, row in enumerate(payload["items"]):
        if not isinstance(row, dict) or set(row) != {"content_sha256", "analysis"}:
            raise ValueError("Codex批次记录结构无效")
        content_sha256 = str(row.get("content_sha256") or "").lower()
        expected_sha256 = str(normalized_expected[index]).lower()
        if content_sha256 != expected_sha256:
            can_restore_single_trailing_character = (
                len(normalized_expected) == 1
                and len(content_sha256) == 63
                and len(expected_sha256) == 64
                and expected_sha256.startswith(content_sha256)
            )
            if can_restore_single_trailing_character:
                content_sha256 = expected_sha256
        if content_sha256 in seen:
            raise ValueError("Codex批次输出哈希重复：{}".format(content_sha256))
        _validate_script_analysis(row.get("analysis"))
        seen.add(content_sha256)
        records.append((content_sha256, row["analysis"]))
    actual_hashes = [content_sha256 for content_sha256, _ in records]
    if actual_hashes != normalized_expected:
        raise ValueError("Codex批次输出哈希或顺序与输入不一致")
    return records


class CodexScriptSupervisor:
    """维护三路Codex任务、断点状态和单实例锁。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = os.path.realpath(args.repo_root)
        self.snapshot_path = _normalize_path(self.repo_root, args.snapshot_path)
        self.source_root = _normalize_path(self.repo_root, args.source_root)
        self.output_root = _normalize_path(self.repo_root, args.output_root)
        self.runtime_directory = _normalize_path(self.repo_root, args.runtime_directory)
        self.model_cache = _normalize_path(self.repo_root, args.model_cache)
        self.schema_path = os.path.join(self.runtime_directory, "output_schema.json")
        self.status_path = os.path.join(self.runtime_directory, "status.json")
        self.retry_path = os.path.join(self.runtime_directory, "retry_counts.json")
        self.stop_path = os.path.join(self.runtime_directory, "STOP")
        self.jobs_directory = os.path.join(self.runtime_directory, "jobs")
        self.lock_handle = None
        self.stop_requested = False
        self.embedder: Optional[ChineseCLIPScriptEmbedder] = None
        self.active: List[dict] = []
        self.retry_counts: Dict[str, int] = {}
        self.retry_ready_at: Dict[str, float] = {}
        self.hashes: List[str] = []
        self.completed = set()
        self.started_at = time.time()
        self.initial_completed = 0
        self.job_sequence = 0

    def _acquire_lock(self) -> None:
        """持有进程级文件锁，避免重复启动消耗额度。"""
        os.makedirs(self.runtime_directory, exist_ok=True)
        lock_path = os.path.join(self.runtime_directory, "supervisor.lock")
        self.lock_handle = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("脚本自动任务已在运行，拒绝重复启动")
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(str(os.getpid()))
        self.lock_handle.flush()

    def _load_retries(self) -> None:
        """恢复每个哈希的失败次数。"""
        if not os.path.isfile(self.retry_path):
            return
        try:
            payload = _read_json(self.retry_path)
            self.retry_counts = {
                str(key): int(value)
                for key, value in payload.items()
                if len(str(key)) == 64 and int(value) > 0
            }
        except (OSError, ValueError, TypeError):
            log_warn("历史重试状态损坏，将从零重新计数")
            self.retry_counts = {}

    def _manifest_and_image(self, content_sha256: str) -> Tuple[str, str]:
        """定位无标签内容清单和全时间轴拼图。"""
        content_directory = os.path.join(
            self.source_root,
            self.args.source_feature_version,
            "content",
            content_sha256[:2],
        )
        manifest_path = os.path.join(content_directory, content_sha256 + ".json")
        manifest = _read_json(manifest_path)
        if str(manifest.get("content_sha256") or "").lower() != content_sha256:
            raise ValueError("内容清单哈希不一致：{}".format(content_sha256))
        image_path = os.path.join(
            content_directory,
            str(manifest.get("contact_sheet_file") or ""),
        )
        if not os.path.isfile(image_path):
            raise ValueError("全时间轴拼图不存在：{}".format(content_sha256))
        return manifest_path, image_path

    def _prepare_job(self, batch_hashes: Sequence[str]) -> dict:
        """为一个批次构造安全提示、附件和可追溯目录。"""
        self.job_sequence += 1
        job_id = "{}_{:04d}".format(
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            self.job_sequence,
        )
        job_directory = os.path.join(self.jobs_directory, job_id)
        os.makedirs(job_directory, exist_ok=False)
        evidence_items = []
        image_paths = []
        manifest_paths = {}
        for content_sha256 in batch_hashes:
            manifest_path, image_path = self._manifest_and_image(content_sha256)
            manifest = _read_json(manifest_path)
            evidence_items.append(build_safe_evidence(manifest, image_path))
            image_paths.append(image_path)
            manifest_paths[content_sha256] = manifest_path
        prompt = build_agent_prompt(evidence_items)
        prompt_path = os.path.join(job_directory, "prompt.txt")
        output_path = os.path.join(job_directory, "output.json")
        log_path = os.path.join(job_directory, "codex.log")
        state_path = os.path.join(job_directory, "state.json")
        schema_path = os.path.join(job_directory, "output_schema.json")
        # 用输入哈希枚举约束身份字段，防止模型抄写长哈希时丢失字符。
        _atomic_write_json(schema_path,build_output_schema(batch_hashes))
        _atomic_write_text(prompt_path, prompt)
        state = {
            "job_id": job_id,
            "status": "prepared",
            "content_sha256": list(batch_hashes),
            "created_at": _now_text(),
            "prompt_sha256": _text_sha256(prompt),
            "model": self.args.model,
            "reasoning_effort": self.args.reasoning_effort,
            "label_fields_excluded": True,
        }
        _atomic_write_json(state_path, state)
        return {
            "job_id": job_id,
            "directory": job_directory,
            "hashes": list(batch_hashes),
            "images": image_paths,
            "manifests": manifest_paths,
            "prompt": prompt,
            "prompt_path": prompt_path,
            "output_path": output_path,
            "log_path": log_path,
            "state_path": state_path,
            "schema_path": schema_path,
            "state": state,
        }

    def _start_job(self, batch_hashes: Sequence[str]) -> None:
        """启动一个只读、无持久会话的Codex Luna进程。"""
        job = self._prepare_job(batch_hashes)
        command = [
            self.args.codex_binary,
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            self.args.model,
            "--cd",
            job["directory"],
            "-c",
            'model_reasoning_effort="{}"'.format(self.args.reasoning_effort),
        ]
        for image_path in job["images"]:
            command.extend(["--image", image_path])
        # 图片参数可接收多个值，后置已命名参数用于明确结束图片列表。
        command.extend([
            "--output-schema",
            job["schema_path"],
            "--output-last-message",
            job["output_path"],
        ])
        command.append(job["prompt"])
        log_handle = open(job["log_path"], "wb")
        try:
            process = subprocess.Popen(
                command,
                cwd=job["directory"],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()
        job["process"] = process
        job["started_monotonic"] = time.monotonic()
        job["state"]["status"] = "running"
        job["state"]["pid"] = process.pid
        job["state"]["started_at"] = _now_text()
        _atomic_write_json(job["state_path"], job["state"])
        self.active.append(job)
        log_info("已启动Luna批次{}，共{}条，哈希{}".format(
            job["job_id"],
            len(batch_hashes),
            ",".join(value[:10] for value in batch_hashes),
        ))

    def _get_embedder(self) -> ChineseCLIPScriptEmbedder:
        """延迟加载并复用Chinese-CLIP，避免每批重复占用内存。"""
        if self.embedder is None:
            log_info("开始加载本地Chinese-CLIP文本模型")
            self.embedder = ChineseCLIPScriptEmbedder(
                model_name=self.args.embedding_model_name,
                model_cache=self.model_cache,
                torch_threads=self.args.torch_threads,
            )
        return self.embedder

    def _save_records(self, job: dict, records: Sequence[Tuple[str, dict]]) -> None:
        """串行编码并原子保存一个成功批次。"""
        prompt_sha256 = str(job["state"]["prompt_sha256"])
        for content_sha256, analysis in records:
            save_local_script_feature(
                embedder=self._get_embedder(),
                content_sha256=content_sha256,
                analysis=analysis,
                source_root=self.source_root,
                source_feature_version=self.args.source_feature_version,
                output_root=self.output_root,
                script_feature_version=self.args.script_feature_version,
                script_model=self.args.model,
                reasoning_effort=self.args.reasoning_effort,
                input_mode="codex_exec_local_archive",
                prompt_version=DEFAULT_PROMPT_VERSION,
                prompt_sha256=_prompt_contract_sha256(),
                audit_details={
                    "codex_job_id": job["job_id"],
                    "response_usage": read_codex_usage(job["log_path"]),
                    "job_prompt_sha256": prompt_sha256,
                    "contact_sheet_sha256": build_safe_evidence(
                        _read_json(job["manifests"][content_sha256]),
                        self._manifest_and_image(content_sha256)[1],
                    )["contact_sheet"]["sha256"],
                },
            )
            self.completed.add(content_sha256)
            self.retry_counts.pop(content_sha256, None)

    def _job_log_tail(self, path: str, maximum: int = 3000) -> str:
        """提取失败日志尾部，限制状态文件体积。"""
        try:
            with open(path, "rb") as source:
                source.seek(0, os.SEEK_END)
                size = source.tell()
                source.seek(max(size - maximum, 0))
                return source.read().decode("utf-8", errors="replace")
        except OSError:
            return "无法读取Codex失败日志"

    def _mark_failure(self, job: dict, reason: str) -> None:
        """记录失败并让重试批次自动降为单条。"""
        for content_sha256 in job["hashes"]:
            if content_sha256 not in self.completed:
                self.retry_counts[content_sha256] = (
                    self.retry_counts.get(content_sha256, 0) + 1
                )
                retry_count = self.retry_counts[content_sha256]
                delay_seconds = min(15.0 * float(2 ** (retry_count - 1)), 300.0)
                self.retry_ready_at[content_sha256] = time.monotonic() + delay_seconds
        _atomic_write_json(self.retry_path, self.retry_counts)
        job["state"]["status"] = "failed"
        job["state"]["finished_at"] = _now_text()
        job["state"]["error"] = reason[-3000:]
        _atomic_write_json(job["state_path"], job["state"])
        log_error("Luna批次{}失败，稍后自动重试，原因：{}".format(
            job["job_id"], reason[-800:]
        ))

    def _finish_job(self, job: dict) -> None:
        """校验、向量化并登记一个已退出任务。"""
        process = job["process"]
        if process.returncode != 0:
            self._mark_failure(
                job,
                "Codex退出码{}；{}".format(
                    process.returncode,
                    self._job_log_tail(job["log_path"]),
                ),
            )
            return
        try:
            records = parse_job_output(job["output_path"], job["hashes"])
            self._save_records(job, records)
        except Exception as exc:
            self._mark_failure(job, "结果校验或向量化失败：{}".format(exc))
            return
        job["state"]["status"] = "completed"
        job["state"]["response_usage"] = read_codex_usage(job["log_path"])
        job["state"]["finished_at"] = _now_text()
        job["state"]["saved_count"] = len(records)
        _atomic_write_json(job["state_path"], job["state"])
        _atomic_write_json(self.retry_path, self.retry_counts)
        log_info("Luna批次{}完成，新增{}条，当前{}/{}".format(
            job["job_id"], len(records), len(self.completed), len(self.hashes)
        ))

    def _poll_jobs(self) -> None:
        """回收已完成或超时的子进程。"""
        remaining = []
        for job in self.active:
            process = job["process"]
            return_code = process.poll()
            elapsed = time.monotonic() - float(job["started_monotonic"])
            if return_code is None and elapsed > self.args.timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                self._mark_failure(job, "Codex批次超过{}秒已终止".format(
                    self.args.timeout_seconds
                ))
            elif return_code is None:
                remaining.append(job)
            else:
                self._finish_job(job)
        self.active = remaining

    def _available_hashes(self) -> List[str]:
        """返回尚未完成、未运行且没有超过重试上限的哈希。"""
        active_hashes = {
            value for job in self.active for value in job["hashes"]
        }
        current_time = time.monotonic()
        return [
            value for value in self.hashes
            if value not in self.completed
            and value not in active_hashes
            and self.retry_counts.get(value, 0) < self.args.max_retries
            and self.retry_ready_at.get(value, 0.0) <= current_time
        ]

    def _next_batch(self, candidates: Sequence[str]) -> List[str]:
        """首次尝试使用小批次，重试自动拆成单条定位异常。"""
        if not candidates:
            return []
        first = candidates[0]
        if self.retry_counts.get(first, 0) > 0:
            return [first]
        batch = []
        for content_sha256 in candidates:
            if self.retry_counts.get(content_sha256, 0) > 0:
                continue
            batch.append(content_sha256)
            if len(batch) >= self.args.batch_size:
                break
        return batch or [first]

    def _write_status(self, state: str, message: str = "") -> None:
        """发布机器可读的实时进度、速率和预计完成时间。"""
        elapsed_minutes = max((time.time() - self.started_at) / 60.0, 0.0001)
        completed_this_run = max(len(self.completed) - self.initial_completed, 0)
        rate = float(completed_this_run) / elapsed_minutes
        remaining_count = len(self.hashes) - len(self.completed)
        eta_minutes = None if rate <= 0 else float(remaining_count) / rate
        exhausted = [
            value for value in self.hashes
            if value not in self.completed
            and self.retry_counts.get(value, 0) >= self.args.max_retries
        ]
        payload = {
            "state": state,
            "message": message,
            "pid": os.getpid(),
            "updated_at": _now_text(),
            "started_at": datetime.datetime.fromtimestamp(
                self.started_at
            ).isoformat(timespec="seconds"),
            "model": self.args.model,
            "reasoning_effort": self.args.reasoning_effort,
            "workers": self.args.workers,
            "batch_size": self.args.batch_size,
            "total_count": len(self.hashes),
            "initial_completed_count": self.initial_completed,
            "completed_count": len(self.completed),
            "completed_this_run": completed_this_run,
            "remaining_count": remaining_count,
            "active_job_count": len(self.active),
            "active_content_count": sum(len(job["hashes"]) for job in self.active),
            "active_jobs": [
                {
                    "job_id": job["job_id"],
                    "content_sha256": job["hashes"],
                    "pid": job["process"].pid,
                }
                for job in self.active
            ],
            "rate_per_minute": round(rate, 3),
            "eta_minutes": None if eta_minutes is None else round(eta_minutes, 1),
            "retrying_count": sum(
                1 for value in self.retry_counts
                if value not in self.completed
            ),
            "exhausted_count": len(exhausted),
            "exhausted_hashes": exhausted,
            "label_fields_excluded": True,
            "production_server_accessed": False,
        }
        _atomic_write_json(self.status_path, payload)

    def _stop_active_jobs(self) -> None:
        """收到停止信号时终止活跃请求，避免继续消耗额度。"""
        for job in self.active:
            process = job["process"]
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            job["state"]["status"] = "stopped"
            job["state"]["finished_at"] = _now_text()
            _atomic_write_json(job["state_path"], job["state"])
        self.active = []

    def _handle_signal(self, signum: int, frame: object) -> None:
        """把系统信号转换为主循环可处理的停止请求。"""
        del signum, frame
        self.stop_requested = True

    def initialize(self) -> None:
        """核对环境、冻结队列和已有Sidecar。"""
        self._acquire_lock()
        os.makedirs(self.jobs_directory, exist_ok=True)
        if os.path.exists(self.stop_path):
            os.remove(self.stop_path)
        if not os.path.isfile(self.snapshot_path):
            raise RuntimeError("冻结快照不存在：{}".format(self.snapshot_path))
        codex_binary = shutil.which(self.args.codex_binary) or self.args.codex_binary
        if not os.path.isfile(codex_binary):
            raise RuntimeError("Codex命令不存在：{}".format(self.args.codex_binary))
        self.args.codex_binary = codex_binary
        self.hashes = load_snapshot_hashes(self.snapshot_path)
        self.completed = {
            content_sha256 for content_sha256 in self.hashes
            if has_script_feature(
                self.output_root,
                self.args.script_feature_version,
                content_sha256,
            )
        }
        self.initial_completed = len(self.completed)
        self._load_retries()
        _atomic_write_json(self.schema_path, build_output_schema())
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        self._write_status("initialized")
        log_info("自动脚本队列初始化完成，总计{}条，已有{}条，待处理{}条".format(
            len(self.hashes),
            self.initial_completed,
            len(self.hashes) - self.initial_completed,
        ))

    def run(self) -> int:
        """持续补充任务，全部完成、停止或达到失败上限后退出。"""
        self.initialize()
        if self.args.dry_run:
            candidates = self._available_hashes()
            if candidates:
                job = self._prepare_job(self._next_batch(candidates))
                job["state"]["status"] = "dry_run"
                _atomic_write_json(job["state_path"], job["state"])
            self._write_status("dry_run", "安全证据与提示词检查完成")
            return 0
        while True:
            if self.stop_requested or os.path.exists(self.stop_path):
                self._stop_active_jobs()
                self._write_status("stopped", "收到停止请求")
                log_warn("自动脚本任务已停止")
                return 2
            self._poll_jobs()
            if len(self.completed) == len(self.hashes):
                self._write_status("completed", "全部脚本特征已完成")
                log_info("自动脚本任务全部完成，共{}条".format(len(self.hashes)))
                return 0
            candidates = self._available_hashes()
            while len(self.active) < self.args.workers and candidates:
                batch = self._next_batch(candidates)
                try:
                    self._start_job(batch)
                except Exception as exc:
                    placeholder = {
                        "job_id": "prepare_{}".format(int(time.time())),
                        "hashes": batch,
                        "state": {},
                        "state_path": os.path.join(
                            self.runtime_directory, "last_prepare_failure.json"
                        ),
                    }
                    self._mark_failure(
                        placeholder,
                        "批次准备或启动失败：{}".format(exc),
                    )
                candidates = self._available_hashes()
            if not self.active and not candidates:
                can_retry_later = any(
                    value not in self.completed
                    and self.retry_counts.get(value, 0) < self.args.max_retries
                    for value in self.hashes
                )
                if can_retry_later:
                    self._write_status("waiting_retry", "等待失败素材冷却后自动重试")
                    time.sleep(self.args.poll_seconds)
                    continue
                self._write_status("failed", "部分素材达到自动重试上限")
                log_error("自动脚本任务未能全部完成，部分素材达到重试上限")
                return 1
            self._write_status("running")
            time.sleep(self.args.poll_seconds)


def _daemonize(args: argparse.Namespace) -> int:
    """脱离当前终端启动守护进程，并把日志留在任务目录。"""
    repo_root = os.path.realpath(args.repo_root)
    runtime_directory = _normalize_path(repo_root, args.runtime_directory)
    os.makedirs(runtime_directory, exist_ok=True)
    stdout_path = os.path.join(runtime_directory, "supervisor.stdout.log")
    child_arguments = [
        value for value in sys.argv[1:]
        if value != "--daemonize"
    ]
    command = [sys.executable, os.path.realpath(__file__)] + child_arguments
    environment = dict(os.environ)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        BACKEND_ROOT if not existing_python_path
        else BACKEND_ROOT + os.pathsep + existing_python_path
    )
    with open(stdout_path, "ab") as output:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print("自动脚本守护进程已启动，PID：{}".format(process.pid))
    print("状态文件：{}".format(os.path.join(runtime_directory, "status.json")))
    return 0


def _show_status(args: argparse.Namespace) -> int:
    """打印当前状态，供人工和自动巡检复用。"""
    runtime_directory = _normalize_path(args.repo_root, args.runtime_directory)
    status_path = os.path.join(runtime_directory, "status.json")
    if not os.path.isfile(status_path):
        print("尚无自动脚本任务状态")
        return 1
    print(json.dumps(_read_json(status_path), ensure_ascii=False, indent=2))
    return 0


def _request_stop(args: argparse.Namespace) -> int:
    """写入停止标记，让守护进程安全终止活跃请求。"""
    runtime_directory = _normalize_path(args.repo_root, args.runtime_directory)
    os.makedirs(runtime_directory, exist_ok=True)
    _atomic_write_text(os.path.join(runtime_directory, "STOP"), _now_text())
    print("已提交停止请求")
    return 0


def parse_args() -> argparse.Namespace:
    """读取自动脚本任务参数。"""
    repo_root = os.path.realpath(os.path.join(BACKEND_ROOT, ".."))
    parser = argparse.ArgumentParser(description="Codex Luna素材脚本自动任务")
    parser.add_argument("--repo-root", default=repo_root)
    parser.add_argument("--snapshot-path", default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument(
        "--source-root",
        default="material_quality/.runtime/raw_features",
    )
    parser.add_argument("--source-feature-version", default=DEFAULT_SOURCE_VERSION)
    parser.add_argument(
        "--output-root",
        default="material_quality/.runtime/script_features",
    )
    parser.add_argument(
        "--script-feature-version",
        default=DEFAULT_SCRIPT_FEATURE_VERSION,
    )
    parser.add_argument("--runtime-directory", default=DEFAULT_RUNTIME_DIRECTORY)
    parser.add_argument(
        "--model-cache",
        default="material_quality/.runtime/models/chinese_clip",
    )
    parser.add_argument("--embedding-model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=DEFAULT_REASONING_EFFORT,
    )
    parser.add_argument(
        "--codex-binary",
        default=shutil.which("codex") or "codex",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--daemonize", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        parser.error("并发数和批次大小必须大于0")
    if args.max_retries <= 0 or args.timeout_seconds <= 0:
        parser.error("重试次数和超时时间必须大于0")
    return args


def main() -> int:
    """启动、查询或停止自动脚本任务。"""
    set_process_name("material_quality_script_supervisor")
    args = parse_args()
    if args.status:
        return _show_status(args)
    if args.stop:
        return _request_stop(args)
    if args.daemonize:
        return _daemonize(args)
    try:
        return CodexScriptSupervisor(args).run()
    except Exception as exc:
        log_error("自动脚本任务启动失败，原因：{}".format(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

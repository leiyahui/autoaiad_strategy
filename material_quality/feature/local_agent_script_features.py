#!/usr/bin/env python3
"""把Codex子Agent脚本结果转换为本地可训练的Chinese-CLIP向量。"""

import argparse
import datetime
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from plan_logger import log_info, set_process_name
from material_quality.feature.script_agent_features import (
    SCRIPT_ANALYSIS_SCHEMA,
    _validate_script_analysis,
    canonical_script_text,
    save_script_feature,
)
from material_quality.feature.semantic_features import (
    aggregate_frame_embeddings,
    split_text_chunks,
)


DEFAULT_SCRIPT_FEATURE_VERSION = "codex_agent_script_cnclip_v1"
DEFAULT_SCRIPT_MODEL = "gpt-5.6-terra-codex-agent"
DEFAULT_EMBEDDING_MODEL = "cn-clip-rn50-text-aggregate"
DEFAULT_MODEL_NAME = "RN50"
DEFAULT_PROMPT_VERSION = "codex_agent_material_script_v1"


def _file_sha256(path: str) -> str:
    """计算输入清单摘要，保证脚本特征可追溯。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_sha256() -> str:
    """固定记录Agent任务使用的结构约束版本。"""
    payload = {
        "prompt_version": DEFAULT_PROMPT_VERSION,
        "schema": SCRIPT_ANALYSIS_SCHEMA,
        "rules": [
            "只读取内容哈希级档案、关键帧拼图、OCR、ASR和时间轴",
            "不读取GMV、素材标签、投放天数和训练快照",
            "不评价素材质量，不预测投放效果，证据不足时明确说明",
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_agent_row(row: object) -> Tuple[str, dict]:
    """兼容顶层分析字段和analysis包装两种Agent落盘格式。"""
    if not isinstance(row, dict):
        raise ValueError("Agent结果每行必须是对象")
    content_sha256 = str(row.get("content_sha256") or "").lower()
    if len(content_sha256) != 64:
        raise ValueError("Agent结果内容哈希无效")
    if "analysis" in row:
        if set(row) != {"content_sha256", "analysis"}:
            raise ValueError("analysis包装格式包含多余字段")
        analysis = row.get("analysis")
    else:
        analysis = {
            key: value for key, value in row.items()
            if key != "content_sha256"
        }
    _validate_script_analysis(analysis)
    return content_sha256, analysis


def load_agent_results(paths: Sequence[str]) -> List[Tuple[str, dict]]:
    """读取多个JSONL并拒绝重复内容，避免覆盖不同Agent结果。"""
    records = []
    seen_hashes = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                text_value = line.strip()
                if not text_value:
                    continue
                try:
                    row = json.loads(text_value)
                    content_sha256, analysis = _normalize_agent_row(row)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Agent结果读取失败：{}第{}行，原因：{}".format(
                        path,
                        line_number,
                        exc,
                    ))
                if content_sha256 in seen_hashes:
                    raise ValueError("Agent结果内容哈希重复：{}".format(
                        content_sha256,
                    ))
                seen_hashes.add(content_sha256)
                records.append((content_sha256, analysis))
    if not records:
        raise ValueError("Agent结果为空")
    return records


class ChineseCLIPScriptEmbedder:
    """仅加载Chinese-CLIP文本侧，避免重复加载OCR和ASR模型。"""

    def __init__(
        self,
        model_name: str,
        model_cache: str,
        torch_threads: int = 4,
        use_modelscope: bool = True,
    ):
        try:
            import torch
            import cn_clip.clip as clip
            from cn_clip.clip import load_from_name
        except ImportError as exc:
            raise RuntimeError("脚本向量依赖尚未安装：{}".format(exc))
        torch.set_num_threads(max(int(torch_threads), 1))
        os.makedirs(model_cache, exist_ok=True)
        self.torch = torch
        self.clip = clip
        self.model, _ = load_from_name(
            model_name,
            device="cpu",
            download_root=model_cache,
            use_modelscope=use_modelscope,
        )
        self.model.eval()

    def encode(self, text_value: str) -> Tuple[np.ndarray, List[str]]:
        """保留完整结构文本，并以分块均值和标准差形成2048维向量。"""
        chunks = split_text_chunks(text_value)
        if not chunks:
            raise ValueError("脚本规范文本为空")
        inputs = self.clip.tokenize(chunks)
        with self.torch.no_grad():
            chunk_embeddings = self.model.encode_text(inputs)
        values = chunk_embeddings.detach().cpu().numpy().astype(np.float32)
        aggregated = np.asarray(
            aggregate_frame_embeddings(values),
            dtype=np.float32,
        )
        if aggregated.ndim != 1 or not np.isfinite(aggregated).all():
            raise ValueError("脚本聚合向量无效")
        return aggregated, chunks


def save_local_script_feature(
    embedder: ChineseCLIPScriptEmbedder,
    content_sha256: str,
    analysis: dict,
    source_root: str,
    source_feature_version: str,
    output_root: str,
    script_feature_version: str,
    script_model: str = DEFAULT_SCRIPT_MODEL,
    reasoning_effort: str = "medium",
    input_mode: str = "codex_subagent_local_archive",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    prompt_sha256: Optional[str] = None,
    audit_details: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    """使用已加载的文本模型保存单条脚本特征，供持续任务复用模型。"""
    content_manifest_path = os.path.join(
        source_root,
        source_feature_version,
        "content",
        content_sha256[:2],
        content_sha256 + ".json",
    )
    if not os.path.isfile(content_manifest_path):
        raise ValueError("源内容清单不存在：{}".format(content_sha256))
    canonical_text = canonical_script_text(analysis)
    embedding, chunks = embedder.encode(canonical_text)
    audit = {
        "input_mode": input_mode,
        "source_manifest_sha256": _file_sha256(content_manifest_path),
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256 or _prompt_sha256(),
        "label_fields_excluded": True,
    }
    if audit_details:
        audit.update(audit_details)
    result = {
        "schema_version": "material_quality_script_agent_v1",
        "script_model": script_model,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "reasoning_effort": reasoning_effort,
        "content_sha256": content_sha256,
        "source_feature_version": source_feature_version,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "embedding_dim": int(embedding.size),
        "canonical_text": canonical_text,
        "canonical_chunks": chunks,
        "analysis": analysis,
        "audit": audit,
    }
    return save_script_feature(
        output_root,
        script_feature_version,
        result,
        embedding,
    )


def build_local_script_features(
    records: Sequence[Tuple[str, dict]],
    source_root: str,
    source_feature_version: str,
    output_root: str,
    script_feature_version: str,
    model_name: str,
    model_cache: str,
    torch_threads: int,
) -> List[Dict[str, str]]:
    """生成内容哈希级Sidecar，原始结构文本保留在JSON中便于重编码。"""
    embedder = ChineseCLIPScriptEmbedder(
        model_name=model_name,
        model_cache=model_cache,
        torch_threads=torch_threads,
    )
    saved_paths = []
    for content_sha256, analysis in records:
        paths = save_local_script_feature(
            embedder=embedder,
            content_sha256=content_sha256,
            analysis=analysis,
            source_root=source_root,
            source_feature_version=source_feature_version,
            output_root=output_root,
            script_feature_version=script_feature_version,
        )
        saved_paths.append(paths)
    return saved_paths


def parse_args() -> argparse.Namespace:
    """解析本地Agent脚本向量化参数。"""
    parser = argparse.ArgumentParser(description="生成Codex Agent脚本向量")
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-feature-version", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--script-feature-version",
        default=DEFAULT_SCRIPT_FEATURE_VERSION,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """执行本地Agent结果校验和向量化。"""
    set_process_name("material_quality_script_agent")
    args = parse_args()
    records = load_agent_results(args.input_jsonl)
    saved_paths = build_local_script_features(
        records=records,
        source_root=args.source_root,
        source_feature_version=args.source_feature_version,
        output_root=args.output_root,
        script_feature_version=args.script_feature_version,
        model_name=args.model_name,
        model_cache=args.model_cache,
        torch_threads=args.torch_threads,
    )
    log_info("本地Agent脚本向量生成完成，共{}条".format(len(saved_paths)))


if __name__ == "__main__":
    main()

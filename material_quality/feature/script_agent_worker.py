#!/usr/bin/env python3
"""在本地批量生成内容级脚本理解Sidecar，不访问生产数据库。"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from material_quality.feature.script_agent_features import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SCRIPT_FEATURE_VERSION,
    DEFAULT_SCRIPT_MODEL,
    OpenAIJSONClient,
    analyze_content_archive,
    has_script_feature,
    save_script_feature,
)
from plan_logger import log_error, log_info, log_warn, set_process_name


def _read_json(path: str) -> dict:
    """读取本地素材引用。"""
    with open(path, "r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("素材引用必须是JSON对象：{}".format(path))
    return payload


def discover_targets(
    raw_feature_root: str,
    source_feature_version: str,
    label_status: str = "ready",
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """按内容哈希确定性排序，且只从无标签内容目录读取模型输入。"""
    version_root = os.path.join(
        os.path.realpath(raw_feature_root),
        source_feature_version,
    )
    asset_root = os.path.join(version_root, "assets")
    content_root = os.path.join(version_root, "content")
    if not os.path.isdir(asset_root) or not os.path.isdir(content_root):
        raise RuntimeError("本地V4素材引用或内容目录不存在")
    targets_by_hash: Dict[str, Dict[str, object]] = {}
    for filename in os.listdir(asset_root):
        if not filename.endswith(".json"):
            continue
        reference = _read_json(os.path.join(asset_root, filename))
        if reference.get("feature_version") != source_feature_version:
            continue
        if label_status != "all" and reference.get("label_status") != label_status:
            continue
        content_sha256 = str(reference.get("content_sha256") or "").lower()
        if len(content_sha256) != 64:
            continue
        manifest_path = os.path.join(
            content_root,
            content_sha256[:2],
            content_sha256 + ".json",
        )
        if not os.path.isfile(manifest_path):
            continue
        targets_by_hash.setdefault(content_sha256, {
            "content_sha256": content_sha256,
            "content_manifest_path": manifest_path,
        })
    targets = [targets_by_hash[key] for key in sorted(targets_by_hash)]
    if limit is not None and limit >= 0:
        return targets[:limit]
    return targets


def run_worker(args: argparse.Namespace) -> dict:
    """执行可恢复批量任务，已有完整Sidecar直接跳过。"""
    targets = discover_targets(
        args.raw_feature_root,
        args.source_feature_version,
        label_status=args.label_status,
        limit=args.scan_limit,
    )
    pending = [
        target for target in targets
        if not has_script_feature(
            args.output_root,
            args.script_feature_version,
            str(target["content_sha256"]),
        )
    ]
    if args.limit is not None:
        pending = pending[:args.limit]
    summary = {
        "discovered_count": len(targets),
        "pending_count": len(pending),
        "success_count": 0,
        "failure_count": 0,
    }
    log_info("脚本理解任务已发现{}条，待处理{}条".format(
        len(targets),
        len(pending),
    ))
    if args.dry_run or not pending:
        return summary

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError("未设置{}环境变量，无法调用GPT-5.6 Luna".format(
            args.api_key_env
        ))
    thread_local = threading.local()

    def get_client() -> OpenAIJSONClient:
        """每个线程复用自己的轻量客户端配置。"""
        client = getattr(thread_local, "client", None)
        if client is None:
            client = OpenAIJSONClient(
                api_key,
                base_url=args.api_base_url,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            )
            thread_local.client = client
        return client

    def process(target: Dict[str, object]) -> Dict[str, object]:
        """处理一个内容哈希，不读取素材标签引用。"""
        content_sha256 = str(target["content_sha256"])
        result, embedding = analyze_content_archive(
            get_client(),
            str(target["content_manifest_path"]),
            script_model=args.script_model,
            embedding_model=args.embedding_model,
            reasoning_effort=args.reasoning_effort,
            max_frames=args.max_frames,
        )
        save_script_feature(
            args.output_root,
            args.script_feature_version,
            result,
            embedding,
        )
        return {
            "content_sha256": content_sha256,
            "embedding_dim": int(embedding.size),
            "response_usage": result.get("response_usage") or {},
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(process, target): target for target in pending}
        for future in as_completed(future_map):
            target = future_map[future]
            try:
                result = future.result()
                summary["success_count"] += 1
                log_info("脚本特征完成，内容{}，向量{}维，进度{}/{}".format(
                    str(result["content_sha256"])[:12],
                    result["embedding_dim"],
                    summary["success_count"] + summary["failure_count"],
                    len(pending),
                ))
            except Exception as exc:
                summary["failure_count"] += 1
                log_error("脚本特征失败，内容{}，原因：{}".format(
                    str(target["content_sha256"])[:12],
                    exc,
                ))
    return summary


def parse_args() -> argparse.Namespace:
    """读取本地脚本理解任务参数。"""
    parser = argparse.ArgumentParser(description="本地GPT-5.6 Luna脚本理解任务")
    parser.add_argument(
        "--raw-feature-root",
        default="material_quality/.runtime/raw_features",
        help="V4原始特征根目录",
    )
    parser.add_argument(
        "--source-feature-version",
        default="multimodal_cnclip_whisper_replayable_v4",
        help="源V4特征版本",
    )
    parser.add_argument(
        "--output-root",
        default="material_quality/.runtime/script_features",
        help="脚本Sidecar输出根目录",
    )
    parser.add_argument(
        "--script-feature-version",
        default=DEFAULT_SCRIPT_FEATURE_VERSION,
        help="脚本特征版本",
    )
    parser.add_argument("--script-model", default=DEFAULT_SCRIPT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument("--label-status", choices=("ready", "window_open", "all"), default="ready")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scan-limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """启动本地脚本理解任务。"""
    set_process_name("material_quality_script_agent")
    args = parse_args()
    if args.workers <= 0 or args.max_frames <= 0:
        log_warn("并发数和代表帧数量必须大于0")
        return 2
    try:
        summary = run_worker(args)
    except Exception as exc:
        log_error("脚本理解任务启动失败，原因：{}".format(exc))
        return 1
    log_info("脚本理解任务结束，成功{}条，失败{}条，待处理{}条".format(
        summary["success_count"],
        summary["failure_count"],
        summary["pending_count"],
    ))
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

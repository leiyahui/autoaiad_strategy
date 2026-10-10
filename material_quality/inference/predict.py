#!/usr/bin/env python3
"""通过内容哈希对已完成特征抽取的素材进行单条或批量评分。"""

import argparse
import json
from typing import List

from material_quality.inference.features import load_content_features
from material_quality.inference.model import MaterialQualityPredictor
from plan_logger import log_info, set_process_name


def parse_args() -> argparse.Namespace:
    """读取本地素材质量评分参数。"""
    parser = argparse.ArgumentParser(description="对已抽取特征的素材进行质量评分")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--content-sha256", action="append", required=True)
    parser.add_argument("--raw-feature-root", required=True)
    parser.add_argument(
        "--raw-feature-version",
        default="multimodal_cnclip_whisper_replayable_v4",
    )
    parser.add_argument("--script-feature-root", required=True)
    parser.add_argument(
        "--script-feature-version",
        default="codex_agent_script_cnclip_v1",
    )
    parser.add_argument("--output-jsonl")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="仅用于迁移诊断，不建议生产使用",
    )
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> List[dict]:
    """加载一次模型，依次读取内容档案并返回评分。"""
    predictor = MaterialQualityPredictor.load(
        args.model_dir,
        strict_versions=not args.allow_version_mismatch,
    )
    results = []
    for content_sha256 in args.content_sha256:
        base_feature, script_feature, metadata = load_content_features(
            content_sha256,
            args.raw_feature_root,
            args.raw_feature_version,
            args.script_feature_root,
            args.script_feature_version,
        )
        result = predictor.predict_one(
            base_feature,
            script_feature,
            content_sha256=content_sha256,
        )
        result["feature_metadata"] = metadata
        results.append(result)
    return results


def main() -> int:
    """执行本地素材质量评分。"""
    set_process_name("material_quality_predict")
    args = parse_args()
    results = run_prediction(args)
    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as output:
            for result in results:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
        log_info("素材质量评分已写入{}，共{}条".format(
            args.output_jsonl,
            len(results),
        ))
    else:
        for result in results:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

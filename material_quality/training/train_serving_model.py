#!/usr/bin/env python3
"""使用全部成熟样本训练并锁定正式素材质量评分模型。"""

import argparse
import datetime
import hashlib
import json
import os
import platform
from typing import Dict, Tuple

import numpy as np

from material_quality.inference.model import (
    DEFAULT_FUSION_WEIGHTS,
    MODEL_SCHEMA_VERSION,
    MaterialQualityPredictor,
)
from material_quality.training.evaluation import evaluate_classifier
from material_quality.training.script_dataset import load_script_augmented_dataset
from material_quality.training.train_late_fusion_classifier import STRICT_PROFILE
from material_quality.training.train_regularized_classifier import _build_classifier
from material_quality.training.train_script_ablation import _build_matrix_and_layout
from plan_logger import log_info, log_warn, set_process_name


SERVING_TOWERS = {
    "visual": ["visual", "media"],
    "script": ["script"],
}


def _file_sha256(path: str) -> str:
    """计算模型文件摘要并写入不可变清单。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fit_serving_towers(
    dataset: Dict[str, np.ndarray],
    labels: np.ndarray,
) -> Tuple[dict, dict]:
    """在同一批全量样本上拟合画面和脚本两个强正则子模型。"""
    models = {}
    probabilities = {}
    for tower_name, groups in SERVING_TOWERS.items():
        matrix, semantic_blocks, statistic_indices = _build_matrix_and_layout(
            dataset,
            groups,
        )
        classifier = _build_classifier(
            semantic_blocks,
            statistic_indices,
            int(STRICT_PROFILE["pca_components"]),
            float(STRICT_PROFILE["c"]),
        )
        classifier.fit(matrix, labels)
        models[tower_name] = classifier
        probabilities[tower_name] = classifier.predict_proba(matrix)[:, 1]
    return models, probabilities


def _write_report(output_dir: str, manifest: dict) -> None:
    """写入正式模型的用途、输入和限制说明。"""
    quantiles = manifest["training_score_quantiles"]
    lines = [
        "# 素材质量正式评分模型",
        "",
        "- 模型版本：{}".format(manifest["model_version"]),
        "- 成熟样本：{}条，正GMV：{}条，零GMV：{}条".format(
            manifest["sample_count"],
            manifest["positive_count"],
            manifest["zero_count"],
        ),
        "- 子模型：画面50% + Luna脚本50%",
        "- 标签：素材首个统计日起7日窗口内，日均GMV是否大于0",
        "- 输出分数：相对训练样本分布的百分位，不是GMV金额预测",
        "- 训练环境：Python {} / NumPy {} / scikit-learn {}".format(
            manifest["runtime"]["python"],
            manifest["runtime"]["numpy"],
            manifest["runtime"]["scikit_learn"],
        ),
        "",
        "## 训练分数参考",
        "",
        "- P10/P50/P90：{:.6f} / {:.6f} / {:.6f}".format(
            quantiles["p10"],
            quantiles["p50"],
            quantiles["p90"],
        ),
        "- 训练集AUC仅用于序列化自检：{:.4f}".format(
            manifest["training_diagnostic"]["roc_auc"]
        ),
        "",
        "## 使用限制",
        "",
        "- 必须先完成同版本画面特征和Luna脚本特征抽取。",
        "- 分数用于素材初筛和排序，暂不允许直接触发自动投放或停投。",
        "- 新行业、平台接口或特征版本变化后必须重新做时间外评估。",
        "",
    ]
    with open(os.path.join(output_dir, "MODEL_CARD.md"), "w", encoding="utf-8") as output:
        output.write("\n".join(lines))


def run_training(args: argparse.Namespace) -> dict:
    """加载冻结快照、全量拟合、保存模型并执行序列化回放。"""
    import joblib
    import sklearn

    dataset, source_manifest = load_script_augmented_dataset(
        args.base_snapshot_dir,
        args.script_feature_root,
        args.script_feature_version,
    )
    binary_labels = (dataset["labels"] > 0).astype(np.int32)
    if np.unique(binary_labels).size != 2:
        raise RuntimeError("正式训练样本必须同时包含正GMV和零GMV素材")
    models, probabilities = _fit_serving_towers(dataset, binary_labels)
    fused = (
        probabilities["visual"] * DEFAULT_FUSION_WEIGHTS["visual"]
        + probabilities["script"] * DEFAULT_FUSION_WEIGHTS["script"]
    )
    reference_scores = np.sort(fused.astype(np.float64))
    package = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_version": args.model_version,
        "models": models,
        "fusion_weights": dict(DEFAULT_FUSION_WEIGHTS),
        "script_feature_dim": int(dataset["script_features"].shape[1]),
        "reference_scores": reference_scores,
        "base_dataset_sha256": source_manifest["base_dataset_sha256"],
        "script_dataset_sha256": source_manifest["script_dataset_sha256"],
    }
    os.makedirs(args.output_dir, exist_ok=False)
    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(package, model_path, compress=3)
    smoke_count = min(16, int(binary_labels.size))
    np.savez_compressed(
        os.path.join(args.output_dir, "smoke_test.npz"),
        base_features=dataset["features"][:smoke_count],
        script_features=dataset["script_features"][:smoke_count],
        positive_probability=fused[:smoke_count],
    )
    diagnostic = evaluate_classifier(binary_labels, fused)
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_version": args.model_version,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "positive_gmv_per_delivery_day_7d",
        "sample_count": int(binary_labels.size),
        "positive_count": int(binary_labels.sum()),
        "zero_count": int((binary_labels == 0).sum()),
        "base_feature_dim": int(dataset["features"].shape[1]),
        "script_feature_dim": int(dataset["script_features"].shape[1]),
        "raw_feature_version": source_manifest.get("feature_version"),
        "script_feature_version": source_manifest["script_feature_version"],
        "script_models": source_manifest["script_models"],
        "script_embedding_models": source_manifest["script_embedding_models"],
        "fusion_weights": dict(DEFAULT_FUSION_WEIGHTS),
        "regularization_profile": dict(STRICT_PROFILE),
        "base_dataset_sha256": source_manifest["base_dataset_sha256"],
        "script_dataset_sha256": source_manifest["script_dataset_sha256"],
        "model_sha256": _file_sha256(model_path),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "training_diagnostic": diagnostic,
        "training_score_quantiles": {
            "p10": float(np.quantile(fused, 0.10)),
            "p25": float(np.quantile(fused, 0.25)),
            "p50": float(np.quantile(fused, 0.50)),
            "p75": float(np.quantile(fused, 0.75)),
            "p90": float(np.quantile(fused, 0.90)),
        },
        "usage_policy": "只用于素材初筛和排序，不直接触发自动投放",
    }
    with open(
        os.path.join(args.output_dir, "manifest.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    _write_report(args.output_dir, manifest)
    predictor = MaterialQualityPredictor.load(args.output_dir)
    smoke_ok, maximum_error = predictor.verify_smoke_test()
    if not smoke_ok:
        raise RuntimeError("模型序列化回放不一致，最大误差{}".format(maximum_error))
    manifest["smoke_test_maximum_error"] = maximum_error
    with open(
        os.path.join(args.output_dir, "manifest.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    log_info("正式素材质量模型训练完成，目录：{}".format(args.output_dir))
    return manifest


def parse_args() -> argparse.Namespace:
    """读取正式模型训练参数。"""
    parser = argparse.ArgumentParser(description="训练正式素材质量评分模型")
    parser.add_argument("--base-snapshot-dir", required=True)
    parser.add_argument("--script-feature-root", required=True)
    parser.add_argument(
        "--script-feature-version",
        default="codex_agent_script_cnclip_v1",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version", required=True)
    return parser.parse_args()


def main() -> int:
    """执行正式素材质量模型训练。"""
    set_process_name("material_quality_serving_train")
    args = parse_args()
    if not args.model_version.strip():
        log_warn("模型版本不能为空")
        return 2
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

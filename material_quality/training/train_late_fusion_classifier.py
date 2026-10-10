#!/usr/bin/env python3
"""评估画面、脚本和ASR独立子模型的防过拟合晚融合方案。"""

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from material_quality.training.evaluation import evaluate_classifier
from material_quality.training.script_dataset import load_script_augmented_dataset
from material_quality.training.semantic_dataset import (
    ASR_SEMANTIC_DIM,
    FEATURE_GROUP_RANGES,
)
from material_quality.training.train_regularized_classifier import (
    _build_classifier,
    build_rolling_time_folds,
    regularized_selection_score,
)
from material_quality.training.train_script_ablation import (
    _build_matrix_and_layout,
    _evaluate_within_industry_classifier,
)
from material_quality.training.train_semantic_baseline import build_time_masks
from plan_logger import log_info, log_warn, set_process_name


STRICT_PROFILE = {
    "name": "严格",
    "pca_components": 8,
    "c": 0.001,
}


TOWER_CONFIGS = {
    "visual": {
        "label": "画面",
        "groups": ["visual", "media"],
    },
    "script": {
        "label": "脚本",
        "groups": ["script"],
    },
    "asr": {
        "label": "ASR",
        "groups": ["asr", "media"],
    },
}


# 权重在查看外层测试集之前固定，避免根据测试结果反复调参。
FUSION_CONFIGS = [
    {
        "name": "asr",
        "label": "仅ASR",
        "weights": {"asr": 1.0},
    },
    {
        "name": "script",
        "label": "仅脚本",
        "weights": {"script": 1.0},
    },
    {
        "name": "visual",
        "label": "仅画面",
        "weights": {"visual": 1.0},
    },
    {
        "name": "visual_script_50_50",
        "label": "画面50%+脚本50%",
        "weights": {"visual": 0.5, "script": 0.5},
    },
    {
        "name": "script_asr_80_20",
        "label": "脚本80%+ASR20%",
        "weights": {"script": 0.8, "asr": 0.2},
    },
    {
        "name": "visual_asr_90_10",
        "label": "画面90%+ASR10%",
        "weights": {"visual": 0.9, "asr": 0.1},
    },
    {
        "name": "visual_asr_80_20",
        "label": "画面80%+ASR20%",
        "weights": {"visual": 0.8, "asr": 0.2},
    },
    {
        "name": "visual_asr_65_35",
        "label": "画面65%+ASR35%",
        "weights": {"visual": 0.65, "asr": 0.35},
    },
    {
        "name": "visual_script_asr_475_475_05",
        "label": "画面47.5%+脚本47.5%+ASR5%",
        "weights": {"visual": 0.475, "script": 0.475, "asr": 0.05},
    },
    {
        "name": "visual_script_asr_45_45_10",
        "label": "画面45%+脚本45%+ASR10%",
        "weights": {"visual": 0.45, "script": 0.45, "asr": 0.1},
    },
    {
        "name": "visual_script_asr_40_40_20",
        "label": "画面40%+脚本40%+ASR20%",
        "weights": {"visual": 0.4, "script": 0.4, "asr": 0.2},
    },
    {
        "name": "visual_script_asr_35_35_30",
        "label": "画面35%+脚本35%+ASR30%",
        "weights": {"visual": 0.35, "script": 0.35, "asr": 0.3},
    },
    {
        "name": "visual_script_asr_equal",
        "label": "画面+脚本+ASR等权",
        "weights": {
            "visual": 1.0 / 3.0,
            "script": 1.0 / 3.0,
            "asr": 1.0 / 3.0,
        },
    },
]


def _parse_date(value: str) -> datetime.date:
    """解析时间切分日期。"""
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD格式")


def weighted_probability(
    probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """按固定非负权重融合各子模型概率。"""
    if not weights:
        raise ValueError("融合权重不能为空")
    total_weight = float(sum(weights.values()))
    if total_weight <= 0 or any(float(value) < 0 for value in weights.values()):
        raise ValueError("融合权重必须非负且总和大于0")
    combined = None
    expected_shape = None
    for tower_name, weight in weights.items():
        if tower_name not in probabilities:
            raise KeyError("缺少子模型概率：{}".format(tower_name))
        values = np.asarray(probabilities[tower_name], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("子模型概率必须是一维数组")
        if expected_shape is None:
            expected_shape = values.shape
            combined = np.zeros(values.shape, dtype=np.float64)
        elif values.shape != expected_shape:
            raise ValueError("不同子模型概率数量不一致")
        combined += values * (float(weight) / total_weight)
    if combined is None:
        raise ValueError("没有可融合的子模型概率")
    return np.clip(combined, 0.0, 1.0)


def select_fusion_candidate(
    candidates: Sequence[dict],
    score_tolerance: float = 0.005,
) -> dict:
    """得分接近时选择子模型更少、ASR权重更低的候选。"""
    if not candidates:
        raise ValueError("融合候选不能为空")
    best_score = max(float(item["selection_score"]) for item in candidates)
    competitive = [
        item for item in candidates
        if float(item["selection_score"]) >= best_score - score_tolerance
    ]
    return min(
        competitive,
        key=lambda item: (
            len(item["weights"]),
            float(item["weights"].get("asr", 0.0)),
            -float(item["selection_score"]),
        ),
    )


def _build_tower_inputs(dataset: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """构造三个独立子模型的输入矩阵和列布局。"""
    towers = {}
    for name, config in TOWER_CONFIGS.items():
        matrix, semantic_blocks, statistic_indices = _build_matrix_and_layout(
            dataset,
            config["groups"],
        )
        towers[name] = {
            "matrix": matrix,
            "semantic_blocks": semantic_blocks,
            "statistic_indices": statistic_indices,
        }
    return towers


def _fit_tower_probabilities(
    towers: Mapping[str, dict],
    labels: np.ndarray,
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    """在同一时间切分上训练三个子模型并返回训练和预测概率。"""
    train_probabilities = {}
    predict_probabilities = {}
    models = {}
    for name, tower in towers.items():
        model = _build_classifier(
            tower["semantic_blocks"],
            tower["statistic_indices"],
            int(STRICT_PROFILE["pca_components"]),
            float(STRICT_PROFILE["c"]),
        )
        model.fit(tower["matrix"][train_mask], labels[train_mask])
        train_probabilities[name] = model.predict_proba(
            tower["matrix"][train_mask]
        )[:, 1]
        predict_probabilities[name] = model.predict_proba(
            tower["matrix"][predict_mask]
        )[:, 1]
        models[name] = model
    return train_probabilities, predict_probabilities, models


def _evaluate_rolling_fusions(
    towers: Mapping[str, dict],
    labels: np.ndarray,
    folds: Sequence[Dict[str, np.ndarray]],
) -> List[dict]:
    """只使用过去预测未来的滚动窗口评估所有固定融合权重。"""
    accumulators = {
        config["name"]: {
            "train_auc": [],
            "validation_auc": [],
            "validation_pr_auc": [],
            "folds": [],
        }
        for config in FUSION_CONFIGS
    }
    for fold in folds:
        train_probabilities, validation_probabilities, _ = (
            _fit_tower_probabilities(
                towers,
                labels,
                fold["train"],
                fold["validation"],
            )
        )
        for config in FUSION_CONFIGS:
            train_probability = weighted_probability(
                train_probabilities,
                config["weights"],
            )
            validation_probability = weighted_probability(
                validation_probabilities,
                config["weights"],
            )
            train_metrics = evaluate_classifier(
                labels[fold["train"]],
                train_probability,
            )
            validation_metrics = evaluate_classifier(
                labels[fold["validation"]],
                validation_probability,
            )
            accumulator = accumulators[config["name"]]
            accumulator["train_auc"].append(float(train_metrics["roc_auc"]))
            accumulator["validation_auc"].append(
                float(validation_metrics["roc_auc"])
            )
            accumulator["validation_pr_auc"].append(
                float(validation_metrics["pr_auc"])
            )
            accumulator["folds"].append({
                "validation_start": datetime.date.fromordinal(
                    int(fold["validation_start"])
                ).isoformat(),
                "validation_end_exclusive": datetime.date.fromordinal(
                    int(fold["validation_end"])
                ).isoformat(),
                "train_auc": float(train_metrics["roc_auc"]),
                "validation_auc": float(validation_metrics["roc_auc"]),
                "validation_pr_auc": float(validation_metrics["pr_auc"]),
            })
    results = []
    for config in FUSION_CONFIGS:
        values = accumulators[config["name"]]
        train_mean = float(np.mean(values["train_auc"]))
        validation_mean = float(np.mean(values["validation_auc"]))
        validation_std = float(np.std(values["validation_auc"]))
        gap = train_mean - validation_mean
        results.append({
            "name": config["name"],
            "label": config["label"],
            "weights": dict(config["weights"]),
            "rolling_train_auc_mean": train_mean,
            "rolling_validation_auc_mean": validation_mean,
            "rolling_validation_auc_std": validation_std,
            "rolling_validation_pr_auc_mean": float(np.mean(
                values["validation_pr_auc"]
            )),
            "rolling_train_validation_gap": float(gap),
            "selection_score": regularized_selection_score(
                validation_mean,
                validation_std,
                gap,
            ),
            "folds": values["folds"],
        })
    return results


def _write_report(output_dir: str, metrics: dict) -> None:
    """生成ASR晚融合实验报告。"""
    lines = [
        "# ASR晚融合防过拟合实验",
        "",
        "- 外层测试起始日：{}".format(metrics["test_from"]),
        "- 最终拟合/测试：{}/{}".format(
            metrics["split_counts"]["train_validation"],
            metrics["split_counts"]["test"],
        ),
        "- ASR覆盖：{}/{}（{:.2%}）".format(
            metrics["asr_coverage_count"],
            metrics["sample_count"],
            metrics["asr_coverage_rate"],
        ),
        "- 滚动验证自动选择：{}".format(metrics["selected_label"]),
        "- 滚动验证最佳含ASR方案：{}".format(
            metrics["selected_asr_label"]
        ),
        "",
        "| 融合方案 | 滚动验证AUC | 波动 | 训练AUC | 测试AUC | 差距 | 测试PR-AUC | 行业内AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["results"]:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                item["label"],
                item["rolling_validation_auc_mean"],
                item["rolling_validation_auc_std"],
                item["train_metrics"]["roc_auc"],
                item["test_metrics"]["roc_auc"],
                item["train_test_auc_gap"],
                item["test_metrics"]["pr_auc"],
                item["test_within_industry"]["pooled_percentile_auc"],
            )
        )
    lines.extend([
        "",
        "所有子模型均使用PCA8和C=0.001；融合权重只按滚动验证结果选择。",
        "",
    ])
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as output:
        output.write("\n".join(lines))


def run_training(args: argparse.Namespace) -> dict:
    """运行ASR独立子模型晚融合训练和评估。"""
    import joblib

    dataset, manifest = load_script_augmented_dataset(
        args.base_snapshot_dir,
        args.script_feature_root,
        args.script_feature_version,
    )
    labels = (dataset["labels"] > 0).astype(np.int32)
    towers = _build_tower_inputs(dataset)
    outer_masks = build_time_masks(
        dataset["date_ordinals"],
        args.test_from,
        args.validation_days,
    )
    folds = build_rolling_time_folds(
        dataset["date_ordinals"],
        args.test_from,
        args.validation_days,
        args.rolling_folds,
        labels,
    )
    rolling_results = _evaluate_rolling_fusions(towers, labels, folds)
    selected = select_fusion_candidate(rolling_results)
    asr_candidates = [
        item for item in rolling_results
        if float(item["weights"].get("asr", 0.0)) > 0
    ]
    selected_asr = select_fusion_candidate(asr_candidates)

    train_probabilities, test_probabilities, models = _fit_tower_probabilities(
        towers,
        labels,
        outer_masks["train_validation"],
        outer_masks["test"],
    )
    results = []
    test_fusion_probabilities = {}
    for rolling_item in rolling_results:
        train_probability = weighted_probability(
            train_probabilities,
            rolling_item["weights"],
        )
        test_probability = weighted_probability(
            test_probabilities,
            rolling_item["weights"],
        )
        train_metrics = evaluate_classifier(
            labels[outer_masks["train_validation"]],
            train_probability,
        )
        test_metrics = evaluate_classifier(
            labels[outer_masks["test"]],
            test_probability,
        )
        result = dict(rolling_item)
        result.update({
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_test_auc_gap": float(
                train_metrics["roc_auc"] - test_metrics["roc_auc"]
            ),
            "test_within_industry": _evaluate_within_industry_classifier(
                dataset["labels"][outer_masks["test"]],
                test_probability,
                dataset["primary_industries"][outer_masks["test"]],
            ),
        })
        results.append(result)
        test_fusion_probabilities[result["name"]] = test_probability

    asr_start = FEATURE_GROUP_RANGES["asr"][0]
    asr_available = np.any(
        np.abs(dataset["features"][:, asr_start:asr_start + ASR_SEMANTIC_DIM]) > 0,
        axis=1,
    )
    metrics = {
        "base_dataset_sha256": manifest["base_dataset_sha256"],
        "script_dataset_sha256": manifest["script_dataset_sha256"],
        "test_from": args.test_from.isoformat(),
        "validation_days": int(args.validation_days),
        "rolling_fold_count": len(folds),
        "sample_count": int(labels.size),
        "asr_coverage_count": int(asr_available.sum()),
        "asr_coverage_rate": float(asr_available.mean()),
        "split_counts": {
            key: int(value.sum()) for key, value in outer_masks.items()
        },
        "profile": dict(STRICT_PROFILE),
        "selected_name": selected["name"],
        "selected_label": selected["label"],
        "selected_asr_name": selected_asr["name"],
        "selected_asr_label": selected_asr["label"],
        "results": results,
    }
    os.makedirs(args.output_dir, exist_ok=False)
    joblib.dump({
        "profile": dict(STRICT_PROFILE),
        "tower_configs": TOWER_CONFIGS,
        "models": models,
        "selected_fusion": {
            "name": selected["name"],
            "weights": selected["weights"],
        },
        "selected_asr_fusion": {
            "name": selected_asr["name"],
            "weights": selected_asr["weights"],
        },
        "base_dataset_sha256": manifest["base_dataset_sha256"],
        "script_dataset_sha256": manifest["script_dataset_sha256"],
    }, os.path.join(args.output_dir, "models.joblib"), compress=3)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    test_indices = np.flatnonzero(outer_masks["test"])
    with open(
        os.path.join(args.output_dir, "test_predictions.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output:
        writer = csv.writer(output)
        fusion_names = [item["name"] for item in rolling_results]
        writer.writerow([
            "material_id",
            "first_stat_date",
            "actual_positive_gmv",
        ] + ["{}_probability".format(name) for name in fusion_names])
        for position, index in enumerate(test_indices):
            writer.writerow([
                str(dataset["material_ids"][index]),
                datetime.date.fromordinal(
                    int(dataset["date_ordinals"][index])
                ).isoformat(),
                int(labels[index]),
            ] + [
                float(test_fusion_probabilities[name][position])
                for name in fusion_names
            ])
    _write_report(args.output_dir, metrics)
    log_info("ASR晚融合实验完成，目录：{}".format(args.output_dir))
    return metrics


def parse_args() -> argparse.Namespace:
    """读取训练参数。"""
    parser = argparse.ArgumentParser(description="训练ASR独立子模型晚融合分类器")
    parser.add_argument("--base-snapshot-dir", required=True)
    parser.add_argument("--script-feature-root", required=True)
    parser.add_argument(
        "--script-feature-version",
        default="codex_agent_script_cnclip_v1",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--test-from",
        type=_parse_date,
        default=datetime.date(2026, 8, 1),
    )
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--rolling-folds", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    """启动ASR晚融合实验。"""
    set_process_name("material_quality_asr_fusion")
    args = parse_args()
    if args.validation_days <= 0 or args.rolling_folds < 2:
        log_warn("验证天数必须大于0，滚动窗口必须至少2个")
        return 2
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

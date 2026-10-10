#!/usr/bin/env python3
"""使用滚动时间验证训练防过拟合的素材跑量分类器。"""

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from material_quality.training.evaluation import evaluate_classifier
from material_quality.training.script_dataset import load_script_augmented_dataset
from material_quality.training.train_script_ablation import (
    _build_matrix_and_layout,
    _evaluate_within_industry_classifier,
)
from material_quality.training.train_semantic_baseline import build_time_masks
from plan_logger import log_info, log_warn, set_process_name


FEATURE_CONFIGS = [
    {"name": "media", "label": "仅媒体属性", "groups": ["media"]},
    {"name": "visual", "label": "画面", "groups": ["visual", "media"]},
    {
        "name": "visual_asr",
        "label": "画面+ASR",
        "groups": ["visual", "asr", "media"],
    },
    {"name": "script", "label": "仅脚本", "groups": ["script"]},
    {
        "name": "visual_script",
        "label": "画面+脚本",
        "groups": ["visual", "media", "script"],
    },
    {
        "name": "visual_asr_script",
        "label": "画面+ASR+脚本",
        "groups": ["visual", "asr", "media", "script"],
    },
]


REGULARIZATION_PROFILES = [
    {"name": "严格", "pca_components": 8, "c": 0.001},
    {"name": "均衡", "pca_components": 16, "c": 0.003},
    {"name": "宽松", "pca_components": 32, "c": 0.003},
]


def _parse_date(value: str) -> datetime.date:
    """解析时间切分日期。"""
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD格式")


def build_rolling_time_folds(
    date_ordinals: np.ndarray,
    test_from: datetime.date,
    validation_days: int,
    fold_count: int,
    binary_labels: np.ndarray,
    minimum_train_count: int = 300,
    minimum_validation_count: int = 50,
) -> List[Dict[str, np.ndarray]]:
    """从外层测试日前向前构造多个只使用过去预测未来的验证窗口。"""
    folds = []
    test_start = test_from.toordinal()
    for offset in range(fold_count):
        validation_end = test_start - offset * validation_days
        validation_start = validation_end - validation_days
        train_mask = date_ordinals < validation_start
        validation_mask = (
            (date_ordinals >= validation_start)
            & (date_ordinals < validation_end)
        )
        if int(train_mask.sum()) < minimum_train_count:
            continue
        if int(validation_mask.sum()) < minimum_validation_count:
            continue
        if np.unique(binary_labels[train_mask]).size < 2:
            continue
        if np.unique(binary_labels[validation_mask]).size < 2:
            continue
        folds.append({
            "train": train_mask,
            "validation": validation_mask,
            "validation_start": validation_start,
            "validation_end": validation_end,
        })
    if len(folds) < 2:
        raise RuntimeError("可用滚动时间验证窗口少于2个")
    return list(reversed(folds))


def regularized_selection_score(
    validation_auc_mean: float,
    validation_auc_std: float,
    train_validation_gap: float,
) -> float:
    """同时惩罚时间波动和训练验证差距，避免只追求单窗口高分。"""
    return float(
        validation_auc_mean
        - 0.25 * validation_auc_std
        - 0.15 * max(train_validation_gap, 0.0)
    )


def select_conservative_candidate(
    candidates: Sequence[dict],
    score_tolerance: float = 0.01,
) -> dict:
    """得分接近时优先选择维度更低、正则更强的候选。"""
    if not candidates:
        raise ValueError("候选配置不能为空")
    best_score = max(float(item["selection_score"]) for item in candidates)
    competitive = [
        item for item in candidates
        if float(item["selection_score"]) >= best_score - score_tolerance
    ]
    return min(
        competitive,
        key=lambda item: (
            int(item.get("complexity_rank", item["profile"]["pca_components"])),
            float(item["profile"]["c"]),
            -float(item["selection_score"]),
        ),
    )


def _build_classifier(
    semantic_blocks: Sequence[Tuple[str, int, int]],
    statistic_indices: Sequence[int],
    pca_components: int,
    c_value: float,
):
    """构造低维、强正则的逻辑回归分类器。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    transformers = []
    for block_name, start, end in semantic_blocks:
        component_count = min(int(pca_components), int(end - start))
        transformers.append((
            "{}语义降维".format(block_name),
            Pipeline([
                ("标准化", StandardScaler()),
                ("主成分", PCA(
                    n_components=component_count,
                    whiten=True,
                    svd_solver="randomized",
                    random_state=20260821,
                )),
            ]),
            list(range(start, end)),
        ))
    if statistic_indices:
        transformers.append((
            "统计特征",
            StandardScaler(),
            list(statistic_indices),
        ))
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    classifier = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=2000,
        random_state=20260821,
    )
    return Pipeline([
        ("分模态降维", preprocessor),
        ("模型", classifier),
    ])


def _evaluate_profile(
    matrix: np.ndarray,
    semantic_blocks: Sequence[Tuple[str, int, int]],
    statistic_indices: Sequence[int],
    binary_labels: np.ndarray,
    folds: Sequence[Dict[str, np.ndarray]],
    profile: dict,
) -> dict:
    """在所有滚动窗口上评估一个复杂度配置。"""
    train_auc_values = []
    validation_auc_values = []
    validation_pr_values = []
    fold_metrics = []
    for fold in folds:
        classifier = _build_classifier(
            semantic_blocks,
            statistic_indices,
            int(profile["pca_components"]),
            float(profile["c"]),
        )
        classifier.fit(matrix[fold["train"]], binary_labels[fold["train"]])
        train_probability = classifier.predict_proba(matrix[fold["train"]])[:, 1]
        validation_probability = classifier.predict_proba(
            matrix[fold["validation"]]
        )[:, 1]
        train_metrics = evaluate_classifier(
            binary_labels[fold["train"]],
            train_probability,
        )
        validation_metrics = evaluate_classifier(
            binary_labels[fold["validation"]],
            validation_probability,
        )
        train_auc_values.append(float(train_metrics["roc_auc"]))
        validation_auc_values.append(float(validation_metrics["roc_auc"]))
        validation_pr_values.append(float(validation_metrics["pr_auc"]))
        fold_metrics.append({
            "train_count": int(fold["train"].sum()),
            "validation_count": int(fold["validation"].sum()),
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
    train_auc_mean = float(np.mean(train_auc_values))
    validation_auc_mean = float(np.mean(validation_auc_values))
    validation_auc_std = float(np.std(validation_auc_values))
    gap = train_auc_mean - validation_auc_mean
    return {
        "profile": dict(profile),
        "rolling_train_auc_mean": train_auc_mean,
        "rolling_validation_auc_mean": validation_auc_mean,
        "rolling_validation_auc_std": validation_auc_std,
        "rolling_validation_pr_auc_mean": float(np.mean(validation_pr_values)),
        "rolling_train_validation_gap": float(gap),
        "selection_score": regularized_selection_score(
            validation_auc_mean,
            validation_auc_std,
            gap,
        ),
        "folds": fold_metrics,
    }


def _select_profile(
    matrix: np.ndarray,
    semantic_blocks: Sequence[Tuple[str, int, int]],
    statistic_indices: Sequence[int],
    binary_labels: np.ndarray,
    folds: Sequence[Dict[str, np.ndarray]],
) -> Tuple[dict, List[dict]]:
    """使用滚动验证选择兼顾稳定性和泛化差距的复杂度。"""
    candidates = []
    for profile in REGULARIZATION_PROFILES:
        candidates.append(_evaluate_profile(
            matrix,
            semantic_blocks,
            statistic_indices,
            binary_labels,
            folds,
            profile,
        ))
    best = select_conservative_candidate(candidates)
    return best, candidates


def _random_cv_diagnostic(
    matrix: np.ndarray,
    labels: np.ndarray,
    semantic_blocks: Sequence[Tuple[str, int, int]],
    statistic_indices: Sequence[int],
    profile: dict,
    fold_count: int,
) -> dict:
    """随机分层交叉验证仅用于和时间测试对照，不参与最终选型。"""
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=20260821,
    )
    auc_values = []
    for train_indices, validation_indices in splitter.split(matrix, labels):
        classifier = _build_classifier(
            semantic_blocks,
            statistic_indices,
            int(profile["pca_components"]),
            float(profile["c"]),
        )
        classifier.fit(matrix[train_indices], labels[train_indices])
        probability = classifier.predict_proba(matrix[validation_indices])[:, 1]
        auc_values.append(float(evaluate_classifier(
            labels[validation_indices],
            probability,
        )["roc_auc"]))
    return {
        "fold_count": int(fold_count),
        "auc_mean": float(np.mean(auc_values)),
        "auc_std": float(np.std(auc_values)),
        "auc_values": auc_values,
        "warning": "随机切分可能包含相似商品或模板，仅用于诊断时间漂移",
    }


def _write_report(output_dir: str, metrics: dict) -> None:
    """生成包含训练测试差距的中文报告。"""
    lines = [
        "# 防过拟合素材跑量分类器",
        "",
        "- 外层测试起始日：{}".format(metrics["test_from"]),
        "- 最终拟合/测试：{}/{}".format(
            metrics["split_counts"]["train_validation"],
            metrics["split_counts"]["test"],
        ),
        "- 滚动验证窗口：{}个".format(metrics["rolling_fold_count"]),
        "- 自动选择：{}".format(metrics["selected_label"]),
        "",
        "| 特征 | 配置 | 滚动验证AUC | 训练AUC | 测试AUC | 差距 | 行业内AUC | 测试PR-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["results"]:
        profile = item["selected_profile"]
        lines.append(
            "| {} | PCA{} / C={} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                item["label"],
                profile["pca_components"],
                profile["c"],
                item["rolling_validation_auc_mean"],
                item["train_metrics"]["roc_auc"],
                item["test_metrics"]["roc_auc"],
                item["train_test_auc_gap"],
                item["test_within_industry"]["pooled_percentile_auc"],
                item["test_metrics"]["pr_auc"],
            )
        )
    random_cv = metrics["selected_random_cv_diagnostic"]
    lines.extend([
        "",
        "## 随机分层诊断",
        "",
        "- 随机{}折AUC：{:.4f} ± {:.4f}".format(
            random_cv["fold_count"],
            random_cv["auc_mean"],
            random_cv["auc_std"],
        ),
        "- 该指标不参与选型，也不能替代时间外测试。",
        "",
    ])
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as output:
        output.write("\n".join(lines))


def run_training(args: argparse.Namespace) -> dict:
    """训练并评估防过拟合分类器。"""
    import joblib

    dataset, manifest = load_script_augmented_dataset(
        args.base_snapshot_dir,
        args.script_feature_root,
        args.script_feature_version,
    )
    labels = (dataset["labels"] > 0).astype(np.int32)
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
    os.makedirs(args.output_dir, exist_ok=False)
    results = []
    bundles = {}
    matrices = {}
    layouts = {}
    for config in FEATURE_CONFIGS:
        log_info("开始滚动验证防过拟合配置：{}".format(config["label"]))
        matrix, semantic_blocks, statistic_indices = _build_matrix_and_layout(
            dataset,
            config["groups"],
        )
        matrices[config["name"]] = matrix
        layouts[config["name"]] = (semantic_blocks, statistic_indices)
        selected, candidates = _select_profile(
            matrix,
            semantic_blocks,
            statistic_indices,
            labels,
            folds,
        )
        profile = selected["profile"]
        classifier = _build_classifier(
            semantic_blocks,
            statistic_indices,
            int(profile["pca_components"]),
            float(profile["c"]),
        )
        classifier.fit(
            matrix[outer_masks["train_validation"]],
            labels[outer_masks["train_validation"]],
        )
        train_probability = classifier.predict_proba(
            matrix[outer_masks["train_validation"]]
        )[:, 1]
        test_probability = classifier.predict_proba(
            matrix[outer_masks["test"]]
        )[:, 1]
        train_metrics = evaluate_classifier(
            labels[outer_masks["train_validation"]],
            train_probability,
        )
        test_metrics = evaluate_classifier(
            labels[outer_masks["test"]],
            test_probability,
        )
        test_within_industry = _evaluate_within_industry_classifier(
            dataset["labels"][outer_masks["test"]],
            test_probability,
            dataset["primary_industries"][outer_masks["test"]],
        )
        results.append({
            "name": config["name"],
            "label": config["label"],
            "groups": config["groups"],
            "selected_profile": profile,
            "selection_score": selected["selection_score"],
            "rolling_train_auc_mean": selected["rolling_train_auc_mean"],
            "rolling_validation_auc_mean": selected["rolling_validation_auc_mean"],
            "rolling_validation_auc_std": selected["rolling_validation_auc_std"],
            "rolling_train_validation_gap": selected["rolling_train_validation_gap"],
            "profile_candidates": candidates,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_test_auc_gap": float(
                train_metrics["roc_auc"] - test_metrics["roc_auc"]
            ),
            "test_within_industry": test_within_industry,
            "test_probability": test_probability,
        })
        bundles[config["name"]] = {
            "config": config,
            "profile": profile,
            "classifier": classifier,
            "base_dataset_sha256": manifest["base_dataset_sha256"],
            "script_dataset_sha256": manifest["script_dataset_sha256"],
        }
    feature_candidates = []
    for item in results:
        candidate = dict(item)
        candidate["profile"] = item["selected_profile"]
        candidate["complexity_rank"] = (
            len(item["groups"]) * 100
            + int(item["selected_profile"]["pca_components"])
        )
        feature_candidates.append(candidate)
    selected_result = select_conservative_candidate(feature_candidates)
    selected_name = selected_result["name"]
    selected_matrix = matrices[selected_name][outer_masks["train_validation"]]
    selected_labels = labels[outer_masks["train_validation"]]
    semantic_blocks, statistic_indices = layouts[selected_name]
    random_cv = _random_cv_diagnostic(
        selected_matrix,
        selected_labels,
        semantic_blocks,
        statistic_indices,
        selected_result["selected_profile"],
        args.random_folds,
    )
    serializable_results = []
    for item in results:
        serializable_item = dict(item)
        serializable_item.pop("test_probability")
        serializable_results.append(serializable_item)
    metrics = {
        "base_dataset_sha256": manifest["base_dataset_sha256"],
        "script_dataset_sha256": manifest["script_dataset_sha256"],
        "test_from": args.test_from.isoformat(),
        "validation_days": int(args.validation_days),
        "rolling_fold_count": len(folds),
        "split_counts": {
            key: int(value.sum()) for key, value in outer_masks.items()
        },
        "selected_name": selected_name,
        "selected_label": selected_result["label"],
        "selected_random_cv_diagnostic": random_cv,
        "results": serializable_results,
    }
    joblib.dump(bundles, os.path.join(args.output_dir, "models.joblib"), compress=3)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    test_indices = np.flatnonzero(outer_masks["test"])
    selected_probability = selected_result["test_probability"]
    with open(
        os.path.join(args.output_dir, "test_predictions.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output:
        writer = csv.writer(output)
        writer.writerow([
            "material_id",
            "first_stat_date",
            "actual_positive_gmv",
            "positive_probability",
        ])
        for position, index in enumerate(test_indices):
            writer.writerow([
                str(dataset["material_ids"][index]),
                datetime.date.fromordinal(
                    int(dataset["date_ordinals"][index])
                ).isoformat(),
                int(labels[index]),
                float(selected_probability[position]),
            ])
    _write_report(args.output_dir, metrics)
    log_info("防过拟合分类器训练完成，目录：{}".format(args.output_dir))
    return metrics


def parse_args() -> argparse.Namespace:
    """读取训练参数。"""
    parser = argparse.ArgumentParser(description="训练防过拟合素材跑量分类器")
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
    parser.add_argument("--random-folds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    """启动防过拟合分类器训练。"""
    set_process_name("material_quality_regularized")
    args = parse_args()
    if args.validation_days <= 0 or args.rolling_folds < 2:
        log_warn("验证天数必须大于0，滚动窗口必须至少2个")
        return 2
    if args.random_folds < 2:
        log_warn("随机诊断折数必须至少2折")
        return 2
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

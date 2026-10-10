#!/usr/bin/env python3
"""比较现有多模态特征与GPT-5.6 Luna脚本向量的增量效果。"""

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from material_quality.training.evaluation import (
    evaluate_classifier,
    evaluate_regression,
    evaluate_top_fraction,
    restore_gmv_from_log,
    safe_spearman,
)
from material_quality.training.script_dataset import load_script_augmented_dataset
from material_quality.training.semantic_dataset import (
    ASR_SEMANTIC_DIM,
    FEATURE_GROUP_RANGES,
    OCR_SEMANTIC_DIM,
    VISUAL_SEMANTIC_DIM,
)
from material_quality.training.train_semantic_baseline import build_time_masks
from plan_logger import log_info, log_warn, set_process_name


FEATURE_CONFIGS = [
    {"name": "visual", "label": "画面基线", "groups": ["visual", "media"]},
    {"name": "visual_script", "label": "画面+脚本", "groups": ["visual", "media", "script"]},
    {"name": "combined", "label": "现有全模态", "groups": ["visual", "ocr", "asr", "media"]},
    {
        "name": "combined_script",
        "label": "现有全模态+脚本",
        "groups": ["visual", "ocr", "asr", "media", "script"],
    },
    {"name": "script", "label": "仅脚本", "groups": ["script"]},
]


def _parse_date(value: str) -> datetime.date:
    """解析时间切分日期。"""
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD格式")


def _build_matrix_and_layout(
    dataset: Dict[str, np.ndarray],
    groups: Sequence[str],
) -> Tuple[np.ndarray, List[Tuple[str, int, int]], List[int]]:
    """按固定顺序组装特征，并记录各语义块和低阶统计列。"""
    base = dataset["features"]
    script = dataset["script_features"]
    semantic_dims = {
        "visual": VISUAL_SEMANTIC_DIM,
        "ocr": OCR_SEMANTIC_DIM,
        "asr": ASR_SEMANTIC_DIM,
    }
    parts = []
    semantic_blocks: List[Tuple[str, int, int]] = []
    statistic_indices: List[int] = []
    offset = 0
    for group_name in groups:
        if group_name == "script":
            group_values = script
            parts.append(group_values)
            semantic_blocks.append(("script", offset, offset + group_values.shape[1]))
            offset += group_values.shape[1]
            continue
        start, end = FEATURE_GROUP_RANGES[group_name]
        group_values = base[:, start:end]
        semantic_dim = semantic_dims.get(group_name, 0)
        if semantic_dim > 0:
            semantic_values = group_values[:, :semantic_dim]
            parts.append(semantic_values)
            semantic_blocks.append((
                group_name,
                offset,
                offset + semantic_values.shape[1],
            ))
            offset += semantic_values.shape[1]
            statistic_values = group_values[:, semantic_dim:]
            if statistic_values.shape[1] > 0:
                parts.append(statistic_values)
                statistic_indices.extend(range(offset, offset + statistic_values.shape[1]))
                offset += statistic_values.shape[1]
        else:
            parts.append(group_values)
            statistic_indices.extend(range(offset, offset + group_values.shape[1]))
            offset += group_values.shape[1]
    matrix = np.hstack(parts).astype(np.float32)
    return matrix, semantic_blocks, statistic_indices


def _build_model(
    semantic_blocks: Sequence[Tuple[str, int, int]],
    statistic_indices: Sequence[int],
    model_type: str,
    regularization: float,
    pca_components: int,
):
    """每个语义模态单独PCA，验证和测试只执行transform。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    transformers = []
    for block_name, start, end in semantic_blocks:
        transformers.append((
            "{}语义降维".format(block_name),
            Pipeline([
                ("标准化", StandardScaler()),
                ("主成分", PCA(
                    n_components=int(pca_components),
                    whiten=True,
                    svd_solver="randomized",
                    random_state=20260820,
                )),
            ]),
            list(range(start, end)),
        ))
    if statistic_indices:
        transformers.append(("统计特征", StandardScaler(), list(statistic_indices)))
    preprocessor = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    if model_type == "classifier":
        model = LogisticRegression(
            C=float(regularization),
            solver="liblinear",
            max_iter=2000,
            random_state=20260820,
        )
    elif model_type == "regressor":
        model = Ridge(alpha=float(regularization), solver="lsqr")
    else:
        raise ValueError("未知模型类型：{}".format(model_type))
    return Pipeline([("分模态降维", preprocessor), ("模型", model)])


def _fit_config(
    dataset: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    config: dict,
    pca_components: int,
) -> dict:
    """仅使用验证集选择正则强度，再在训练加验证集重拟合。"""
    matrix, semantic_blocks, statistic_indices = _build_matrix_and_layout(
        dataset,
        config["groups"],
    )
    labels = dataset["labels"]
    binary = (labels > 0).astype(np.int32)
    classifier_candidates = []
    for c_value in (0.01, 0.1):
        model = _build_model(
            semantic_blocks,
            statistic_indices,
            "classifier",
            c_value,
            pca_components,
        )
        model.fit(matrix[masks["train"]], binary[masks["train"]])
        probability = model.predict_proba(matrix[masks["validation"]])[:, 1]
        metrics = evaluate_classifier(binary[masks["validation"]], probability)
        classifier_candidates.append({"c": c_value, "metrics": metrics})
    best_classifier = max(
        classifier_candidates,
        key=lambda item: (item["metrics"]["pr_auc"], item["metrics"]["roc_auc"]),
    )

    positive_train = masks["train"] & (labels > 0)
    positive_validation = masks["validation"] & (labels > 0)
    regressor_candidates = []
    for alpha in (100.0, 1000.0):
        model = _build_model(
            semantic_blocks,
            statistic_indices,
            "regressor",
            alpha,
            pca_components,
        )
        model.fit(matrix[positive_train], np.log1p(labels[positive_train]))
        log_upper = float(np.quantile(np.log1p(labels[positive_train]), 0.995))
        prediction = restore_gmv_from_log(
            model.predict(matrix[positive_validation]),
            log_upper,
        )
        metrics = evaluate_regression(labels[positive_validation], prediction)
        regressor_candidates.append({"alpha": alpha, "metrics": metrics})
    best_regressor = min(
        regressor_candidates,
        key=lambda item: (item["metrics"]["log_mae"], -item["metrics"]["spearman"]),
    )

    train_validation = masks["train_validation"]
    positive_train_validation = train_validation & (labels > 0)
    classifier = _build_model(
        semantic_blocks,
        statistic_indices,
        "classifier",
        float(best_classifier["c"]),
        pca_components,
    )
    classifier.fit(matrix[train_validation], binary[train_validation])
    regressor = _build_model(
        semantic_blocks,
        statistic_indices,
        "regressor",
        float(best_regressor["alpha"]),
        pca_components,
    )
    regressor.fit(
        matrix[positive_train_validation],
        np.log1p(labels[positive_train_validation]),
    )
    log_upper = float(np.quantile(
        np.log1p(labels[positive_train_validation]),
        0.995,
    ))
    probability = classifier.predict_proba(matrix[masks["test"]])[:, 1]
    conditional = restore_gmv_from_log(
        regressor.predict(matrix[masks["test"]]),
        log_upper,
    )
    expected = probability * conditional
    actual = labels[masks["test"]]
    return {
        "config": config,
        "classifier_c": float(best_classifier["c"]),
        "regressor_alpha": float(best_regressor["alpha"]),
        "classifier": classifier,
        "regressor": regressor,
        "conditional_log_upper": log_upper,
        "probability": probability,
        "conditional": conditional,
        "expected": expected,
        "metrics": {
            "classifier": evaluate_classifier((actual > 0).astype(np.int32), probability),
            "regression": evaluate_regression(actual, expected),
            "positive_conditional_regression": evaluate_regression(
                actual[actual > 0],
                conditional[actual > 0],
            ),
            "probability_top20": evaluate_top_fraction(actual, probability, 0.2),
            "top20": evaluate_top_fraction(actual, expected, 0.2),
        },
    }


def _evaluate_within_industry_classifier(
    actual: np.ndarray,
    probability: np.ndarray,
    industries: np.ndarray,
) -> dict:
    """用行业内百分位和分行业AUC隔离行业基础跑量差异。"""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    binary = (actual > 0).astype(np.int32)
    normalized_probability = np.zeros(probability.shape, dtype=np.float64)
    industry_auc = []
    industry_weights = []
    for industry in sorted(set(industries.tolist())):
        mask = industries == industry
        count = int(mask.sum())
        normalized_probability[mask] = (
            rankdata(probability[mask], method="average") / float(count + 1)
        )
        if count < 10 or np.unique(binary[mask]).size < 2:
            continue
        industry_auc.append(float(roc_auc_score(binary[mask], probability[mask])))
        industry_weights.append(count)
    result = {
        "pooled_percentile_auc": float(roc_auc_score(
            binary,
            normalized_probability,
        )),
        "eligible_industry_count": len(industry_auc),
    }
    if industry_auc:
        result.update({
            "macro_auc": float(np.mean(industry_auc)),
            "weighted_auc": float(np.average(
                industry_auc,
                weights=industry_weights,
            )),
        })
    return result


def _evaluate_within_industry_top_fraction(
    actual: np.ndarray,
    probability: np.ndarray,
    industries: np.ndarray,
    fraction: float,
) -> dict:
    """每个行业分别选头部素材，再汇总真实跑量率和GMV。"""
    selected_parts = []
    for industry in sorted(set(industries.tolist())):
        industry_indices = np.flatnonzero(industries == industry)
        selected_count = max(1, int(np.ceil(industry_indices.size * fraction)))
        order = np.argsort(probability[industry_indices], kind="mergesort")
        selected_parts.append(industry_indices[order[-selected_count:]])
    selected = np.concatenate(selected_parts)
    positive_rate = float((actual > 0).mean())
    mean_gmv = float(actual.mean())
    return {
        "fraction": float(fraction),
        "selected_count": int(selected.size),
        "positive_rate": float((actual[selected] > 0).mean()),
        "positive_rate_lift": (
            float((actual[selected] > 0).mean()) / positive_rate
            if positive_rate > 0 else 0.0
        ),
        "mean_gmv": float(actual[selected].mean()),
        "mean_gmv_lift": (
            float(actual[selected].mean()) / mean_gmv
            if mean_gmv > 0 else 0.0
        ),
        "gmv_capture": (
            float(actual[selected].sum()) / float(actual.sum())
            if float(actual.sum()) > 0 else 0.0
        ),
    }


def _paired_bootstrap(
    actual: np.ndarray,
    baseline: dict,
    augmented: dict,
    iterations: int,
    seed: int,
) -> dict:
    """对同一批测试素材计算脚本特征增量的配对置信区间。"""
    rng = np.random.RandomState(seed)
    values = {
        "roc_auc_delta": [],
        "pr_auc_delta": [],
        "probability_top20_positive_rate_lift_delta": [],
        "spearman_delta": [],
        "top20_mean_gmv_lift_delta": [],
    }
    for _ in range(iterations):
        indices = rng.randint(0, actual.size, size=actual.size)
        sampled_actual = actual[indices]
        if np.unique(sampled_actual > 0).size < 2:
            continue
        binary = (sampled_actual > 0).astype(np.int32)
        baseline_classifier = evaluate_classifier(
            binary,
            baseline["probability"][indices],
        )
        augmented_classifier = evaluate_classifier(
            binary,
            augmented["probability"][indices],
        )
        baseline_probability_top20 = evaluate_top_fraction(
            sampled_actual,
            baseline["probability"][indices],
            0.2,
        )
        augmented_probability_top20 = evaluate_top_fraction(
            sampled_actual,
            augmented["probability"][indices],
            0.2,
        )
        baseline_top20 = evaluate_top_fraction(
            sampled_actual,
            baseline["expected"][indices],
            0.2,
        )
        augmented_top20 = evaluate_top_fraction(
            sampled_actual,
            augmented["expected"][indices],
            0.2,
        )
        values["roc_auc_delta"].append(
            augmented_classifier["roc_auc"] - baseline_classifier["roc_auc"]
        )
        values["pr_auc_delta"].append(
            augmented_classifier["pr_auc"] - baseline_classifier["pr_auc"]
        )
        values["probability_top20_positive_rate_lift_delta"].append(
            augmented_probability_top20["positive_rate_lift"]
            - baseline_probability_top20["positive_rate_lift"]
        )
        values["spearman_delta"].append(
            safe_spearman(sampled_actual, augmented["expected"][indices])
            - safe_spearman(sampled_actual, baseline["expected"][indices])
        )
        values["top20_mean_gmv_lift_delta"].append(
            augmented_top20["mean_gmv_lift"] - baseline_top20["mean_gmv_lift"]
        )
    return {
        key: [
            float(np.quantile(metric_values, 0.025)),
            float(np.quantile(metric_values, 0.975)),
        ]
        for key, metric_values in values.items()
        if metric_values
    }


def _write_report(output_dir: str, metrics: dict) -> None:
    """生成脚本特征增量中文报告。"""
    lines = [
        "# GPT-5.6 Luna脚本特征增量评估",
        "",
        "- 样本：{}条".format(metrics["sample_count"]),
        "- 训练/验证/测试：{}/{}/{}".format(
            metrics["split_counts"]["train"],
            metrics["split_counts"]["validation"],
            metrics["split_counts"]["test"],
        ),
        "- 脚本向量：{}维".format(metrics["script_embedding_dim"]),
        "",
        "| 特征 | ROC-AUC | 行业内AUC | PR-AUC | 概率Top20正GMV率提升 | 行业内Top20正GMV率提升 | 秩相关 | GMV Top20提升 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["results"]:
        item_metrics = item["metrics"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.3f}倍 | {:.3f}倍 | {:.4f} | {:.3f}倍 |".format(
            item["label"],
            item_metrics["classifier"]["roc_auc"],
            item_metrics["within_industry_classifier"]["pooled_percentile_auc"],
            item_metrics["classifier"]["pr_auc"],
            item_metrics["probability_top20"]["positive_rate_lift"],
            item_metrics["within_industry_probability_top20"]["positive_rate_lift"],
            item_metrics["regression"]["spearman"],
            item_metrics["top20"]["mean_gmv_lift"],
        ))
    lines.extend(["", "## 配对增量95%区间", ""])
    for comparison_name, intervals in metrics["paired_bootstrap_95"].items():
        lines.append("### {}".format(comparison_name))
        for metric_name, interval in intervals.items():
            lines.append("- {}：[{:.4f}, {:.4f}]".format(
                metric_name,
                interval[0],
                interval[1],
            ))
        lines.append("")
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def run_training(args: argparse.Namespace) -> dict:
    """运行严格同样本、同时间切分的脚本特征消融。"""
    import joblib

    dataset, manifest = load_script_augmented_dataset(
        args.base_snapshot_dir,
        args.script_feature_root,
        args.script_feature_version,
    )
    masks = build_time_masks(dataset["date_ordinals"], args.test_from, args.validation_days)
    os.makedirs(args.output_dir, exist_ok=False)
    fitted_results = []
    for config in FEATURE_CONFIGS:
        log_info("开始训练脚本消融配置：{}".format(config["label"]))
        fitted_results.append(_fit_config(
            dataset,
            masks,
            config,
            args.pca_components,
        ))
    by_name = {item["config"]["name"]: item for item in fitted_results}
    actual = dataset["labels"][masks["test"]]
    test_industries = dataset["primary_industries"][masks["test"]]
    for item in fitted_results:
        item["metrics"]["within_industry_classifier"] = (
            _evaluate_within_industry_classifier(
                actual,
                item["probability"],
                test_industries,
            )
        )
        item["metrics"]["within_industry_probability_top20"] = (
            _evaluate_within_industry_top_fraction(
                actual,
                item["probability"],
                test_industries,
                0.2,
            )
        )
    paired = {
        "画面增加脚本": _paired_bootstrap(
            actual,
            by_name["visual"],
            by_name["visual_script"],
            args.bootstrap_iterations,
            20260820,
        ),
        "全模态增加脚本": _paired_bootstrap(
            actual,
            by_name["combined"],
            by_name["combined_script"],
            args.bootstrap_iterations,
            20260821,
        ),
    }
    metrics = {
        "base_dataset_sha256": manifest["base_dataset_sha256"],
        "script_dataset_sha256": manifest["script_dataset_sha256"],
        "script_feature_version": manifest["script_feature_version"],
        "script_models": manifest["script_models"],
        "script_embedding_models": manifest["script_embedding_models"],
        "sample_count": int(dataset["labels"].size),
        "script_embedding_dim": int(dataset["script_features"].shape[1]),
        "test_from": args.test_from.isoformat(),
        "split_counts": {key: int(value.sum()) for key, value in masks.items()},
        "results": [{
            "name": item["config"]["name"],
            "label": item["config"]["label"],
            "groups": item["config"]["groups"],
            "classifier_c": item["classifier_c"],
            "regressor_alpha": item["regressor_alpha"],
            "metrics": item["metrics"],
        } for item in fitted_results],
        "paired_bootstrap_95": paired,
    }
    bundles = {
        item["config"]["name"]: {
            "config": item["config"],
            "classifier": item["classifier"],
            "regressor": item["regressor"],
            "conditional_log_upper": item["conditional_log_upper"],
            "base_dataset_sha256": manifest["base_dataset_sha256"],
            "script_dataset_sha256": manifest["script_dataset_sha256"],
        }
        for item in fitted_results
    }
    joblib.dump(bundles, os.path.join(args.output_dir, "models.joblib"), compress=3)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    test_indices = np.flatnonzero(masks["test"])
    with open(
        os.path.join(args.output_dir, "test_predictions.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output:
        writer = csv.writer(output)
        header = ["material_id", "first_stat_date", "actual_gmv_per_delivery_day"]
        for item in fitted_results:
            header.extend([
                "{}_positive_probability".format(item["config"]["name"]),
                "{}_predicted_gmv".format(item["config"]["name"]),
            ])
        writer.writerow(header)
        for position, index in enumerate(test_indices):
            row = [
                str(dataset["material_ids"][index]),
                datetime.date.fromordinal(int(dataset["date_ordinals"][index])).isoformat(),
                float(dataset["labels"][index]),
            ]
            for item in fitted_results:
                row.extend([
                    float(item["probability"][position]),
                    float(item["expected"][position]),
                ])
            writer.writerow(row)
    _write_report(args.output_dir, metrics)
    log_info("脚本特征增量评估完成，目录：{}".format(args.output_dir))
    return metrics


def parse_args() -> argparse.Namespace:
    """读取脚本特征消融参数。"""
    parser = argparse.ArgumentParser(description="评估GPT-5.6 Luna脚本特征增量")
    parser.add_argument("--base-snapshot-dir", required=True)
    parser.add_argument("--script-feature-root", required=True)
    parser.add_argument("--script-feature-version", default="gpt56_luna_script_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-from", type=_parse_date, default=datetime.date(2026, 8, 1))
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    """启动脚本特征消融训练。"""
    set_process_name("material_quality_script_training")
    args = parse_args()
    if args.validation_days <= 0 or args.pca_components <= 0:
        log_warn("验证天数和PCA维度必须大于0")
        return 2
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

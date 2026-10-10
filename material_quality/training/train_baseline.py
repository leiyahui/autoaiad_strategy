#!/usr/bin/env python3
"""训练并评估素材质量的两阶段基线模型。"""

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from material_quality.training.dataset import (
    FEATURE_NAMES,
    MEDIA_FEATURE_NAMES,
    VISUAL_FEATURE_DIM,
    build_population_weights,
    load_feature_dataset,
    save_dataset_snapshot,
)
from material_quality.training.evaluation import (
    bootstrap_intervals,
    bootstrap_ranking_intervals,
    evaluate_classifier,
    evaluate_regression,
    evaluate_top_fraction,
    restore_gmv_from_log,
)
from plan_logger import log_info, log_warn, set_process_name


DEFAULT_COHORT = "food_v1"
DEFAULT_FEATURE_VERSION = "media_visual_stats_v1"
DEFAULT_OUTPUT_ROOT = "material_quality/.runtime/artifacts"


CLASSIFIER_CANDIDATES = [
    {"name": "媒体属性_逻辑回归", "feature_set": "media", "algorithm": "logistic", "c": 1.0},
    {"name": "视觉向量_逻辑回归", "feature_set": "visual", "algorithm": "logistic", "c": 1.0},
    {"name": "组合特征_逻辑回归", "feature_set": "combined", "algorithm": "logistic", "c": 0.1},
    {"name": "组合特征_逻辑回归强", "feature_set": "combined", "algorithm": "logistic", "c": 1.0},
    {"name": "组合特征_梯度树7", "feature_set": "combined", "algorithm": "hist", "max_leaf_nodes": 7},
    {"name": "组合特征_梯度树15", "feature_set": "combined", "algorithm": "hist", "max_leaf_nodes": 15},
]

REGRESSOR_CANDIDATES = [
    {"name": "媒体属性_岭回归", "feature_set": "media", "algorithm": "ridge", "alpha": 10.0},
    {"name": "视觉向量_岭回归", "feature_set": "visual", "algorithm": "ridge", "alpha": 10.0},
    {"name": "组合特征_岭回归10", "feature_set": "combined", "algorithm": "ridge", "alpha": 10.0},
    {"name": "组合特征_岭回归100", "feature_set": "combined", "algorithm": "ridge", "alpha": 100.0},
    {"name": "组合特征_梯度树7", "feature_set": "combined", "algorithm": "hist", "max_leaf_nodes": 7},
    {"name": "组合特征_梯度树15", "feature_set": "combined", "algorithm": "hist", "max_leaf_nodes": 15},
]


def _ordinal(value: str) -> int:
    """把配置日期转换为便于向量比较的序数。"""
    return datetime.datetime.strptime(value, "%Y-%m-%d").date().toordinal()


def _feature_indices(feature_set: str) -> np.ndarray:
    """返回候选模型允许使用的特征列。"""
    if feature_set == "visual":
        return np.arange(VISUAL_FEATURE_DIM)
    if feature_set == "media":
        return np.arange(VISUAL_FEATURE_DIM, len(FEATURE_NAMES))
    if feature_set == "combined":
        return np.arange(len(FEATURE_NAMES))
    raise ValueError("未知特征集合：{}".format(feature_set))


def _build_classifier(candidate: dict):
    """按候选配置创建分类流水线。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if candidate["algorithm"] == "logistic":
        model = LogisticRegression(
            C=float(candidate["c"]),
            max_iter=2000,
            random_state=20260818,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=int(candidate["max_leaf_nodes"]),
            min_samples_leaf=30,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=20260818,
        )
    return Pipeline([("标准化", StandardScaler()), ("模型", model)])


def _build_regressor(candidate: dict):
    """按候选配置创建正GMV对数回归流水线。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if candidate["algorithm"] == "ridge":
        model = Ridge(alpha=float(candidate["alpha"]))
    else:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=int(candidate["max_leaf_nodes"]),
            min_samples_leaf=30,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=20260818,
        )
    return Pipeline([("标准化", StandardScaler()), ("模型", model)])


def _build_direct_challenger():
    """创建直接拟合非负GMV的泊松梯度树对照模型。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.035,
        max_iter=300,
        max_leaf_nodes=7,
        min_samples_leaf=30,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=20260818,
    )
    return Pipeline([("标准化", StandardScaler()), ("模型", model)])


def _fit_pipeline(model, features, labels, sample_weight):
    """统一向流水线末端模型传递样本权重。"""
    model.fit(features, labels, **{"模型__sample_weight": sample_weight})
    return model


def _folds() -> List[dict]:
    """使用连续月份进行滚动回测，避免随机切分造成未来泄漏。"""
    return [
        {
            "name": "验证六月",
            "train_before": _ordinal("2026-06-01"),
            "valid_from": _ordinal("2026-06-01"),
            "valid_before": _ordinal("2026-07-01"),
        },
        {
            "name": "验证七月",
            "train_before": _ordinal("2026-07-01"),
            "valid_from": _ordinal("2026-07-01"),
            "valid_before": _ordinal("2026-08-01"),
        },
    ]


def _select_classifier(dataset, population_counts) -> Tuple[dict, List[dict]]:
    """以滚动验证的加权PR-AUC选择跑量分类器。"""
    all_results = []
    features = dataset["features"]
    labels = (dataset["labels"] > 0).astype(np.int32)
    dates = dataset["date_ordinals"]
    for candidate in CLASSIFIER_CANDIDATES:
        indices = _feature_indices(candidate["feature_set"])
        fold_results = []
        for fold in _folds():
            train_mask = dates < fold["train_before"]
            valid_mask = (dates >= fold["valid_from"]) & (dates < fold["valid_before"])
            train_weight = build_population_weights(
                dates[train_mask], dataset["labels"][train_mask], population_counts
            )
            valid_weight = build_population_weights(
                dates[valid_mask], dataset["labels"][valid_mask], population_counts
            )
            model = _fit_pipeline(
                _build_classifier(candidate),
                features[train_mask][:, indices],
                labels[train_mask],
                train_weight,
            )
            probability = model.predict_proba(features[valid_mask][:, indices])[:, 1]
            metrics = evaluate_classifier(labels[valid_mask], probability, valid_weight)
            fold_results.append({"fold": fold["name"], "metrics": metrics})
        mean_pr_auc = float(np.mean([
            item["metrics"]["pr_auc"] for item in fold_results
        ]))
        all_results.append({
            "candidate": candidate,
            "folds": fold_results,
            "mean_pr_auc": mean_pr_auc,
        })
    best = max(all_results, key=lambda item: item["mean_pr_auc"])
    return best["candidate"], all_results


def _select_regressor(dataset, population_counts) -> Tuple[dict, List[dict]]:
    """以正GMV样本的加权对数MAE选择条件回归器。"""
    all_results = []
    features = dataset["features"]
    labels = dataset["labels"]
    dates = dataset["date_ordinals"]
    for candidate in REGRESSOR_CANDIDATES:
        indices = _feature_indices(candidate["feature_set"])
        fold_results = []
        for fold in _folds():
            train_mask = (dates < fold["train_before"]) & (labels > 0)
            valid_mask = (
                (dates >= fold["valid_from"])
                & (dates < fold["valid_before"])
                & (labels > 0)
            )
            train_weight = build_population_weights(
                dates[train_mask], labels[train_mask], population_counts
            )
            valid_weight = build_population_weights(
                dates[valid_mask], labels[valid_mask], population_counts
            )
            model = _fit_pipeline(
                _build_regressor(candidate),
                features[train_mask][:, indices],
                np.log1p(labels[train_mask]),
                train_weight,
            )
            predicted_log = model.predict(features[valid_mask][:, indices])
            log_upper = float(np.quantile(np.log1p(labels[train_mask]), 0.995))
            predicted = restore_gmv_from_log(predicted_log, log_upper)
            metrics = evaluate_regression(labels[valid_mask], predicted, valid_weight)
            fold_results.append({"fold": fold["name"], "metrics": metrics})
        mean_log_mae = float(np.mean([
            item["metrics"]["log_mae"] for item in fold_results
        ]))
        all_results.append({
            "candidate": candidate,
            "folds": fold_results,
            "mean_log_mae": mean_log_mae,
        })
    best = min(all_results, key=lambda item: item["mean_log_mae"])
    return best["candidate"], all_results


def _evaluate_direct_backtests(dataset, population_counts) -> List[dict]:
    """检查直接泊松模型在六月和七月是否保持同方向表现。"""
    results = []
    features = dataset["features"]
    labels = dataset["labels"]
    dates = dataset["date_ordinals"]
    for fold in _folds():
        train_mask = dates < fold["train_before"]
        valid_mask = (dates >= fold["valid_from"]) & (dates < fold["valid_before"])
        train_weight = build_population_weights(
            dates[train_mask], labels[train_mask], population_counts
        )
        valid_weight = build_population_weights(
            dates[valid_mask], labels[valid_mask], population_counts
        )
        model = _fit_pipeline(
            _build_direct_challenger(),
            features[train_mask],
            labels[train_mask],
            train_weight,
        )
        predicted = np.maximum(model.predict(features[valid_mask]), 0.0)
        results.append({
            "fold": fold["name"],
            "regression": evaluate_regression(
                labels[valid_mask], predicted, valid_weight
            ),
            "top20": evaluate_top_fraction(
                labels[valid_mask], predicted, 0.2, valid_weight
            ),
        })
    return results


def _quality_scores(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """把组合预估值转换为训练参考分布中的百分位分数。"""
    sorted_reference = np.sort(reference)
    ranks = np.searchsorted(sorted_reference, values, side="right")
    return ranks.astype(np.float64) / float(sorted_reference.size) * 100.0


def _write_predictions(
    output_dir: str,
    dataset,
    test_mask,
    probability,
    conditional_gmv,
    predicted_gmv,
    quality_score,
    direct_predicted_gmv,
) -> None:
    """保存锁定测试集逐素材预测，便于人工复核。"""
    path = os.path.join(output_dir, "test_predictions.csv")
    test_indices = np.flatnonzero(test_mask)
    with open(path, "w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "sample_id",
            "material_id",
            "first_stat_date",
            "actual_gmv_7d",
            "positive_probability",
            "conditional_positive_gmv",
            "predicted_gmv_7d",
            "quality_score",
            "direct_poisson_gmv",
        ])
        for position, index in enumerate(test_indices):
            writer.writerow([
                int(dataset["sample_ids"][index]),
                str(dataset["material_ids"][index]),
                datetime.date.fromordinal(int(dataset["date_ordinals"][index])).isoformat(),
                float(dataset["labels"][index]),
                float(probability[position]),
                float(conditional_gmv[position]),
                float(predicted_gmv[position]),
                float(quality_score[position]),
                float(direct_predicted_gmv[position]),
            ])


def _write_report(output_dir: str, metrics: dict) -> None:
    """生成便于直接阅读的中文评估摘要。"""
    test = metrics["test"]
    top20 = test["two_stage"]["top20"]
    direct = test["direct_poisson"]
    lines = [
        "# 素材质量基线模型评估",
        "",
        "- 数据快照：`{}`".format(metrics["dataset_sha256"]),
        "- 样本数：{}".format(metrics["sample_count"]),
        "- 分类模型：{}".format(metrics["selected_classifier"]["name"]),
        "- 正GMV回归模型：{}".format(metrics["selected_regressor"]["name"]),
        "- 八月锁定测试样本：{}".format(test["sample_count"]),
        "",
        "## 八月锁定测试结果",
        "",
        "- PR-AUC：{:.4f}".format(test["classifier"]["pr_auc"]),
        "- ROC-AUC：{:.4f}".format(test["classifier"]["roc_auc"]),
        "- Brier：{:.4f}".format(test["classifier"]["brier"]),
        "- GMV秩相关：{:.4f}".format(test["two_stage"]["regression"]["spearman"]),
        "- GMV对数MAE：{:.4f}".format(test["two_stage"]["regression"]["log_mae"]),
        "- Top20%正GMV率提升：{:.3f}倍".format(top20["positive_rate_lift"]),
        "- Top20%平均GMV提升：{:.3f}倍".format(top20["mean_gmv_lift"]),
        "- Top20%捕获GMV：{:.2%}".format(top20["gmv_capture"]),
        "",
        "## 直接泊松对照模型",
        "",
        "- GMV秩相关：{:.4f}".format(direct["regression"]["spearman"]),
        "- GMV对数MAE：{:.4f}".format(direct["regression"]["log_mae"]),
        "- Top20%正GMV率提升：{:.3f}倍".format(direct["top20"]["positive_rate_lift"]),
        "- Top20%平均GMV提升：{:.3f}倍".format(direct["top20"]["mean_gmv_lift"]),
        "- Top20%捕获GMV：{:.2%}".format(direct["top20"]["gmv_capture"]),
        "",
        "完整候选模型、对照模型和置信区间请查看 `metrics.json`。",
    ]
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def run_training(args: argparse.Namespace) -> dict:
    """冻结数据、选择模型并完成一次锁定测试。"""
    import joblib

    dataset, population_counts, manifest = load_feature_dataset(
        args.cohort,
        args.feature_version,
    )
    output_dir = os.path.join(args.output_root, args.run_id)
    save_dataset_snapshot(output_dir, dataset, population_counts, manifest)
    log_info("训练快照已冻结，样本{}条，特征{}维".format(
        manifest["sample_count"],
        manifest["feature_dim"],
    ))

    best_classifier, classifier_results = _select_classifier(dataset, population_counts)
    best_regressor, regressor_results = _select_regressor(dataset, population_counts)
    direct_backtests = _evaluate_direct_backtests(dataset, population_counts)
    log_info("滚动验证选择分类模型：{}".format(best_classifier["name"]))
    log_info("滚动验证选择正GMV回归模型：{}".format(best_regressor["name"]))

    dates = dataset["date_ordinals"]
    labels = dataset["labels"]
    features = dataset["features"]
    final_train_mask = dates < _ordinal("2026-08-01")
    test_mask = dates >= _ordinal("2026-08-01")
    if int(test_mask.sum()) == 0:
        raise RuntimeError("八月锁定测试集为空")

    classifier_indices = _feature_indices(best_classifier["feature_set"])
    classifier_weight = build_population_weights(
        dates[final_train_mask], labels[final_train_mask], population_counts
    )
    classifier = _fit_pipeline(
        _build_classifier(best_classifier),
        features[final_train_mask][:, classifier_indices],
        (labels[final_train_mask] > 0).astype(np.int32),
        classifier_weight,
    )

    positive_train_mask = final_train_mask & (labels > 0)
    regressor_indices = _feature_indices(best_regressor["feature_set"])
    regressor_weight = build_population_weights(
        dates[positive_train_mask], labels[positive_train_mask], population_counts
    )
    regressor = _fit_pipeline(
        _build_regressor(best_regressor),
        features[positive_train_mask][:, regressor_indices],
        np.log1p(labels[positive_train_mask]),
        regressor_weight,
    )

    train_probability = classifier.predict_proba(
        features[final_train_mask][:, classifier_indices]
    )[:, 1]
    train_log_gmv = regressor.predict(features[final_train_mask][:, regressor_indices])
    log_upper = float(np.quantile(np.log1p(labels[positive_train_mask]), 0.995))
    train_conditional = restore_gmv_from_log(train_log_gmv, log_upper)
    train_expected = train_probability * train_conditional

    test_probability = classifier.predict_proba(features[test_mask][:, classifier_indices])[:, 1]
    test_log_gmv = regressor.predict(features[test_mask][:, regressor_indices])
    test_conditional = restore_gmv_from_log(test_log_gmv, log_upper)
    test_expected = test_probability * test_conditional
    test_quality_score = _quality_scores(train_expected, test_expected)
    test_weight = build_population_weights(
        dates[test_mask], labels[test_mask], population_counts
    )

    classifier_metrics = evaluate_classifier(
        (labels[test_mask] > 0).astype(np.int32),
        test_probability,
        test_weight,
    )
    two_stage_regression = evaluate_regression(
        labels[test_mask],
        test_expected,
        test_weight,
    )
    top10 = evaluate_top_fraction(labels[test_mask], test_expected, 0.1, test_weight)
    top20 = evaluate_top_fraction(labels[test_mask], test_expected, 0.2, test_weight)

    direct_weight = build_population_weights(
        dates[final_train_mask], labels[final_train_mask], population_counts
    )
    direct_model = _fit_pipeline(
        _build_direct_challenger(),
        features[final_train_mask],
        labels[final_train_mask],
        direct_weight,
    )
    direct_prediction = np.maximum(direct_model.predict(features[test_mask]), 0.0)

    baseline_probability = float(np.average(
        (labels[final_train_mask] > 0).astype(np.float64),
        weights=classifier_weight,
    ))
    baseline_positive_log = float(np.average(
        np.log1p(labels[positive_train_mask]),
        weights=regressor_weight,
    ))
    baseline_prediction = np.full(
        int(test_mask.sum()),
        baseline_probability * np.expm1(baseline_positive_log),
    )

    metrics = {
        "run_id": args.run_id,
        "dataset_sha256": manifest["dataset_sha256"],
        "sample_count": manifest["sample_count"],
        "selected_classifier": best_classifier,
        "selected_regressor": best_regressor,
        "classifier_candidates": classifier_results,
        "regressor_candidates": regressor_results,
        "direct_poisson_backtests": direct_backtests,
        "test": {
            "sample_count": int(test_mask.sum()),
            "classifier": classifier_metrics,
            "two_stage": {
                "regression": two_stage_regression,
                "top10": top10,
                "top20": top20,
            },
            "direct_poisson": {
                "regression": evaluate_regression(labels[test_mask], direct_prediction, test_weight),
                "top20": evaluate_top_fraction(labels[test_mask], direct_prediction, 0.2, test_weight),
                "bootstrap_95": bootstrap_ranking_intervals(
                    labels[test_mask],
                    direct_prediction,
                    test_weight,
                    iterations=args.bootstrap_iterations,
                ),
            },
            "constant_baseline": {
                "regression": evaluate_regression(labels[test_mask], baseline_prediction, test_weight),
            },
            "bootstrap_95": bootstrap_intervals(
                labels[test_mask],
                test_probability,
                test_expected,
                test_weight,
                iterations=args.bootstrap_iterations,
            ),
        },
    }

    bundle = {
        "run_id": args.run_id,
        "cohort": args.cohort,
        "feature_version": args.feature_version,
        "dataset_sha256": manifest["dataset_sha256"],
        "feature_names": FEATURE_NAMES,
        "classifier": classifier,
        "classifier_feature_indices": classifier_indices,
        "regressor": regressor,
        "regressor_feature_indices": regressor_indices,
        "conditional_log_upper": log_upper,
        "reference_expected_gmv": np.sort(train_expected),
        "direct_poisson": direct_model,
    }
    joblib.dump(bundle, os.path.join(output_dir, "models.joblib"), compress=3)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "feature_names.json"), "w", encoding="utf-8") as output:
        json.dump(FEATURE_NAMES, output, ensure_ascii=False, indent=2)
    _write_predictions(
        output_dir,
        dataset,
        test_mask,
        test_probability,
        test_conditional,
        test_expected,
        test_quality_score,
        direct_prediction,
    )
    _write_report(output_dir, metrics)
    log_info("模型训练与锁定测试完成，产物目录：{}".format(output_dir))
    return metrics


def parse_args() -> argparse.Namespace:
    """读取离线训练参数。"""
    parser = argparse.ArgumentParser(description="训练素材质量两阶段基线模型")
    parser.add_argument("--cohort", default=DEFAULT_COHORT, help="样本范围版本")
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION, help="特征版本")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="模型产物根目录")
    parser.add_argument(
        "--run-id",
        default="mq_food_v1_media_stats_v1_{}".format(datetime.date.today().strftime("%Y%m%d")),
        help="不可重复的模型运行版本",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=500,
        help="锁定测试集置信区间抽样次数",
    )
    return parser.parse_args()


def main() -> int:
    """离线训练命令入口。"""
    set_process_name("material_quality_training")
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        log_warn("置信区间抽样次数必须大于0")
        return 2
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

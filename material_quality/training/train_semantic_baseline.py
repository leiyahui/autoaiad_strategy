#!/usr/bin/env python3
"""训练并评估多模态V2素材质量快速基线。"""

import argparse
import csv
import datetime
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from material_quality.training.evaluation import (
    bootstrap_intervals,
    bootstrap_ranking_intervals,
    evaluate_classifier,
    evaluate_regression,
    evaluate_top_fraction,
    restore_gmv_from_log,
)
from material_quality.training.semantic_dataset import (
    ASR_SEMANTIC_DIM,
    FEATURE_GROUP_RANGES,
    FEATURE_NAMES,
    OCR_SEMANTIC_DIM,
    VISUAL_SEMANTIC_DIM,
    load_local_semantic_dataset,
    load_semantic_dataset,
    load_semantic_snapshot,
    save_semantic_snapshot,
)
from plan_logger import log_info, log_warn, set_process_name


DEFAULT_COHORT = "all_industries_v1"
DEFAULT_FEATURE_VERSION = "multimodal_cnclip_whisper_small_v2"

FEATURE_SET_GROUPS = {
    "media": ["media"],
    "visual": ["visual", "media"],
    "visual_ocr": ["visual", "ocr", "media"],
    "visual_asr": ["visual", "asr", "media"],
    "combined": ["visual", "ocr", "asr", "media"],
}

FEATURE_SET_LABELS = {
    "media": "仅媒体属性",
    "visual": "画面语义",
    "visual_ocr": "画面+OCR",
    "visual_asr": "画面+ASR",
    "combined": "全模态",
}

CLASSIFIER_CANDIDATES = [
    {"name": "仅媒体属性_逻辑回归", "feature_set": "media", "c": 0.1},
    {"name": "画面语义_逻辑回归", "feature_set": "visual", "c": 0.01},
    {"name": "画面OCR_逻辑回归", "feature_set": "visual_ocr", "c": 0.01},
    {"name": "画面ASR_逻辑回归", "feature_set": "visual_asr", "c": 0.01},
    {"name": "全模态_逻辑回归强正则", "feature_set": "combined", "c": 0.001},
    {"name": "全模态_逻辑回归", "feature_set": "combined", "c": 0.01},
    {
        "name": "画面语义_分组PCA64_逻辑回归",
        "feature_set": "visual",
        "c": 0.1,
        "reduction": "group_pca",
        "pca_components": 64,
    },
    {
        "name": "画面OCR_分组PCA64_逻辑回归",
        "feature_set": "visual_ocr",
        "c": 0.1,
        "reduction": "group_pca",
        "pca_components": 64,
    },
    {
        "name": "画面ASR_分组PCA64_逻辑回归",
        "feature_set": "visual_asr",
        "c": 0.1,
        "reduction": "group_pca",
        "pca_components": 64,
    },
    {
        "name": "全模态_分组PCA64_逻辑回归",
        "feature_set": "combined",
        "c": 0.1,
        "reduction": "group_pca",
        "pca_components": 64,
    },
]

REGRESSOR_CANDIDATES = [
    {"name": "仅媒体属性_岭回归", "feature_set": "media", "alpha": 10.0},
    {"name": "画面语义_岭回归", "feature_set": "visual", "alpha": 100.0},
    {"name": "画面OCR_岭回归", "feature_set": "visual_ocr", "alpha": 100.0},
    {"name": "画面ASR_岭回归", "feature_set": "visual_asr", "alpha": 100.0},
    {"name": "全模态_岭回归", "feature_set": "combined", "alpha": 100.0},
    {"name": "全模态_岭回归强正则", "feature_set": "combined", "alpha": 1000.0},
    {
        "name": "画面语义_分组PCA64_岭回归",
        "feature_set": "visual",
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
    },
    {
        "name": "画面OCR_分组PCA64_岭回归",
        "feature_set": "visual_ocr",
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
    },
    {
        "name": "画面ASR_分组PCA64_岭回归",
        "feature_set": "visual_asr",
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
    },
    {
        "name": "全模态_分组PCA64_岭回归",
        "feature_set": "combined",
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
    },
]

ABLATION_CONFIGS = [
    {"feature_set": "media", "c": 0.1, "alpha": 10.0},
    {
        "feature_set": "visual",
        "c": 0.1,
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
        "label": "画面语义（PCA64）",
    },
    {
        "feature_set": "visual_ocr",
        "c": 0.1,
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
        "label": "画面+OCR（PCA64）",
    },
    {
        "feature_set": "visual_asr",
        "c": 0.1,
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
        "label": "画面+ASR（PCA64）",
    },
    {
        "feature_set": "combined",
        "c": 0.1,
        "alpha": 100.0,
        "reduction": "group_pca",
        "pca_components": 64,
        "label": "全模态（PCA64）",
    },
    {
        "feature_set": "combined",
        "c": 0.01,
        "alpha": 100.0,
        "label": "全模态（原始高维）",
    },
]


def _parse_date(value: str) -> datetime.date:
    """解析命令行日期。"""
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD格式")


def _feature_indices(feature_set: str) -> np.ndarray:
    """返回指定模态组合对应的特征列。"""
    groups = FEATURE_SET_GROUPS.get(feature_set)
    if not groups:
        raise ValueError("未知特征集合：{}".format(feature_set))
    parts = []
    for group_name in groups:
        start, end = FEATURE_GROUP_RANGES[group_name]
        parts.append(np.arange(start, end, dtype=np.int32))
    return np.concatenate(parts)


def build_time_masks(
    date_ordinals: np.ndarray,
    test_from: datetime.date,
    validation_days: int,
) -> Dict[str, np.ndarray]:
    """按日期构造训练、验证和锁定测试集合。"""
    test_start = test_from.toordinal()
    validation_start = test_start - validation_days
    masks = {
        "train": date_ordinals < validation_start,
        "validation": (
            (date_ordinals >= validation_start)
            & (date_ordinals < test_start)
        ),
        "train_validation": date_ordinals < test_start,
        "test": date_ordinals >= test_start,
    }
    for name in ("train", "validation", "test"):
        if int(masks[name].sum()) == 0:
            raise RuntimeError("{}时间集合为空".format(name))
    return masks


def _group_pca_transformer(feature_set: str, component_count: int):
    """按画面、OCR和ASR分别降维，统计特征只做标准化。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    semantic_dims = {
        "visual": VISUAL_SEMANTIC_DIM,
        "ocr": OCR_SEMANTIC_DIM,
        "asr": ASR_SEMANTIC_DIM,
    }
    transformers = []
    statistic_indices: List[int] = []
    offset = 0
    for group_name in FEATURE_SET_GROUPS[feature_set]:
        group_start, group_end = FEATURE_GROUP_RANGES[group_name]
        group_dim = group_end - group_start
        semantic_dim = semantic_dims.get(group_name, 0)
        if semantic_dim > 0:
            semantic_indices = list(range(offset, offset + semantic_dim))
            transformers.append((
                "{}语义降维".format(group_name),
                Pipeline([
                    ("标准化", StandardScaler()),
                    ("主成分", PCA(
                        n_components=int(component_count),
                        whiten=True,
                        svd_solver="randomized",
                        random_state=20260820,
                    )),
                ]),
                semantic_indices,
            ))
            statistic_indices.extend(range(offset + semantic_dim, offset + group_dim))
        else:
            statistic_indices.extend(range(offset, offset + group_dim))
        offset += group_dim
    if statistic_indices:
        transformers.append(("统计特征", StandardScaler(), statistic_indices))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


def _build_classifier(
    c_value: float,
    feature_set: str,
    reduction: str = "none",
    pca_components: int = 64,
):
    """创建适合小样本高维向量的强正则逻辑回归。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        max_iter=2000,
        random_state=20260819,
    )
    if reduction == "group_pca":
        return Pipeline([
            ("分模态降维", _group_pca_transformer(feature_set, pca_components)),
            ("模型", model),
        ])
    return Pipeline([("标准化", StandardScaler()), ("模型", model)])


def _build_regressor(
    alpha: float,
    feature_set: str,
    reduction: str = "none",
    pca_components: int = 64,
):
    """创建避免高维矩阵直接求逆的岭回归。"""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Ridge(alpha=float(alpha), solver="lsqr")
    if reduction == "group_pca":
        return Pipeline([
            ("分模态降维", _group_pca_transformer(feature_set, pca_components)),
            ("模型", model),
        ])
    return Pipeline([("标准化", StandardScaler()), ("模型", model)])


def _validate_binary_labels(values: np.ndarray, set_name: str) -> None:
    """确保分类集合同时包含零GMV和正GMV。"""
    if np.unique(values > 0).size < 2:
        raise RuntimeError("{}集合缺少零GMV或正GMV样本".format(set_name))


def _select_classifier(
    dataset: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> Tuple[dict, List[dict]]:
    """仅使用七月末验证集选择跑量分类器。"""
    labels = (dataset["labels"] > 0).astype(np.int32)
    results = []
    for candidate in CLASSIFIER_CANDIDATES:
        indices = _feature_indices(candidate["feature_set"])
        model = _build_classifier(
            float(candidate["c"]),
            candidate["feature_set"],
            str(candidate.get("reduction") or "none"),
            int(candidate.get("pca_components") or 64),
        )
        model.fit(dataset["features"][masks["train"]][:, indices], labels[masks["train"]])
        probability = model.predict_proba(
            dataset["features"][masks["validation"]][:, indices]
        )[:, 1]
        metrics = evaluate_classifier(labels[masks["validation"]], probability)
        results.append({"candidate": candidate, "metrics": metrics})
    best = max(
        results,
        key=lambda item: (
            item["metrics"]["pr_auc"],
            item["metrics"]["roc_auc"],
        ),
    )
    return dict(best["candidate"]), results


def _select_regressor(
    dataset: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> Tuple[dict, List[dict]]:
    """仅使用七月末正GMV样本选择条件回归器。"""
    labels = dataset["labels"]
    train_mask = masks["train"] & (labels > 0)
    validation_mask = masks["validation"] & (labels > 0)
    if int(train_mask.sum()) < 30 or int(validation_mask.sum()) < 10:
        raise RuntimeError("正GMV训练或验证样本不足")
    results = []
    for candidate in REGRESSOR_CANDIDATES:
        indices = _feature_indices(candidate["feature_set"])
        model = _build_regressor(
            float(candidate["alpha"]),
            candidate["feature_set"],
            str(candidate.get("reduction") or "none"),
            int(candidate.get("pca_components") or 64),
        )
        model.fit(
            dataset["features"][train_mask][:, indices],
            np.log1p(labels[train_mask]),
        )
        log_upper = float(np.quantile(np.log1p(labels[train_mask]), 0.995))
        predicted = restore_gmv_from_log(
            model.predict(dataset["features"][validation_mask][:, indices]),
            log_upper,
        )
        metrics = evaluate_regression(labels[validation_mask], predicted)
        results.append({"candidate": candidate, "metrics": metrics})
    best = min(
        results,
        key=lambda item: (
            item["metrics"]["log_mae"],
            -item["metrics"]["spearman"],
        ),
    )
    return dict(best["candidate"]), results


def _fit_two_stage(
    dataset: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    classifier_config: dict,
    regressor_config: dict,
) -> dict:
    """拟合是否跑量分类器和正GMV条件回归器。"""
    labels = dataset["labels"]
    classifier_indices = _feature_indices(classifier_config["feature_set"])
    classifier = _build_classifier(
        float(classifier_config["c"]),
        classifier_config["feature_set"],
        str(classifier_config.get("reduction") or "none"),
        int(classifier_config.get("pca_components") or 64),
    )
    classifier.fit(
        dataset["features"][train_mask][:, classifier_indices],
        (labels[train_mask] > 0).astype(np.int32),
    )

    positive_train_mask = train_mask & (labels > 0)
    regressor_indices = _feature_indices(regressor_config["feature_set"])
    regressor = _build_regressor(
        float(regressor_config["alpha"]),
        regressor_config["feature_set"],
        str(regressor_config.get("reduction") or "none"),
        int(regressor_config.get("pca_components") or 64),
    )
    regressor.fit(
        dataset["features"][positive_train_mask][:, regressor_indices],
        np.log1p(labels[positive_train_mask]),
    )
    log_upper = float(np.quantile(np.log1p(labels[positive_train_mask]), 0.995))
    probability = classifier.predict_proba(
        dataset["features"][predict_mask][:, classifier_indices]
    )[:, 1]
    conditional_gmv = restore_gmv_from_log(
        regressor.predict(dataset["features"][predict_mask][:, regressor_indices]),
        log_upper,
    )
    return {
        "classifier": classifier,
        "classifier_indices": classifier_indices,
        "regressor": regressor,
        "regressor_indices": regressor_indices,
        "log_upper": log_upper,
        "probability": probability,
        "conditional_gmv": conditional_gmv,
        "expected_gmv": probability * conditional_gmv,
    }


def _evaluate_predictions(
    actual: np.ndarray,
    probability: np.ndarray,
    conditional_gmv: np.ndarray,
    expected_gmv: np.ndarray,
) -> dict:
    """汇总分类、回归和头部素材筛选指标。"""
    positive_mask = actual > 0
    return {
        "classifier": evaluate_classifier(
            (actual > 0).astype(np.int32),
            probability,
        ),
        "regression": evaluate_regression(actual, expected_gmv),
        "positive_conditional_regression": evaluate_regression(
            actual[positive_mask],
            conditional_gmv[positive_mask],
        ),
        "probability_top20": evaluate_top_fraction(actual, probability, 0.2),
        "top10": evaluate_top_fraction(actual, expected_gmv, 0.1),
        "top20": evaluate_top_fraction(actual, expected_gmv, 0.2),
    }


def _run_ablations(
    dataset: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> List[dict]:
    """使用固定超参数比较各模态在锁定测试集上的贡献。"""
    results = []
    actual = dataset["labels"][test_mask]
    for config in ABLATION_CONFIGS:
        classifier_config = {
            "feature_set": config["feature_set"],
            "c": config["c"],
            "reduction": config.get("reduction", "none"),
            "pca_components": config.get("pca_components", 64),
        }
        regressor_config = {
            "feature_set": config["feature_set"],
            "alpha": config["alpha"],
            "reduction": config.get("reduction", "none"),
            "pca_components": config.get("pca_components", 64),
        }
        fitted = _fit_two_stage(
            dataset,
            train_mask,
            test_mask,
            classifier_config,
            regressor_config,
        )
        results.append({
            "feature_set": config["feature_set"],
            "label": config.get(
                "label",
                FEATURE_SET_LABELS[config["feature_set"]],
            ),
            "metrics": _evaluate_predictions(
                actual,
                fitted["probability"],
                fitted["conditional_gmv"],
                fitted["expected_gmv"],
            ),
        })
    return results


def _quality_scores(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """将预估日均GMV映射到训练分布百分位。"""
    sorted_reference = np.sort(reference)
    ranks = np.searchsorted(sorted_reference, values, side="right")
    return ranks.astype(np.float64) / float(sorted_reference.size) * 100.0


def _write_predictions(
    output_dir: str,
    dataset: Dict[str, np.ndarray],
    test_mask: np.ndarray,
    fitted: dict,
    quality_scores: np.ndarray,
) -> None:
    """保存时间外锁定测试集逐素材预测。"""
    path = os.path.join(output_dir, "test_predictions.csv")
    test_indices = np.flatnonzero(test_mask)
    with open(path, "w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "sample_id",
            "material_id",
            "first_stat_date",
            "delivery_day_count",
            "actual_gmv_per_delivery_day",
            "positive_probability",
            "conditional_positive_gmv_per_day",
            "predicted_gmv_per_delivery_day",
            "quality_score",
        ])
        for position, index in enumerate(test_indices):
            writer.writerow([
                int(dataset["sample_ids"][index]),
                str(dataset["material_ids"][index]),
                datetime.date.fromordinal(
                    int(dataset["date_ordinals"][index])
                ).isoformat(),
                int(dataset["delivery_day_counts"][index]),
                float(dataset["labels"][index]),
                float(fitted["probability"][position]),
                float(fitted["conditional_gmv"][position]),
                float(fitted["expected_gmv"][position]),
                float(quality_scores[position]),
            ])


def _write_report(output_dir: str, metrics: dict) -> None:
    """生成快速验证中文报告。"""
    test = metrics["test"]
    probability_top20 = test["metrics"]["probability_top20"]
    top20 = test["metrics"]["top20"]
    lines = [
        "# 多模态素材质量快速基线",
        "",
        "- 成熟样本：{}条".format(metrics["sample_count"]),
        "- 训练/验证/测试：{}/{}/{}".format(
            metrics["split_counts"]["train"],
            metrics["split_counts"]["validation"],
            metrics["split_counts"]["test"],
        ),
        "- 标签：实际投放日均GMV",
        "- 分类模型：{}".format(metrics["selected_classifier"]["name"]),
        "- 条件回归模型：{}".format(metrics["selected_regressor"]["name"]),
        "",
        "## 时间外锁定测试结果",
        "",
        "- 正GMV率：{:.2%}".format(test["metrics"]["classifier"]["positive_rate"]),
        "- PR-AUC：{:.4f}".format(test["metrics"]["classifier"]["pr_auc"]),
        "- ROC-AUC：{:.4f}".format(test["metrics"]["classifier"]["roc_auc"]),
        "- Brier：{:.4f}".format(test["metrics"]["classifier"]["brier"]),
        "- 日均GMV秩相关：{:.4f}".format(test["metrics"]["regression"]["spearman"]),
        "- 正GMV样本条件金额秩相关：{:.4f}".format(
            test["metrics"]["positive_conditional_regression"]["spearman"]
        ),
        "- 日均GMV对数MAE：{:.4f}".format(test["metrics"]["regression"]["log_mae"]),
        "- 跑量概率Top20%正GMV率提升：{:.3f}倍".format(
            probability_top20["positive_rate_lift"]
        ),
        "- 跑量概率Top20%平均日均GMV提升：{:.3f}倍".format(
            probability_top20["mean_gmv_lift"]
        ),
        "- 预计GMV Top20%平均日均GMV提升：{:.3f}倍".format(
            top20["mean_gmv_lift"]
        ),
        "- 预计GMV Top20%捕获GMV：{:.2%}".format(top20["gmv_capture"]),
        "- 预计GMV Top10%捕获GMV：{:.2%}".format(
            test["metrics"]["top10"]["gmv_capture"]
        ),
    ]
    ranking_intervals = test.get("ranking_bootstrap_95") or {}
    if ranking_intervals:
        spearman_interval = ranking_intervals.get("spearman", [0.0, 0.0])
        lift_interval = ranking_intervals.get("top20_mean_gmv_lift", [0.0, 0.0])
        capture_interval = ranking_intervals.get("top20_gmv_capture", [0.0, 0.0])
        lines.extend([
            "- 日均GMV秩相关95%区间：[{:.4f}, {:.4f}]".format(
                spearman_interval[0],
                spearman_interval[1],
            ),
            "- Top20%平均日均GMV提升95%区间：[{:.3f}, {:.3f}]倍".format(
                lift_interval[0],
                lift_interval[1],
            ),
            "- Top20%捕获GMV 95%区间：[{:.2%}, {:.2%}]".format(
                capture_interval[0],
                capture_interval[1],
            ),
        ])
    lines.extend([
        "",
        "## 模态消融",
        "",
        "| 特征 | ROC-AUC | PR-AUC | 概率Top20正GMV率提升 | 秩相关 | GMV Top20提升 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for item in metrics["ablations"]:
        item_metrics = item["metrics"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.3f}倍 | {:.4f} | {:.3f}倍 |".format(
                item["label"],
                item_metrics["classifier"]["roc_auc"],
                item_metrics["classifier"]["pr_auc"],
                item_metrics["probability_top20"]["positive_rate_lift"],
                item_metrics["regression"]["spearman"],
                item_metrics["top20"]["mean_gmv_lift"],
            )
        )
    lines.extend([
        "",
        "当前结果用于验证信号是否存在，样本仍在持续补充，不能直接作为线上承诺指标。",
        "完整候选结果、置信区间和逐素材预测请查看同目录文件。",
    ])
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def run_training(args: argparse.Namespace) -> dict:
    """从冻结快照训练并完成八月锁定测试。"""
    import joblib

    dataset, manifest = load_semantic_snapshot(args.snapshot_dir)
    masks = build_time_masks(
        dataset["date_ordinals"],
        args.test_from,
        args.validation_days,
    )
    for set_name in ("train", "validation", "test"):
        _validate_binary_labels(dataset["labels"][masks[set_name]], set_name)

    os.makedirs(args.output_dir, exist_ok=False)
    log_info("多模态训练快照已读取，成熟样本{}条，特征{}维".format(
        manifest["sample_count"],
        manifest["feature_dim"],
    ))
    best_classifier, classifier_results = _select_classifier(dataset, masks)
    best_regressor, regressor_results = _select_regressor(dataset, masks)
    log_info("验证集选择分类模型：{}".format(best_classifier["name"]))
    log_info("验证集选择条件回归模型：{}".format(best_regressor["name"]))

    fitted = _fit_two_stage(
        dataset,
        masks["train_validation"],
        masks["test"],
        best_classifier,
        best_regressor,
    )
    actual = dataset["labels"][masks["test"]]
    test_metrics = _evaluate_predictions(
        actual,
        fitted["probability"],
        fitted["conditional_gmv"],
        fitted["expected_gmv"],
    )

    train_reference = _fit_two_stage(
        dataset,
        masks["train_validation"],
        masks["train_validation"],
        best_classifier,
        best_regressor,
    )["expected_gmv"]
    quality_scores = _quality_scores(train_reference, fitted["expected_gmv"])
    ablations = _run_ablations(
        dataset,
        masks["train_validation"],
        masks["test"],
    )

    train_labels = dataset["labels"][masks["train_validation"]]
    baseline_probability = float((train_labels > 0).mean())
    positive_train_labels = train_labels[train_labels > 0]
    baseline_expected = baseline_probability * float(np.expm1(
        np.mean(np.log1p(positive_train_labels))
    ))
    baseline_prediction = np.full(actual.size, baseline_expected)

    metrics = {
        "dataset_sha256": manifest["dataset_sha256"],
        "sample_count": int(manifest["sample_count"]),
        "positive_count": int(manifest["positive_count"]),
        "feature_dim": int(manifest["feature_dim"]),
        "test_from": args.test_from.isoformat(),
        "validation_days": args.validation_days,
        "split_counts": {
            key: int(mask.sum()) for key, mask in masks.items()
        },
        "selected_classifier": best_classifier,
        "selected_regressor": best_regressor,
        "classifier_candidates": classifier_results,
        "regressor_candidates": regressor_results,
        "ablations": ablations,
        "test": {
            "sample_count": int(actual.size),
            "positive_count": int((actual > 0).sum()),
            "metrics": test_metrics,
            "constant_baseline": evaluate_regression(actual, baseline_prediction),
            "bootstrap_95": bootstrap_intervals(
                actual,
                fitted["probability"],
                fitted["expected_gmv"],
                np.ones(actual.size, dtype=np.float64),
                iterations=args.bootstrap_iterations,
            ),
            "ranking_bootstrap_95": bootstrap_ranking_intervals(
                actual,
                fitted["expected_gmv"],
                np.ones(actual.size, dtype=np.float64),
                iterations=args.bootstrap_iterations,
            ),
        },
    }

    bundle = {
        "cohort": manifest["cohort"],
        "feature_version": manifest["feature_version"],
        "label_version": manifest["label_version"],
        "dataset_sha256": manifest["dataset_sha256"],
        "feature_names": FEATURE_NAMES,
        "classifier": fitted["classifier"],
        "classifier_indices": fitted["classifier_indices"],
        "regressor": fitted["regressor"],
        "regressor_indices": fitted["regressor_indices"],
        "conditional_log_upper": fitted["log_upper"],
        "reference_expected_gmv": np.sort(train_reference),
    }
    joblib.dump(bundle, os.path.join(args.output_dir, "models.joblib"), compress=3)
    with open(
        os.path.join(args.output_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    _write_predictions(args.output_dir, dataset, masks["test"], fitted, quality_scores)
    _write_report(args.output_dir, metrics)
    log_info("多模态快速基线训练完成，产物目录：{}".format(args.output_dir))
    return metrics


def parse_args() -> argparse.Namespace:
    """读取快照导出或训练参数。"""
    parser = argparse.ArgumentParser(description="训练多模态V2素材质量快速基线")
    parser.add_argument("--cohort", default=DEFAULT_COHORT, help="样本范围版本")
    parser.add_argument(
        "--feature-version",
        default=DEFAULT_FEATURE_VERSION,
        help="多模态特征版本",
    )
    parser.add_argument("--export-snapshot", help="仅从数据库导出冻结快照到指定目录")
    parser.add_argument(
        "--export-local-snapshot",
        help="仅从本地V4原始特征档案导出冻结快照",
    )
    parser.add_argument(
        "--raw-feature-root",
        help="本地原始特征根目录，仅导出本地快照时使用",
    )
    parser.add_argument("--snapshot-dir", help="本地训练使用的冻结快照目录")
    parser.add_argument("--output-dir", help="模型和评估产物目录")
    parser.add_argument(
        "--test-from",
        type=_parse_date,
        default=datetime.date(2026, 8, 1),
        help="锁定测试集起始日期",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=7,
        help="测试日前用于模型选择的连续验证天数",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=200,
        help="测试指标置信区间抽样次数",
    )
    return parser.parse_args()


def main() -> int:
    """多模态快照导出和训练入口。"""
    set_process_name("material_quality_training")
    args = parse_args()
    if args.validation_days <= 0 or args.bootstrap_iterations <= 0:
        log_warn("验证天数和置信区间抽样次数必须大于0")
        return 2
    if args.export_snapshot:
        dataset, manifest = load_semantic_dataset(
            args.cohort,
            args.feature_version,
        )
        save_semantic_snapshot(args.export_snapshot, dataset, manifest)
        log_info("多模态训练快照导出完成，成熟样本{}条，目录：{}".format(
            manifest["sample_count"],
            args.export_snapshot,
        ))
        return 0
    if args.export_local_snapshot:
        if not args.raw_feature_root:
            log_warn("导出本地快照时必须指定原始特征根目录")
            return 2
        dataset, manifest = load_local_semantic_dataset(
            args.raw_feature_root,
            args.cohort,
            args.feature_version,
        )
        save_semantic_snapshot(args.export_local_snapshot, dataset, manifest)
        log_info("本地V4训练快照导出完成，成熟样本{}条，目录：{}".format(
            manifest["sample_count"],
            args.export_local_snapshot,
        ))
        return 0
    if not args.snapshot_dir or not args.output_dir:
        log_warn("训练时必须同时指定快照目录和产物目录")
        return 2
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""素材质量模型评估指标。"""

from typing import Dict, Optional

import numpy as np


def _weighted_mean(values: np.ndarray, sample_weight: Optional[np.ndarray]) -> float:
    """计算可选样本权重的均值。"""
    if values.size == 0:
        return 0.0
    if sample_weight is None:
        return float(values.mean())
    return float(np.average(values, weights=sample_weight))


def safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    """常数预测时把无法定义的秩相关记为0。"""
    if actual.size < 2 or np.all(predicted == predicted[0]) or np.all(actual == actual[0]):
        return 0.0
    from scipy.stats import spearmanr

    value = float(spearmanr(actual, predicted).correlation)
    return value if np.isfinite(value) else 0.0


def evaluate_classifier(
    actual: np.ndarray,
    probability: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """评估是否产生GMV的分类概率。"""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "positive_rate": _weighted_mean(actual.astype(np.float64), sample_weight),
        "pr_auc": float(average_precision_score(actual, clipped, sample_weight=sample_weight)),
        "roc_auc": float(roc_auc_score(actual, clipped, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(actual, clipped, sample_weight=sample_weight)),
    }


def evaluate_regression(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """同时评估原始GMV、对数误差和排序能力。"""
    safe_predicted = np.maximum(predicted, 0.0)
    errors = np.abs(actual - safe_predicted)
    log_errors = np.log1p(actual) - np.log1p(safe_predicted)
    return {
        "mae": _weighted_mean(errors, sample_weight),
        "log_mae": _weighted_mean(np.abs(log_errors), sample_weight),
        "rmsle": float(np.sqrt(_weighted_mean(log_errors ** 2, sample_weight))),
        "spearman": safe_spearman(actual, safe_predicted),
    }


def restore_gmv_from_log(predicted_log: np.ndarray, log_upper: float) -> np.ndarray:
    """在训练标签范围内还原GMV，防止线性候选指数溢出。"""
    if not np.isfinite(log_upper) or log_upper <= 0:
        raise ValueError("GMV对数还原上限必须是正有限数")
    clipped = np.clip(predicted_log, 0.0, log_upper)
    return np.expm1(clipped)


def evaluate_top_fraction(
    actual: np.ndarray,
    predicted: np.ndarray,
    fraction: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """评估模型优先选择一部分素材时的业务提升。"""
    if not 0 < fraction <= 1:
        raise ValueError("头部素材比例必须在0到1之间")
    count = max(1, int(np.ceil(actual.size * fraction)))
    selected = np.argsort(-predicted, kind="mergesort")[:count]
    weights = sample_weight[selected] if sample_weight is not None else None
    overall_positive_rate = _weighted_mean((actual > 0).astype(np.float64), sample_weight)
    selected_positive_rate = _weighted_mean((actual[selected] > 0).astype(np.float64), weights)
    overall_gmv = _weighted_mean(actual, sample_weight)
    selected_gmv = _weighted_mean(actual[selected], weights)
    total_gmv = float(np.sum(actual if sample_weight is None else actual * sample_weight))
    selected_total_gmv = float(np.sum(
        actual[selected] if weights is None else actual[selected] * weights
    ))
    return {
        "fraction": float(fraction),
        "selected_count": int(count),
        "positive_rate": selected_positive_rate,
        "positive_rate_lift": (
            selected_positive_rate / overall_positive_rate if overall_positive_rate > 0 else 0.0
        ),
        "mean_gmv": selected_gmv,
        "mean_gmv_lift": selected_gmv / overall_gmv if overall_gmv > 0 else 0.0,
        "gmv_capture": selected_total_gmv / total_gmv if total_gmv > 0 else 0.0,
    }


def bootstrap_intervals(
    actual: np.ndarray,
    probability: np.ndarray,
    predicted_gmv: np.ndarray,
    sample_weight: np.ndarray,
    iterations: int = 500,
    seed: int = 20260818,
) -> Dict[str, list]:
    """使用测试集自助抽样给关键指标提供不确定性范围。"""
    rng = np.random.RandomState(seed)
    values = {
        "pr_auc": [],
        "roc_auc": [],
        "spearman": [],
        "top20_positive_rate_lift": [],
    }
    for _ in range(iterations):
        indices = rng.randint(0, actual.size, size=actual.size)
        sampled_actual = actual[indices]
        if np.unique(sampled_actual > 0).size < 2:
            continue
        sampled_weight = sample_weight[indices]
        classifier = evaluate_classifier(
            (sampled_actual > 0).astype(np.int32),
            probability[indices],
            sampled_weight,
        )
        top20 = evaluate_top_fraction(
            sampled_actual,
            predicted_gmv[indices],
            0.2,
            sampled_weight,
        )
        values["pr_auc"].append(classifier["pr_auc"])
        values["roc_auc"].append(classifier["roc_auc"])
        values["spearman"].append(safe_spearman(sampled_actual, predicted_gmv[indices]))
        values["top20_positive_rate_lift"].append(top20["positive_rate_lift"])
    return {
        key: [float(np.quantile(metric_values, 0.025)), float(np.quantile(metric_values, 0.975))]
        for key, metric_values in values.items()
        if metric_values
    }


def bootstrap_ranking_intervals(
    actual: np.ndarray,
    predicted_gmv: np.ndarray,
    sample_weight: np.ndarray,
    iterations: int = 500,
    seed: int = 20260818,
) -> Dict[str, list]:
    """为GMV排序和头部素材提升计算自助抽样区间。"""
    rng = np.random.RandomState(seed)
    values = {
        "spearman": [],
        "top20_positive_rate_lift": [],
        "top20_mean_gmv_lift": [],
        "top20_gmv_capture": [],
    }
    for _ in range(iterations):
        indices = rng.randint(0, actual.size, size=actual.size)
        sampled_actual = actual[indices]
        sampled_prediction = predicted_gmv[indices]
        sampled_weight = sample_weight[indices]
        top20 = evaluate_top_fraction(
            sampled_actual,
            sampled_prediction,
            0.2,
            sampled_weight,
        )
        values["spearman"].append(safe_spearman(sampled_actual, sampled_prediction))
        values["top20_positive_rate_lift"].append(top20["positive_rate_lift"])
        values["top20_mean_gmv_lift"].append(top20["mean_gmv_lift"])
        values["top20_gmv_capture"].append(top20["gmv_capture"])
    return {
        key: [float(np.quantile(metric_values, 0.025)), float(np.quantile(metric_values, 0.975))]
        for key, metric_values in values.items()
        if metric_values
    }

#!/usr/bin/env python3
"""评估内容特征对同一广告账户内素材消耗的区分能力。"""

import argparse
import csv
import datetime
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sqlalchemy import bindparam, text

from material_quality.database import engine
from material_quality.training.script_dataset import load_script_augmented_dataset
from material_quality.training.train_regularized_classifier import _build_classifier
from material_quality.training.train_script_ablation import _build_matrix_and_layout
from plan_logger import log_info, log_warn, set_process_name


FEATURE_CONFIGS = [
    {"name": "media", "label": "仅媒体属性", "groups": ["media"]},
    {"name": "visual", "label": "画面", "groups": ["visual", "media"]},
    {"name": "script", "label": "脚本", "groups": ["script"]},
    {"name": "asr", "label": "ASR", "groups": ["asr", "media"]},
    {
        "name": "visual_script",
        "label": "画面+脚本拼接",
        "groups": ["visual", "media", "script"],
    },
]


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """分批查询素材ID，避免单次参数过多。"""
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _parse_date(value: str) -> datetime.date:
    """解析时间外测试日期。"""
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD格式")


def build_account_relative_labels(
    account_ids: np.ndarray,
    costs: np.ndarray,
    candidate_mask: np.ndarray,
    minimum_materials: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按账户内消耗中位数构造标签，并让每个账户和类别等权。"""
    labels = np.full(costs.shape, -1, dtype=np.int32)
    weights = np.zeros(costs.shape, dtype=np.float64)
    valid = np.zeros(costs.shape, dtype=bool)
    for account_id in np.unique(account_ids[candidate_mask]):
        indices = np.flatnonzero(candidate_mask & (account_ids == account_id))
        if indices.size < minimum_materials:
            continue
        account_costs = costs[indices]
        median_cost = float(np.median(account_costs))
        account_labels = (account_costs > median_cost).astype(np.int32)
        positive_count = int(account_labels.sum())
        negative_count = int(account_labels.size - positive_count)
        if positive_count <= 0 or negative_count <= 0:
            continue
        labels[indices] = account_labels
        valid[indices] = True
        weights[indices[account_labels == 1]] = 0.5 / float(positive_count)
        weights[indices[account_labels == 0]] = 0.5 / float(negative_count)
    return labels, weights, valid


def _load_account_spend_rows(
    cohort: str,
    material_ids: Sequence[str],
    batch_size: int,
) -> List[dict]:
    """从已聚合日表只读获取广告账户素材前7日消耗。"""
    query = text("""
        SELECT
            d.material_id,
            CAST(p.advertiser_id AS CHAR) AS advertiser_id,
            SUM(COALESCE(d.cost, 0)) AS cost_7d,
            SUM(COALESCE(d.gmv, 0)) AS gmv_7d,
            COUNT(DISTINCT d.stat_date) AS delivery_day_count,
            COUNT(DISTINCT d.account_plan_id) AS plan_count
        FROM material_quality_daily_stat AS d
        INNER JOIN material_quality_sample AS s
          ON s.cohort = d.cohort
         AND s.material_id = d.material_id
        INNER JOIN accountplan AS p ON p.id = d.account_plan_id
        WHERE d.cohort = :cohort
          AND d.material_id IN :material_ids
          AND d.stat_date >= s.first_stat_date
          AND d.stat_date < s.window_end_date
          AND p.advertiser_id IS NOT NULL
          AND p.advertiser_id <> ''
        GROUP BY d.material_id, p.advertiser_id
    """).bindparams(bindparam("material_ids", expanding=True))
    rows = []
    with engine.connect() as connection:
        for batch in _chunks(material_ids, batch_size):
            batch_rows = connection.execute(query, {
                "cohort": cohort,
                "material_ids": list(batch),
            }).mappings().all()
            rows.extend(dict(row) for row in batch_rows)
    return rows


def _build_pair_arrays(
    dataset: Mapping[str, np.ndarray],
    rows: Sequence[dict],
) -> Dict[str, np.ndarray]:
    """把账户素材行严格对齐到冻结内容特征。"""
    material_index = {
        str(material_id): index
        for index, material_id in enumerate(dataset["material_ids"].tolist())
    }
    valid_rows = [
        row for row in rows
        if str(row.get("material_id") or "") in material_index
    ]
    costs_7d = np.asarray([
        max(float(row.get("cost_7d") or 0.0), 0.0) for row in valid_rows
    ], dtype=np.float64)
    return {
        "material_indices": np.asarray([
            material_index[str(row["material_id"])] for row in valid_rows
        ], dtype=np.int32),
        "material_ids": np.asarray([
            str(row["material_id"]) for row in valid_rows
        ]),
        "account_ids": np.asarray([
            str(row["advertiser_id"]) for row in valid_rows
        ]),
        "costs_7d": costs_7d,
        "costs": costs_7d.copy(),
        "gmvs": np.asarray([
            max(float(row.get("gmv_7d") or 0.0), 0.0) for row in valid_rows
        ], dtype=np.float64),
        "delivery_days": np.asarray([
            int(row.get("delivery_day_count") or 0) for row in valid_rows
        ], dtype=np.int32),
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray, weights=None) -> float:
    """只有同时存在两类时才计算AUC。"""
    from sklearn.metrics import roc_auc_score

    if labels.size <= 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def _safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    """消耗或预测无变化时返回空值。"""
    from scipy.stats import spearmanr

    if actual.size < 3 or np.unique(actual).size < 2:
        return float("nan")
    if np.unique(predicted).size < 2:
        return float("nan")
    value = spearmanr(actual, predicted).statistic
    return float(value) if np.isfinite(value) else float("nan")


def evaluate_within_account(
    account_ids: np.ndarray,
    costs: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    valid_mask: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """计算账户等权AUC、秩相关和预测头部消耗提升。"""
    valid_indices = np.flatnonzero(valid_mask)
    pooled_auc = _safe_auc(
        labels[valid_indices],
        scores[valid_indices],
        weights[valid_indices],
    )
    account_auc_values = []
    account_spearman_values = []
    top30_lift_values = []
    for account_id in np.unique(account_ids[valid_mask]):
        indices = np.flatnonzero(valid_mask & (account_ids == account_id))
        account_auc = _safe_auc(labels[indices], scores[indices])
        if np.isfinite(account_auc):
            account_auc_values.append(account_auc)
        correlation = _safe_spearman(costs[indices], scores[indices])
        if np.isfinite(correlation):
            account_spearman_values.append(correlation)
        selected_count = max(int(math.ceil(indices.size * 0.3)), 1)
        ranking = indices[np.argsort(scores[indices])[::-1]]
        baseline_mean = float(np.mean(costs[indices]))
        selected_mean = float(np.mean(costs[ranking[:selected_count]]))
        if baseline_mean > 0:
            top30_lift_values.append(selected_mean / baseline_mean)
    return {
        "pair_count": int(valid_indices.size),
        "account_count": int(np.unique(account_ids[valid_mask]).size),
        "pooled_account_weighted_auc": pooled_auc,
        "macro_account_auc": (
            float(np.mean(account_auc_values)) if account_auc_values else float("nan")
        ),
        "macro_account_auc_std": (
            float(np.std(account_auc_values)) if account_auc_values else float("nan")
        ),
        "macro_account_spearman": (
            float(np.mean(account_spearman_values))
            if account_spearman_values else float("nan")
        ),
        "macro_top30_cost_lift": (
            float(np.mean(top30_lift_values)) if top30_lift_values else float("nan")
        ),
    }


def _account_distribution(account_ids: np.ndarray, costs: np.ndarray) -> dict:
    """描述同账户素材数量和消耗差异，为可分性提供数据基础。"""
    counts = []
    positive_rates = []
    coefficient_values = []
    for account_id in np.unique(account_ids):
        values = costs[account_ids == account_id]
        counts.append(int(values.size))
        positive_rates.append(float((values > 0).mean()))
        mean_value = float(np.mean(values))
        if values.size >= 2 and mean_value > 0:
            coefficient_values.append(float(np.std(values) / mean_value))
    count_values = np.asarray(counts, dtype=np.int32)
    return {
        "pair_count": int(costs.size),
        "material_count": None,
        "account_count": int(count_values.size),
        "accounts_with_4_materials": int((count_values >= 4).sum()),
        "accounts_with_8_materials": int((count_values >= 8).sum()),
        "median_materials_per_account": float(np.median(count_values)),
        "p75_materials_per_account": float(np.quantile(count_values, 0.75)),
        "overall_zero_cost_rate": float((costs <= 0).mean()),
        "macro_positive_cost_rate": float(np.mean(positive_rates)),
        "macro_cost_coefficient_of_variation": (
            float(np.mean(coefficient_values)) if coefficient_values else float("nan")
        ),
    }


def _fit_spend_models(
    dataset: Mapping[str, np.ndarray],
    pair_data: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    test_labels: np.ndarray,
    test_weights: np.ndarray,
) -> Tuple[List[dict], Dict[str, np.ndarray]]:
    """用固定PCA8和强正则训练内容消耗模型。"""
    material_indices = pair_data["material_indices"]
    probabilities = {}
    results = []
    for config in FEATURE_CONFIGS:
        matrix, semantic_blocks, statistic_indices = _build_matrix_and_layout(
            dataset,
            config["groups"],
        )
        pair_matrix = matrix[material_indices]
        model = _build_classifier(
            semantic_blocks,
            statistic_indices,
            pca_components=8,
            c_value=0.001,
        )
        model.fit(
            pair_matrix[train_mask],
            train_labels[train_mask],
            **{"模型__sample_weight": train_weights[train_mask]},
        )
        scores = model.predict_proba(pair_matrix)[:, 1]
        probabilities[config["name"]] = scores
        test_metrics = evaluate_within_account(
            pair_data["account_ids"],
            pair_data["costs"],
            test_labels,
            test_weights,
            test_mask,
            scores,
        )
        results.append({
            "name": config["name"],
            "label": config["label"],
            "test": test_metrics,
        })
    fusion_scores = (
        probabilities["visual"] * 0.5 + probabilities["script"] * 0.5
    )
    probabilities["visual_script_late"] = fusion_scores
    results.append({
        "name": "visual_script_late",
        "label": "画面50%+脚本50%晚融合",
        "test": evaluate_within_account(
            pair_data["account_ids"],
            pair_data["costs"],
            test_labels,
            test_weights,
            test_mask,
            fusion_scores,
        ),
    })
    return results, probabilities


def _evaluate_existing_quality_score(
    dataset: Mapping[str, np.ndarray],
    pair_data: Mapping[str, np.ndarray],
    model_path: str,
    test_labels: np.ndarray,
    test_weights: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    """评估现有GMV质量分是否也能在同账户内排序消耗。"""
    import joblib

    bundles = joblib.load(model_path)
    scores_by_name = {}
    for name, groups in (
        ("visual", ["visual", "media"]),
        ("script", ["script"]),
    ):
        matrix, _, _ = _build_matrix_and_layout(dataset, groups)
        scores_by_name[name] = bundles[name]["classifier"].predict_proba(
            matrix
        )[:, 1]
    material_scores = scores_by_name["visual"] * 0.5 + scores_by_name["script"] * 0.5
    pair_scores = material_scores[pair_data["material_indices"]]
    return evaluate_within_account(
        pair_data["account_ids"],
        pair_data["costs"],
        test_labels,
        test_weights,
        test_mask,
        pair_scores,
    )


def _write_report(output_dir: str, metrics: dict) -> None:
    """写出账户内消耗可分性中文报告。"""
    distribution = metrics["distribution"]
    lines = [
        "# 同账户素材消耗可分性评估",
        "",
        "- 消耗口径：{}".format(metrics["cost_target_label"]),
        "- 账户素材对：{}".format(distribution["pair_count"]),
        "- 唯一素材：{}".format(distribution["material_count"]),
        "- 广告账户：{}".format(distribution["account_count"]),
        "- 至少4条素材的账户：{}".format(
            distribution["accounts_with_4_materials"]
        ),
        "- 零消耗素材占比：{:.2%}".format(
            distribution["overall_zero_cost_rate"]
        ),
        "",
    ]
    for split in metrics["splits"]:
        lines.extend([
            "## 时间外测试：{}".format(split["test_from"]),
            "",
            "- 训练账户素材对：{}".format(split["train_pair_count"]),
            "- 测试账户素材对：{}".format(split["test_pair_count"]),
            "- 测试账户：{}".format(split["test_account_count"]),
            "- 现有GMV质量分账户内AUC：{:.4f}".format(
                split["existing_quality_score"]["macro_account_auc"]
            ),
            "",
            "| 内容特征 | 账户等权AUC | 账户宏平均AUC | 账户内Spearman | Top30%消耗提升 |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in split["spend_models"]:
            test_metrics = item["test"]
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.3f}x |".format(
                    item["label"],
                    test_metrics["pooled_account_weighted_auc"],
                    test_metrics["macro_account_auc"],
                    test_metrics["macro_account_spearman"],
                    test_metrics["macro_top30_cost_lift"],
                )
            )
        lines.append("")
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as output:
        output.write("\n".join(lines))


def run_evaluation(args: argparse.Namespace) -> dict:
    """执行同账户素材消耗可分性评估。"""
    dataset, manifest = load_script_augmented_dataset(
        args.snapshot_dir,
        args.script_feature_root,
        args.script_feature_version,
    )
    rows = _load_account_spend_rows(
        args.cohort,
        [str(value) for value in dataset["material_ids"].tolist()],
        args.query_batch_size,
    )
    pair_data = _build_pair_arrays(dataset, rows)
    if args.cost_target == "daily":
        pair_data["costs"] = pair_data["costs_7d"] / np.maximum(
            pair_data["delivery_days"].astype(np.float64),
            1.0,
        )
    distribution = _account_distribution(
        pair_data["account_ids"],
        pair_data["costs"],
    )
    distribution["material_count"] = int(
        np.unique(pair_data["material_ids"]).size
    )
    pair_dates = dataset["date_ordinals"][pair_data["material_indices"]]
    split_results = []
    prediction_columns = {}
    for split_index, test_from in enumerate(args.test_from):
        cutoff = test_from.toordinal()
        train_candidates = pair_dates < cutoff
        test_candidates = pair_dates >= cutoff
        train_labels, train_weights, train_valid = build_account_relative_labels(
            pair_data["account_ids"],
            pair_data["costs"],
            train_candidates,
            args.minimum_account_materials,
        )
        test_labels, test_weights, test_valid = build_account_relative_labels(
            pair_data["account_ids"],
            pair_data["costs"],
            test_candidates,
            args.minimum_account_materials,
        )
        if np.unique(train_labels[train_valid]).size < 2:
            raise RuntimeError("训练时间窗没有足够的账户内高低消耗样本")
        if np.unique(test_labels[test_valid]).size < 2:
            raise RuntimeError("测试时间窗没有足够的账户内高低消耗样本")
        spend_results, probabilities = _fit_spend_models(
            dataset,
            pair_data,
            train_valid,
            test_valid,
            train_labels,
            train_weights,
            test_labels,
            test_weights,
        )
        existing_score = _evaluate_existing_quality_score(
            dataset,
            pair_data,
            args.existing_model_path[split_index],
            test_labels,
            test_weights,
            test_valid,
        )
        split_results.append({
            "test_from": test_from.isoformat(),
            "train_pair_count": int(train_valid.sum()),
            "train_account_count": int(np.unique(
                pair_data["account_ids"][train_valid]
            ).size),
            "test_pair_count": int(test_valid.sum()),
            "test_account_count": int(np.unique(
                pair_data["account_ids"][test_valid]
            ).size),
            "existing_quality_score": existing_score,
            "spend_models": spend_results,
        })
        for name, values in probabilities.items():
            prediction_columns["{}_{}".format(test_from.isoformat(), name)] = values
    metrics = {
        "base_dataset_sha256": manifest["base_dataset_sha256"],
        "script_dataset_sha256": manifest["script_dataset_sha256"],
        "cohort": args.cohort,
        "cost_target": args.cost_target,
        "cost_target_label": (
            "账户素材7日窗口日均消耗"
            if args.cost_target == "daily" else "账户素材7日总消耗"
        ),
        "minimum_account_materials": int(args.minimum_account_materials),
        "distribution": distribution,
        "splits": split_results,
    }
    os.makedirs(args.output_dir, exist_ok=False)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.output_dir, "account_material_pairs.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as output:
        writer = csv.writer(output)
        prediction_names = sorted(prediction_columns)
        writer.writerow([
            "material_id", "advertiser_id", "first_stat_date",
            "cost_7d", "target_cost", "gmv_7d", "delivery_day_count",
        ] + prediction_names)
        for index in range(pair_data["costs"].size):
            material_index = int(pair_data["material_indices"][index])
            writer.writerow([
                str(pair_data["material_ids"][index]),
                str(pair_data["account_ids"][index]),
                datetime.date.fromordinal(
                    int(dataset["date_ordinals"][material_index])
                ).isoformat(),
                float(pair_data["costs_7d"][index]),
                float(pair_data["costs"][index]),
                float(pair_data["gmvs"][index]),
                int(pair_data["delivery_days"][index]),
            ] + [
                float(prediction_columns[name][index]) for name in prediction_names
            ])
    _write_report(args.output_dir, metrics)
    log_info("同账户素材消耗可分性评估完成，目录：{}".format(args.output_dir))
    return metrics


def parse_args() -> argparse.Namespace:
    """读取评估参数。"""
    parser = argparse.ArgumentParser(description="评估同账户素材消耗可分性")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--script-feature-root", required=True)
    parser.add_argument("--script-feature-version", required=True)
    parser.add_argument(
        "--existing-model-path",
        action="append",
        required=True,
        help="必须与时间外测试日期一一对应",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cohort", default="all_industries_v1")
    parser.add_argument("--minimum-account-materials", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=300)
    parser.add_argument(
        "--cost-target",
        choices=("total", "daily"),
        default="daily",
        help="使用7日总消耗或按实际投放天数归一化的日均消耗",
    )
    parser.add_argument(
        "--test-from",
        action="append",
        type=_parse_date,
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    """启动账户内消耗可分性评估。"""
    set_process_name("material_quality_account_spend")
    args = parse_args()
    if args.minimum_account_materials < 3 or args.query_batch_size <= 0:
        log_warn("账户最少素材数必须至少3，查询批次大小必须大于0")
        return 2
    if len(args.existing_model_path) != len(args.test_from):
        log_warn("现有模型路径必须与时间外测试日期数量一致")
        return 2
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

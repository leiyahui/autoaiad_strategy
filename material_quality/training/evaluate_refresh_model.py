"""使用固定双塔方案和七日隔离窗口评估增量训练的真实收益。"""

import argparse
import datetime
from pathlib import Path

import numpy as np

from material_quality.inference.model import MaterialQualityPredictor
from material_quality.sample.refresh_training_data import atomic_json
from material_quality.training.evaluation import evaluate_classifier, evaluate_top_fraction
from material_quality.training.script_dataset import load_script_augmented_dataset
from material_quality.training.train_serving_model import _fit_serving_towers
from plan_logger import log_info, set_process_name


def purged_masks(dates: np.ndarray, start: datetime.date, end: datetime.date = None) -> tuple:
    """确保训练标签窗口在预测起始日之前闭合。"""
    boundary = start.toordinal()
    train = dates+7<=boundary
    test = dates>=boundary
    if end is not None:
        test &= dates<end.toordinal()
    return train,test


def fit_predict(dataset: dict, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """只对过去样本拟合降维和分类器，再预测未来样本。"""
    from material_quality.inference.model import build_serving_tower_matrices

    labels = (dataset["labels"]>0).astype(np.int32)
    if np.unique(labels[train]).size!=2 or np.unique(labels[test]).size!=2:
        raise ValueError("时间切分必须同时包含正样本与零样本")
    models,_ = _fit_serving_towers({k:v[train] for k,v in dataset.items()},labels[train])
    matrices = build_serving_tower_matrices(dataset["features"][test],dataset["script_features"][test],
                                           dataset["script_features"].shape[1])
    return 0.5*models["visual"].predict_proba(matrices["visual"])[:,1]+0.5*models["script"].predict_proba(matrices["script"])[:,1]


def paired_auc_intervals(labels: np.ndarray, current: np.ndarray, previous: np.ndarray,
                         iterations: int = 1000) -> dict:
    """按相同样本重抽样估计新旧模型差值，保留配对比较关系。"""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(20261011)
    new_values, deltas = [], []
    for _ in range(iterations):
        indices = rng.randint(0,len(labels),len(labels))
        if np.unique(labels[indices]).size<2:
            continue
        new_auc = roc_auc_score(labels[indices],current[indices])
        old_auc = roc_auc_score(labels[indices],previous[indices])
        new_values.append(new_auc)
        deltas.append(new_auc-old_auc)
    if not deltas:
        raise ValueError("自助抽样未得到同时包含两类标签的样本")
    return {"new_auc_95_interval":np.quantile(new_values,[0.025,0.975]).tolist(),
            "auc_delta_95_interval":np.quantile(deltas,[0.025,0.975]).tolist(),
            "bootstrap_unit":"内容去重后的素材，未按账户聚类",
            "bootstrap_iterations":iterations}


def evaluate(args: argparse.Namespace) -> dict:
    """先写入固定评估方案，再执行滚动验证和一次锁定测试。"""
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    test_start = datetime.date.fromisoformat(args.test_from)
    protocol = {"test_from":args.test_from,"label_window_days":7,"fusion_weights":{"visual":0.5,"script":0.5},
                "profile":{"pca_components":8,"c":0.001},"rolling_folds":3,"validation_days":7,
                "selection":"沿用既有方案，不依据本轮验证或测试结果调参"}
    atomic_json(output/"protocol.json",protocol)
    dataset, manifest = load_script_augmented_dataset(args.snapshot_dir,args.script_feature_root,
                                                      "codex_agent_script_cnclip_v1")
    old = MaterialQualityPredictor.load(args.previous_model_dir)
    all_labels = (dataset["labels"]>0).astype(np.int32)
    results = []
    for offset in [21,14,7,0]:
        start = test_start-datetime.timedelta(days=offset)
        end = start+datetime.timedelta(days=7) if offset else None
        train,test = purged_masks(dataset["date_ordinals"],start,end)
        log_info("开始时间验证：{}，训练{}条，评估{}条".format(start,int(train.sum()),int(test.sum())))
        new_probability = fit_predict(dataset,train,test)
        old_probability = old.predict_arrays(dataset["features"][test],dataset["script_features"][test])["positive_probability"]
        labels = all_labels[test]
        result = {"kind":"rolling_validation" if offset else "locked_test","from":start.isoformat(),
                  "to_exclusive":end.isoformat() if end else None,"train_count":int(train.sum()),
                  "test_count":int(test.sum()),"positive_count":int(labels.sum()),
                  "new":evaluate_classifier(labels,new_probability),
                  "previous":evaluate_classifier(labels,old_probability),
                  "new_top20":evaluate_top_fraction(dataset["labels"][test],new_probability,0.2),
                  "previous_top20":evaluate_top_fraction(dataset["labels"][test],old_probability,0.2)}
        if not offset:
            result.update(paired_auc_intervals(labels,new_probability,old_probability))
            np.savez_compressed(output/"locked_test_predictions.npz",content_sha256=dataset["content_sha256"][test],
                                labels=labels,new_probability=new_probability,previous_probability=old_probability)
        results.append(result)
        atomic_json(output/"progress.json",{"completed_folds":len(results),"total_folds":4})
    metrics = {"protocol":protocol,"dataset_sha256":manifest["base_dataset_sha256"],
               "script_dataset_sha256":manifest["script_dataset_sha256"],
               "previous_model_version":old.manifest["model_version"],"sample_count":len(all_labels),"results":results}
    atomic_json(output/"metrics.json",metrics)
    lines = ["# 增量训练时间外评估","","训练和测试之间隔离7日标签窗口；画面与脚本各50%，不在测试集调参。","",
             "| 起始日 | 训练数 | 评估数 | 新模型AUC | 旧模型AUC | 新模型PR-AUC | 旧模型PR-AUC |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            r["from"],r["train_count"],r["test_count"],r["new"]["roc_auc"],r["previous"]["roc_auc"],
            r["new"]["pr_auc"],r["previous"]["pr_auc"]))
    lines += ["","评估仅适用于本次可获取素材；账户与投放预算等混杂因素尚未消除。",
              "置信区间按素材重抽样，未按账户聚类。全量训练后的模型不使用训练集AUC声称泛化效果。"]
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    log_info("时间外评估完成：{}".format(output))
    return metrics


def main() -> None:
    """读取评估输入，固定时间边界后运行离线比较。"""
    set_process_name("material_quality_refresh_eval")
    parser = argparse.ArgumentParser(description="评估增量样本的固定双塔模型")
    parser.add_argument("--snapshot-dir",required=True)
    parser.add_argument("--script-feature-root",required=True)
    parser.add_argument("--previous-model-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--test-from",default="2026-09-15")
    evaluate(parser.parse_args())


if __name__=="__main__":
    main()

"""暂停采集后冻结已完成内容，要求所有入选素材的脚本特征完整。"""

import argparse
import datetime
import json
from collections import Counter
from pathlib import Path

import numpy as np

from material_quality.feature.refresh_collection import VERSION
from material_quality.inference.features import load_script_feature_vector
from material_quality.sample.refresh_training_data import atomic_json
from material_quality.training.semantic_dataset import load_local_semantic_dataset, save_semantic_snapshot
from plan_logger import log_info, set_process_name


def freeze_snapshot(artifact_root: str, output_dir: str) -> dict:
    """按内容去重后校验双模态输入，避免遗漏脚本改变评估人群。"""
    artifacts = Path(artifact_root).resolve()
    raw = artifacts/"raw_features"/VERSION
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("快照目录已存在，禁止覆盖冻结样本")
    selected = {}
    for path in (raw/"assets").glob("*.json"):
        reference = json.loads(path.read_text())
        if reference.get("label_status")!="ready" or int(reference.get("delivery_day_count") or 0)<2:
            continue
        key = reference["content_sha256"]
        previous = selected.get(key)
        if previous is None or (reference["first_stat_date"],reference["asset_id"]) < (
                previous["first_stat_date"],previous["asset_id"]):
            selected[key] = reference
    if not selected:
        raise ValueError("没有成熟内容可供冻结")
    for key in selected:
        load_script_feature_vector(str(artifacts/"script_features_luna_consistent"),
                                   "codex_agent_script_cnclip_v1",key)
    # 新建视图保留原始内容档案，固定本次使用的标签引用。
    view = output.parent/(output.name+"_references")/VERSION
    (view/"assets").mkdir(parents=True,exist_ok=False)
    (view/"content").symlink_to(raw/"content",target_is_directory=True)
    for key, reference in selected.items():
        atomic_json(view/"assets"/(str(reference["asset_id"])+".json"),reference)
    dataset, manifest = load_local_semantic_dataset(str(view.parent),"all_industries_v1",VERSION)
    if not np.isfinite(dataset["features"]).all():
        raise ValueError("原始特征包含非有限值")
    if manifest["sample_count"]!=len(selected):
        raise ValueError("冻结样本数量与入选内容数量不一致")
    manifest.update({"script_feature_version":"codex_agent_script_cnclip_v1",
                     "script_complete_count":len(selected),
                     "month_counts":dict(Counter(datetime.date.fromordinal(int(v)).isoformat()[:7]
                                                  for v in dataset["date_ordinals"]))})
    save_semantic_snapshot(str(output),dataset,manifest)
    log_info("冻结完成：{}条，正样本{}条，零样本{}条".format(
        manifest["sample_count"],manifest["positive_count"],manifest["zero_count"]))
    return manifest


def main() -> None:
    """读取本地归档参数并冻结训练快照。"""
    set_process_name("material_quality_freeze")
    parser = argparse.ArgumentParser(description="冻结完整本地训练样本")
    parser.add_argument("--artifact-root",required=True)
    parser.add_argument("--output-dir",required=True)
    args = parser.parse_args()
    freeze_snapshot(args.artifact_root,args.output_dir)


if __name__=="__main__":
    main()

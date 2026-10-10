"""协调样本刷新、特征采集和脚本队列，形成完整本地训练快照。"""

import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from material_quality.feature.codex_script_supervisor import read_codex_usage
from material_quality.feature.refresh_collection import VERSION
from material_quality.inference.features import load_script_feature_vector
from material_quality.sample.refresh_training_data import atomic_json
from material_quality.training.semantic_dataset import (
    _dataset_digest, load_local_semantic_dataset, save_semantic_snapshot,
)
from plan_logger import log_info, log_warn, set_process_name


def current_content(raw_root: Path) -> list:
    """只把成熟素材的无标签内容哈希交给脚本理解队列。"""
    hashes = set()
    for path in (raw_root/VERSION/"assets").glob("*.json"):
        value = json.loads(path.read_text())
        if value.get("label_status")=="ready" and int(value.get("delivery_day_count") or 0)>=2:
            hashes.add(value["content_sha256"])
    return sorted(hashes)


def run(args: argparse.Namespace) -> None:
    """持续消费采集结果，所有进程退出后校验完整特征并冻结训练数据。"""
    root = Path(args.runtime)
    artifacts = Path(args.artifact_root)
    raw_root = artifacts/"raw_features"
    script_root = artifacts/"script_features_luna_consistent"
    log_info("刷新流水线启动，目标{}条，特征并发{}，脚本并发{}，模型GPT-5.6 Luna，推理medium".format(
        args.target_count,args.feature_workers,args.script_workers))
    version = "codex_agent_script_cnclip_v1"
    command = [sys.executable,"-u","-m","material_quality.feature.refresh_collection",
               "--runtime",str(root),"--artifact-root",str(artifacts),"--config",args.config,
               "--workers",str(args.feature_workers),"--target-count",str(args.target_count)]
    log = (root/"collection.log").open("ab")
    collection = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    log.close()
    active = None
    wave = 0
    failed_hashes = set()
    while True:
        if (root/"STOP").exists():
            log_warn("收到停止请求，等待当前素材任务结束并关闭脚本队列")
            if active and active.poll() is None:
                active.terminate()
            atomic_json(root/"pipeline_status.json",{"phase":"stopping","collection_pid":collection.pid})
            collection.wait()
            if active:
                active.wait()
            return
        missing = []
        hashes = current_content(raw_root)
        for content_hash in hashes:
            if content_hash in failed_hashes:
                continue
            base = script_root/version/"content"/content_hash[:2]/content_hash
            if not base.with_suffix(".npz").exists() or not base.with_suffix(".json").exists():
                missing.append(content_hash)
        if active and active.poll() is not None:
            if active.returncode:
                status_path = root/"script_waves"/str(wave)/"status.json"
                if status_path.exists():
                    state = json.loads(status_path.read_text())
                    failed_hashes.update(state.get("exhausted_hashes") or [])
                if not failed_hashes:
                    raise RuntimeError("脚本队列异常退出，请查看对应批次日志")
            active = None
            missing = [value for value in missing if value not in failed_hashes]
        if not active and missing:
            wave += 1
            wave_root = root/"script_waves"/str(wave)
            wave_root.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(wave_root/"queue.npz",content_sha256=np.asarray(missing[:200]))
            cmd = [sys.executable,"-u","material_quality/feature/codex_script_supervisor.py",
                   "--repo-root",os.getcwd(),"--snapshot-path",str(wave_root/"queue.npz"),
                   "--source-root",str(raw_root),"--output-root",str(script_root),
                   "--runtime-directory",str(wave_root),"--model-cache",str(artifacts/"models/chinese_clip"),
                   "--workers",str(args.script_workers),"--torch-threads","2",
                   "--model","gpt-5.6-luna","--reasoning-effort","medium"]
            with (wave_root/"supervisor.log").open("ab") as output:
                active = subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT)
            log_info("启动脚本波次{}，{}条，并发{}".format(wave,min(len(missing),200),args.script_workers))
        usage = Counter()
        for path in (root/"script_waves").glob("*/jobs/*/codex.log"):
            usage.update(read_codex_usage(str(path)))
        exported = (root/"source_manifest.json").exists() and json.loads(
            (root/"source_manifest.json").read_text()).get("complete")
        status = {"phase":"collecting" if exported else "waiting_export","collection_pid":collection.pid,
                  "collection_exit":collection.poll(),"script_wave":wave,
                  "script_pid":active.pid if active else None,"ready_content":len(hashes),
                  "pending_scripts":len(missing),"failed_scripts":len(failed_hashes),
                  "reported_usage":dict(usage),"updated_at":datetime.datetime.now().isoformat()}
        atomic_json(root/"pipeline_status.json",status)
        if collection.poll() is not None and not active and not missing:
            if collection.returncode:
                raise RuntimeError("素材采集进程异常退出，请查看采集日志")
            break
        time.sleep(15)
    collection_state = json.loads((root/"collection_status.json").read_text())
    if collection_state["phase"]!="finished":
        log_warn("小批采集未通过扩大条件，保留已有成果供检查")
        atomic_json(root/"pipeline_status.json",{**status,"phase":"needs_attention"})
        return
    view = root/"snapshot_features"/VERSION
    (view/"assets").mkdir(parents=True,exist_ok=True)
    if not (view/"content").exists():
        (view/"content").symlink_to(raw_root/VERSION/"content",target_is_directory=True)
    selected = {}
    for path in (raw_root/VERSION/"assets").glob("*.json"):
        reference = json.loads(path.read_text())
        if reference.get("label_status")!="ready":
            continue
        content_hash = reference["content_sha256"]
        previous = selected.get(content_hash)
        if previous is None or (reference["first_stat_date"],reference["asset_id"]) < (
            previous["first_stat_date"],previous["asset_id"]):
            selected[content_hash] = reference
    for reference in selected.values():
        atomic_json(view/"assets"/(str(reference["asset_id"])+".json"),reference)
    dataset,manifest = load_local_semantic_dataset(str(view.parent),"all_industries_v1",VERSION)
    indices = []
    for index,content_hash in enumerate(dataset["content_sha256"].tolist()):
        try:
            load_script_feature_vector(str(script_root),version,str(content_hash))
            indices.append(index)
        except (OSError,ValueError,KeyError):
            continue
    dataset = {key:value[indices] for key,value in dataset.items()}
    manifest.update({"sample_count":len(indices),"positive_count":int((dataset["labels"]>0).sum()),
                     "zero_count":int((dataset["labels"]==0).sum()),
                     "script_feature_version":version,"source_refresh":str(root),
                     "ready_without_script":len(hashes)-len(indices)})
    manifest["dataset_sha256"] = _dataset_digest(dataset)
    manifest["min_first_stat_date"] = datetime.date.fromordinal(int(dataset["date_ordinals"].min())).isoformat()
    manifest["max_first_stat_date"] = datetime.date.fromordinal(int(dataset["date_ordinals"].max())).isoformat()
    manifest["primary_industry_counts"] = dict(Counter(dataset["primary_industries"].tolist()))
    save_semantic_snapshot(str(root/"snapshot"),dataset,manifest)
    atomic_json(root/"pipeline_status.json",{**status,"phase":"samples_complete",
                "sample_count":len(indices),"snapshot":str(root/"snapshot")})
    log_info("最新完整训练样本已冻结，共{}条".format(len(indices)))


def main() -> None:
    """启动本地刷新流水线。"""
    set_process_name("material_quality_refresh_pipeline")
    parser = argparse.ArgumentParser(description="刷新训练数据完整流水线")
    parser.add_argument("--runtime",required=True)
    parser.add_argument("--artifact-root",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--feature-workers",type=int,default=6)
    parser.add_argument("--script-workers",type=int,default=6)
    parser.add_argument("--target-count",type=int,default=10000)
    args = parser.parse_args()
    lock_path = Path(args.runtime)/"pipeline.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("该运行目录已有刷新流水线")
        run(args)


if __name__=="__main__":
    main()

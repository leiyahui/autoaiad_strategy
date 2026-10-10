"""消费本地冻结样本，以独立进程并发采集可复用内容特征。"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List

from sqlalchemy import create_engine, text

from material_quality.sample.refresh_training_data import load_database_url, atomic_json
from material_quality.feature.local_artifact_store import LocalRawFeatureStore
from material_quality.feature.local_url_resolver import _auth_status, _request_video, _valid_video_url
from material_quality.feature.media_features import download_video
from plan_logger import log_info, log_warn, set_process_name


VERSION = "multimodal_cnclip_whisper_replayable_v4"
_engine = None
_extractor = None
_auth = {}
_auth_loaded = 0.0
_settings = {}


def initialize_worker(settings: dict) -> None:
    """每进程独立加载模型，避免多个线程争用推理引擎。"""
    global _engine, _settings
    _settings = settings
    set_process_name("material_quality_collection")
    os.environ["MATERIAL_QUALITY_RATE_LIMIT_FILE"] = settings["rate_limit_file"]
    os.environ["MATERIAL_QUALITY_URL_INTERVAL_SECONDS"] = "1.0"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    _engine = create_engine(load_database_url(settings["config"]), pool_size=1,
                            max_overflow=0, connect_args={"read_timeout":30,"connect_timeout":10})


def load_auth() -> Dict[int, dict]:
    """授权只留在进程内存，定时重读最新令牌，不刷新或写入线上授权。"""
    global _auth, _auth_loaded
    if time.monotonic()-_auth_loaded < 240 and _auth:
        return _auth
    with _engine.connect() as connection:
        connection.execute(text("SET SESSION MAX_EXECUTION_TIME=10000"))
        connection.execute(text("START TRANSACTION READ ONLY"))
        try:
            rows = connection.execute(text("""
                WITH latest AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY user_id,advertiser_id,COALESCE(NULLIF(app_key,''),'default')
                        ORDER BY id DESC) AS sequence_id FROM qcadvertiserauth
                )
                SELECT p.id,p.user_id,p.advertiser_id,
                    COALESCE(NULLIF(p.app_key,''),'default') AS app_key,
                    COALESCE(a.access_token,p.access_token) AS access_token,
                    COALESCE(a.token_expire_at,p.token_expire_at) AS token_expire_at,
                    COALESCE(a.refresh_token_expire_at,p.refresh_token_expire_at) AS refresh_token_expire_at
                FROM accountplan p LEFT JOIN latest a ON a.sequence_id=1
                    AND a.user_id=p.user_id AND a.advertiser_id=p.advertiser_id
                    AND COALESCE(NULLIF(a.app_key,''),'default')=COALESCE(NULLIF(p.app_key,''),'default')
            """)).mappings().all()
            _auth = {int(row["id"]):dict(row) for row in rows}
            _auth_loaded = time.monotonic()
        finally:
            connection.rollback()
    return _auth


def request_account_video(candidate: dict, material_id: int) -> dict:
    """共享账户权限失败，避免同账户在并发素材上反复请求。"""
    identity = json.dumps([str(candidate["user_id"]),str(candidate["advertiser_id"]),
                           candidate["app_key"]],ensure_ascii=False)
    key = hashlib.sha256(identity.encode()).hexdigest()
    fingerprint = hashlib.sha256(candidate["access_token"].encode()).hexdigest()
    directory = Path(_settings["runtime"])/"account_access"
    directory.mkdir(parents=True,exist_ok=True)
    state_path = directory/(key+".json")
    # 锁覆盖首次探测，确保多个进程不会同时发现同一账户失效。
    with (directory/(key+".lock")).open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
        except (ValueError,OSError):
            state = {}
        if state.get("fingerprint")==fingerprint and state.get("retry_at",0)>time.time():
            return {"error":"账户无权限","account_cache_hit":True}
        response = _request_video(candidate["access_token"],str(candidate["advertiser_id"]),
                                  material_id,candidate["app_key"])
        if response.get("error")=="账户无权限":
            atomic_json(state_path,{"fingerprint":fingerprint,"retry_at":time.time()+900})
            log_warn("账户权限暂不可用，15分钟内跳过重复请求；账户标识：{}".format(key[:12]))
        elif state:
            atomic_json(state_path,{})
        return response


def collect_one(target: dict) -> dict:
    """获取一条素材并在当前素材结束时清理临时视频。"""
    global _extractor
    started = time.monotonic()
    summary = {"asset_id":target["asset_id"],"material_id":target["material_id"]}
    content_lock = None
    try:
        if shutil.disk_usage(_settings["runtime"]).free < 25*1024**3:
            return {**summary,"status":"disk_low"}
        auth = load_auth()
        detail = None
        errors = []
        seen = set()
        for plan_id in target["plan_ids"]:
            candidate = auth.get(int(plan_id))
            if not candidate:
                continue
            key = (candidate["user_id"],candidate["advertiser_id"],candidate["app_key"])
            if key in seen:
                continue
            seen.add(key)
            status = _auth_status(candidate)
            if status != "ready":
                summary["local_auth_skips"] = summary.get("local_auth_skips",0)+1
                errors.append("授权暂不可用")
                continue
            for attempt in range(3):
                response = request_account_video(candidate,int(target["material_id"]))
                if response.get("account_cache_hit"):
                    summary["account_cache_skips"] = summary.get("account_cache_skips",0)+1
                if response.get("request_id"):
                    summary.setdefault("external_request_ids",[]).append(response["request_id"])
                error = str(response.get("error") or "")
                if "限流" not in error and "接口临时异常" not in error:
                    break
                summary["api_retry_count"] = summary.get("api_retry_count",0)+1
                if attempt<2:
                    time.sleep(2**(attempt+1))
            possible = response.get("detail")
            if possible and _valid_video_url(possible.get("url")):
                detail = possible
                break
            errors.append(str(response.get("error") or "素材地址不可用"))
        if detail is None:
            return {**summary,"status":"url_unavailable","reason":"；".join(sorted(set(errors))) or "无关联授权"}
        store = LocalRawFeatureStore(_settings["raw_root"], VERSION)
        with tempfile.TemporaryDirectory(prefix="refresh_asset_",dir=_settings["temp_root"]) as directory:
            video = os.path.join(directory,"source.mp4")
            downloaded = download_video(str(detail["url"]),video,max_video_bytes=500*1024**2)
            content_hash = downloaded["content_sha256"]
            # 同一视频可能关联多个素材ID，按内容加锁避免并发覆盖帧目录。
            content_lock = open(os.path.join(_settings["slots"],content_hash+".lock"),"a+")
            fcntl.flock(content_lock.fileno(),fcntl.LOCK_EX)
            if store.has_content(content_hash):
                store.save_asset_reference(target,content_hash)
                return {**summary,"status":"ready","content_sha256":content_hash,"reused":True}
            if _extractor is None:
                from material_quality.feature.semantic_features import SemanticFeatureExtractor
                _extractor = SemanticFeatureExtractor(
                    "RN50",_settings["clip_cache"],1,use_modelscope=True,
                    asr_model_name="small",asr_model_cache=_settings["asr_cache"],
                    asr_compute_type="int8",asr_cpu_threads=1,ocr_cpu_threads=1,
                    ocr_worker_count=1,asr_num_workers=1,asr_split_min_seconds=45,
                    parallel_slot_dir=_settings["slots"],asr_extra_slots=1)
            features = _extractor.extract(video,directory,keyframe_count=48,frame_long_side=960)
            features["visual_model_name"] = "RN50"
            features["extractor_config"] = {
                "keyframe_count":48,"frame_long_side":960,
                "frame_sampling_strategy":"opening_scene_timeline_v1",
                "opening_seconds":5.0,"opening_interval_seconds":0.5,
                "timeline_interval_seconds":2.0,"scene_threshold":0.30,"max_scene_changes":12,
                "visual_model_name":"RN50","asr_model_name":"small","asr_compute_type":"int8",
                "torch_threads":1,"ocr_cpu_threads":1,"ocr_worker_count":1,
                "asr_cpu_threads":1,"asr_num_workers":1,"asr_split_min_seconds":45,
                "asr_extra_slots":1,
            }
            store.save(target,content_hash,features,source_video_path=video)
        return {**summary,"status":"ready","content_sha256":content_hash,"reused":False,
                "seconds":round(time.monotonic()-started,2)}
    except Exception as error:
        # 网络异常可能携带签名地址，只记录异常类型，详细原因由阶段统计定位。
        return {**summary,"status":"failed","reason":"素材处理异常："+type(error).__name__}
    finally:
        if content_lock is not None:
            content_lock.close()


def stratified_targets(targets: List[dict]) -> List[dict]:
    """在月份和行业间轮转，组内用内容无关哈希排序，保留自然标签分布。"""
    groups = defaultdict(list)
    for target in targets:
        groups[(target["first_stat_date"][:7],target["primary_industry"])].append(target)
    queues = []
    for key in sorted(groups,reverse=True):
        queues.append(deque(sorted(groups[key],key=lambda item:hashlib.sha256(
            item["material_id"].encode()).hexdigest())))
    result = []
    while queues:
        for queue in queues:
            result.append(queue.popleft())
        queues = [queue for queue in queues if queue]
    return result


def run(args: argparse.Namespace) -> None:
    """完成小批采集后自动扩大并发，进度和逐条结果均可恢复。"""
    root = Path(args.runtime)
    source = root / "source_manifest.json"
    while not source.exists() or not json.loads(source.read_text()).get("complete"):
        if (root/"STOP").exists():
            log_warn("收到停止请求，结束等待样本导出")
            return
        progress = root/"status.json"
        if progress.exists() and json.loads(progress.read_text()).get("phase")=="failed":
            raise RuntimeError("上游数据导出失败，采集队列已停止等待")
        time.sleep(10)
    all_targets = [json.loads(line) for line in (root/"candidates.jsonl").read_text().splitlines()]
    targets = [target for target in all_targets if target["label_status"]=="ready"]
    raw_root = Path(args.artifact_root)/"raw_features"
    store = LocalRawFeatureStore(str(raw_root),VERSION)
    references = {}
    for path in (raw_root/VERSION/"assets").glob("*.json"):
        value = json.loads(path.read_text())
        references[value["material_id"]] = value
    by_material = {target["material_id"]:target for target in all_targets}
    unique_hashes = set()
    refreshed = 0
    for mid,reference in references.items():
        if mid in by_material:
            target = by_material[mid]
            target["asset_id"] = reference["asset_id"]
            store.save_asset_reference(target,reference["content_sha256"])
            reference = {**reference,**target}
            refreshed += 1
        if reference["label_status"]=="ready":
            unique_hashes.add(reference["content_sha256"])
    pending = stratified_targets([target for target in targets if target["material_id"] not in references])
    done = set()
    counters = Counter()
    account_cache_skips = 0
    local_auth_skips = 0
    results_path = root/"collection_results.jsonl"
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            result = json.loads(line)
            done.add(result["material_id"])
            counters[result["status"]] += 1
            account_cache_skips += result.get("account_cache_skips",0)
            local_auth_skips += result.get("local_auth_skips",0)
            if result["status"]=="ready":
                unique_hashes.add(result["content_sha256"])
    pending = [target for target in pending if target["material_id"] not in done]
    settings = {"config":args.config,"runtime":str(root),"raw_root":str(raw_root),
                "temp_root":str(root/"temp"),"clip_cache":str(Path(args.artifact_root)/"models/chinese_clip"),
                "asr_cache":str(Path(args.artifact_root)/"models/faster_whisper"),
                "slots":str(root/"slots"),"rate_limit_file":str(root/"api_rate_limit.json")}
    for key in ("temp_root","slots"):
        Path(settings[key]).mkdir(parents=True,exist_ok=True)
    log_info("标签已刷新{}条，已有唯一成熟内容{}条，待采集{}条".format(refreshed,len(unique_hashes),len(pending)))
    phases = [("pilot",pending[:200],min(args.workers,4)),("expanded",pending[200:],args.workers)]
    if len(done)>=200 and counters["ready"]>=10 and counters["failed"]<=counters["ready"]:
        phases = [("expanded",pending,args.workers)]
    for phase,phase_targets,workers in phases:
        if len(unique_hashes)>=args.target_count:
            break
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(settings,)) as pool:
            active = {}
            iterator = iter(phase_targets)
            exhausted = False
            while active or not exhausted:
                stopped = (root/"STOP").exists() or counters["disk_low"]>0
                while not exhausted and not stopped and len(active)<workers and len(unique_hashes)<args.target_count:
                    target = next(iterator,None)
                    if target is None:
                        exhausted = True
                    else:
                        active[pool.submit(collect_one,target)] = target
                if not active:
                    break
                finished,_ = wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for future in finished:
                    target = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {"material_id":target["material_id"],"status":"failed",
                                  "reason":"进程异常："+type(error).__name__}
                    counters[result["status"]] += 1
                    account_cache_skips += result.get("account_cache_skips",0)
                    local_auth_skips += result.get("local_auth_skips",0)
                    if result["status"]=="ready":
                        unique_hashes.add(result["content_sha256"])
                    with results_path.open("a") as stream:
                        stream.write(json.dumps(result,ensure_ascii=False)+"\n")
                    log_info("素材任务结束：{}，状态{}，原因{}".format(
                        result["material_id"],result["status"],result.get("reason","")))
                    if result.get("external_request_ids"):
                        log_info("素材{}外部请求标识：{}".format(
                            result["material_id"],"，".join(result["external_request_ids"])))
                atomic_json(root/"collection_status.json",{
                    "phase":phase,"workers":workers,"counts":dict(counters),
                    "account_cache_skips":account_cache_skips,"local_auth_skips":local_auth_skips,
                    "unique_ready":len(unique_hashes),"target_count":args.target_count,
                    "pending_in_process":len(active),"updated_at":datetime.datetime.now().isoformat()})
        if phase=="pilot" and counters["ready"]<10:
            log_warn("小批采集成功不足10条，暂停扩大并发，请检查授权和素材可获取率")
            atomic_json(root/"collection_status.json",{"phase":"pilot_needs_attention","counts":dict(counters),
                        "unique_ready":len(unique_hashes)})
            return
        if phase=="pilot" and counters["failed"]>max(counters["ready"],1):
            raise RuntimeError("小批特征处理异常较多，停止扩大采集")
    atomic_json(root/"collection_status.json",{"phase":"finished","counts":dict(counters),
                "account_cache_skips":account_cache_skips,"local_auth_skips":local_auth_skips,
                "unique_ready":len(unique_hashes),"target_count":args.target_count})


def main() -> None:
    """启动独立素材采集队列。"""
    set_process_name("material_quality_collection")
    parser = argparse.ArgumentParser(description="采集最新训练素材特征")
    parser.add_argument("--runtime",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--artifact-root",required=True)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--target-count",type=int,default=10000)
    args = parser.parse_args()
    if not 1<=args.workers<=8:
        raise ValueError("特征采集并发必须在1至8之间")
    run(args)


if __name__=="__main__":
    main()

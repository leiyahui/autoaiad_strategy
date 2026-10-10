"""本地离线任务使用的只读千川素材地址解析。"""

import json
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from plan_logger import log_info, log_warn


VIDEO_GET_URL = "https://ad.oceanengine.com/open_api/v1.0/qianchuan/video/get/"
INVALID_URL_MARKERS = ("素材所属主体与开发者主体不一致", "无法获取URL")
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TIMES: Dict[str, float] = {}


def _load_plan_candidates(conn, cohort: str, material_id: str) -> List[dict]:
    """按历史GMV贡献排序素材实际投放过的广告主。"""
    from sqlalchemy import text

    rows = conn.execute(text("""
        SELECT
            p.user_id,
            p.advertiser_id,
            COALESCE(NULLIF(p.app_key, ''), 'default') AS app_key,
            p.access_token,
            p.refresh_token,
            p.token_expire_at,
            p.refresh_token_expire_at,
            SUM(ABS(d.gmv)) AS priority_gmv,
            MAX(d.stat_date) AS latest_stat_date
        FROM material_quality_daily_stat AS d
        INNER JOIN accountplan AS p ON p.id = d.account_plan_id
        WHERE d.cohort = :cohort AND d.material_id = :material_id
        GROUP BY
            p.user_id, p.advertiser_id, p.app_key, p.access_token,
            p.refresh_token, p.token_expire_at, p.refresh_token_expire_at
        ORDER BY priority_gmv DESC, latest_stat_date DESC, p.token_expire_at DESC
    """), {"cohort": cohort, "material_id": material_id}).mappings().all()
    seen = set()
    candidates = []
    for row in rows:
        candidate = dict(row)
        key = (
            int(candidate["user_id"]),
            str(candidate["advertiser_id"]),
            str(candidate.get("app_key") or "default"),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def _load_latest_user_auth(conn, candidate: dict) -> Optional[dict]:
    """读取授权表中的最新Token，但不在本地执行刷新或写授权表。"""
    from sqlalchemy import text

    row = conn.execute(text("""
        SELECT
            user_id, advertiser_id,
            COALESCE(NULLIF(app_key, ''), 'default') AS app_key,
            access_token, refresh_token,
            token_expire_at, refresh_token_expire_at
        FROM qcadvertiserauth
        WHERE user_id = :user_id
          AND advertiser_id = :advertiser_id
          AND COALESCE(NULLIF(app_key, ''), 'default') = :app_key
        ORDER BY id DESC
        LIMIT 1
    """), {
        "user_id": candidate["user_id"],
        "advertiser_id": candidate["advertiser_id"],
        "app_key": candidate.get("app_key") or "default",
    }).mappings().first()
    return dict(row) if row else None


def _auth_status(candidate: dict, now: Optional[int] = None) -> str:
    """区分账户授权硬过期和本地暂不可用的访问Token。"""
    resolved_now = int(time.time()) if now is None else int(now)
    refresh_token_expire_at = int(candidate.get("refresh_token_expire_at") or 0)
    token_expire_at = int(candidate.get("token_expire_at") or 0)
    if refresh_token_expire_at > 0 and refresh_token_expire_at <= resolved_now:
        return "auth_expired"
    if not str(candidate.get("access_token") or "").strip():
        return "access_token_missing"
    if token_expire_at <= resolved_now + 300:
        return "token_expiring"
    return "ready"


def _wait_request_interval(app_key: str) -> None:
    """按应用跨进程限制接口节奏，避免提高并发后叠加请求频率。"""
    interval = max(
        float(os.getenv("MATERIAL_QUALITY_URL_INTERVAL_SECONDS", "1.0")),
        0.0,
    )
    normalized_app_key = str(app_key or "default")
    state_path = os.getenv("MATERIAL_QUALITY_RATE_LIMIT_FILE", "").strip()
    if state_path:
        import fcntl

        resolved_path = os.path.realpath(state_path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, "a+", encoding="utf-8") as state_file:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
            try:
                state_file.seek(0)
                try:
                    state = json.load(state_file)
                except (ValueError, TypeError):
                    state = {}
                current_time = time.time()
                last_request_time = float(state.get(normalized_app_key) or 0.0)
                wait_seconds = interval - (current_time - last_request_time)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                state[normalized_app_key] = time.time()
                state_file.seek(0)
                state_file.truncate()
                json.dump(state, state_file, ensure_ascii=False, sort_keys=True)
                state_file.flush()
                os.fsync(state_file.fileno())
            finally:
                fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)
        return

    with _REQUEST_LOCK:
        last_request_time = _LAST_REQUEST_TIMES.get(normalized_app_key, 0.0)
        wait_seconds = interval - (time.monotonic() - last_request_time)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _LAST_REQUEST_TIMES[normalized_app_key] = time.monotonic()


def _request_video(
    access_token: str,
    advertiser_id: str,
    material_id: int,
    app_key: str,
) -> dict:
    """直接调用素材查询接口，只返回必要结果和错误分类。"""
    import requests

    _wait_request_interval(app_key)
    try:
        response = requests.get(
            VIDEO_GET_URL,
            headers={"Access-Token": access_token},
            params={
                "advertiser_id": advertiser_id,
                "page": 1,
                "page_size": 1,
                "filtering": json.dumps(
                    {"material_ids": [material_id]},
                    ensure_ascii=False,
                ),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return {"error": "接口临时异常request_timeout"}
    except (requests.RequestException, ValueError) as exc:
        return {"error": "接口临时异常{}".format(type(exc).__name__)}
    error_code = str(payload.get("code") or "")
    if error_code not in ("", "0"):
        if error_code in ("40100", "40110", "40130"):
            return {"error": "接口临时限流"}
        if error_code in ("40002", "40003"):
            return {"error": "账户无权限"}
        return {"error": "接口临时异常{}".format(error_code)}
    video_list = (payload.get("data") or {}).get("list") or []
    if not video_list:
        return {"error": "未返回素材"}
    return {"detail": dict(video_list[0])}


def _valid_video_url(value: object) -> bool:
    """过滤素材接口URL字段中的业务错误文本。"""
    url = str(value or "").strip()
    if not url.startswith(("http://", "https://")):
        return False
    return not any(marker in url for marker in INVALID_URL_MARKERS)


def preload_material_auth_candidates(
    conn,
    cohort: str,
    material_ids: Sequence[str],
) -> Dict[str, object]:
    """批量读取候选账户和最新授权，在领取素材前完成账户级过滤。"""
    from sqlalchemy import bindparam, text

    normalized_ids = sorted({str(value) for value in material_ids if str(value)})
    if not normalized_ids:
        return {
            "material_candidates": {},
            "auth_cache": {},
            "usable_auth_cache": {},
            "deferred_errors": {},
            "account_status_counts": {},
        }
    query = text("""
        WITH latest_authorization AS (
            SELECT
                user_id, advertiser_id,
                COALESCE(NULLIF(app_key, ''), 'default') AS app_key,
                access_token, refresh_token,
                token_expire_at, refresh_token_expire_at,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        user_id,
                        advertiser_id,
                        COALESCE(NULLIF(app_key, ''), 'default')
                    ORDER BY id DESC
                ) AS auth_sequence
            FROM qcadvertiserauth
        )
        SELECT
            d.material_id,
            p.user_id,
            p.advertiser_id,
            COALESCE(NULLIF(p.app_key, ''), 'default') AS app_key,
            COALESCE(a.access_token, p.access_token) AS access_token,
            COALESCE(a.refresh_token, p.refresh_token) AS refresh_token,
            COALESCE(a.token_expire_at, p.token_expire_at) AS token_expire_at,
            COALESCE(
                a.refresh_token_expire_at,
                p.refresh_token_expire_at
            ) AS refresh_token_expire_at,
            SUM(ABS(d.gmv)) AS priority_gmv,
            MAX(d.stat_date) AS latest_stat_date
        FROM material_quality_daily_stat AS d
        INNER JOIN accountplan AS p ON p.id = d.account_plan_id
        LEFT JOIN latest_authorization AS a
          ON a.user_id = p.user_id
         AND a.advertiser_id = p.advertiser_id
         AND a.app_key = COALESCE(NULLIF(p.app_key, ''), 'default')
         AND a.auth_sequence = 1
        WHERE d.cohort = :cohort
          AND d.material_id IN :material_ids
        GROUP BY
            d.material_id,
            p.user_id,
            p.advertiser_id,
            COALESCE(NULLIF(p.app_key, ''), 'default'),
            COALESCE(a.access_token, p.access_token),
            COALESCE(a.refresh_token, p.refresh_token),
            COALESCE(a.token_expire_at, p.token_expire_at),
            COALESCE(a.refresh_token_expire_at, p.refresh_token_expire_at)
        ORDER BY
            d.material_id,
            priority_gmv DESC,
            latest_stat_date DESC,
            token_expire_at DESC
    """).bindparams(bindparam("material_ids", expanding=True))
    rows = conn.execute(query, {
        "cohort": cohort,
        "material_ids": normalized_ids,
    }).mappings().all()

    material_candidates: Dict[str, List[dict]] = {
        material_id: [] for material_id in normalized_ids
    }
    auth_cache: Dict[Tuple[int, str, str], dict] = {}
    usable_auth_cache: Dict[
        Tuple[int, str, str], Tuple[Optional[dict], str]
    ] = {}
    seen_material_accounts = set()
    for row in rows:
        candidate = dict(row)
        material_id = str(candidate.pop("material_id"))
        cache_key = (
            int(candidate["user_id"]),
            str(candidate["advertiser_id"]),
            str(candidate.get("app_key") or "default"),
        )
        material_account_key = (material_id,) + cache_key
        if material_account_key in seen_material_accounts:
            continue
        seen_material_accounts.add(material_account_key)
        material_candidates.setdefault(material_id, []).append(candidate)
        if cache_key not in auth_cache:
            auth_cache[cache_key] = candidate
            status = _auth_status(candidate)
            usable_auth_cache[cache_key] = (
                candidate if status == "ready" else None,
                status,
            )

    account_status_counts: Dict[str, int] = {}
    for _, status in usable_auth_cache.values():
        account_status_counts[status] = account_status_counts.get(status, 0) + 1
    deferred_errors = {}
    for material_id in normalized_ids:
        candidates = material_candidates.get(material_id, [])
        statuses = []
        for candidate in candidates:
            cache_key = (
                int(candidate["user_id"]),
                str(candidate["advertiser_id"]),
                str(candidate.get("app_key") or "default"),
            )
            _, status = usable_auth_cache[cache_key]
            statuses.append("广告主{}:{}".format(candidate["advertiser_id"], status))
        if not candidates:
            deferred_errors[material_id] = "素材没有可关联的投放计划"
        elif not any(
            usable_auth_cache[(
                int(candidate["user_id"]),
                str(candidate["advertiser_id"]),
                str(candidate.get("app_key") or "default"),
            )][0]
            for candidate in candidates
        ):
            deferred_errors[material_id] = "；".join(statuses[-5:])
    return {
        "material_candidates": material_candidates,
        "auth_cache": auth_cache,
        "usable_auth_cache": usable_auth_cache,
        "deferred_errors": deferred_errors,
        "account_status_counts": account_status_counts,
    }


def resolve_material_url_local(
    conn,
    cohort: str,
    material_id: str,
    auth_cache: Optional[Dict[Tuple[int, str, str], dict]] = None,
    usable_auth_cache: Optional[
        Dict[Tuple[int, str, str], Tuple[Optional[dict], str]]
    ] = None,
    material_candidates_cache: Optional[Dict[str, List[dict]]] = None,
) -> Tuple[Optional[Dict[str, object]], str]:
    """本地只使用仍有效的Token解析URL，不刷新和修改授权记录。"""
    try:
        numeric_material_id = int(material_id)
    except (TypeError, ValueError):
        return None, "素材ID不是纯数字，无法调用千川素材接口"

    candidates = (
        material_candidates_cache.get(str(material_id), [])
        if material_candidates_cache is not None
        else _load_plan_candidates(conn, cohort, material_id)
    )
    if not candidates:
        return None, "素材没有可关联的投放计划"
    errors = []
    for plan_candidate in candidates:
        cache_key = (
            int(plan_candidate["user_id"]),
            str(plan_candidate["advertiser_id"]),
            str(plan_candidate.get("app_key") or "default"),
        )
        if auth_cache is not None and cache_key in auth_cache:
            candidate = auth_cache[cache_key]
        else:
            candidate = _load_latest_user_auth(conn, plan_candidate) or plan_candidate
            if auth_cache is not None:
                auth_cache[cache_key] = candidate

        cached_auth = usable_auth_cache.get(cache_key) if usable_auth_cache is not None else None
        if cached_auth is None:
            status = _auth_status(candidate)
            usable_auth = candidate if status == "ready" else None
            cached_auth = (usable_auth, status)
            if usable_auth_cache is not None:
                usable_auth_cache[cache_key] = cached_auth
        usable_auth, status = cached_auth
        advertiser_id = str(candidate["advertiser_id"])
        app_key = str(candidate.get("app_key") or "default")
        if not usable_auth:
            errors.append("广告主{}:{}".format(advertiser_id, status))
            continue

        result = _request_video(
            str(usable_auth["access_token"]),
            advertiser_id,
            numeric_material_id,
            app_key,
        )
        detail = result.get("detail")
        if not detail:
            result_error = str(result.get("error") or "未返回素材")
            if result_error == "账户无权限" and usable_auth_cache is not None:
                # 权限错误属于账户维度，当前批次后续素材无需继续调用接口。
                usable_auth_cache[cache_key] = (None, "account_permission_denied")
            errors.append("广告主{}:{}".format(
                advertiser_id,
                result_error,
            ))
            continue
        if not _valid_video_url(detail.get("url")):
            errors.append("广告主{}:视频地址不可用".format(advertiser_id))
            continue
        detail["user_id"] = usable_auth.get("user_id")
        detail["advertiser_id"] = advertiser_id
        detail["app_key"] = str(usable_auth.get("app_key") or "default")
        log_info("素材{}在本地刷新视频地址成功，广告主{}".format(
            material_id,
            advertiser_id,
        ))
        return detail, ""

    error_summary = "；".join(errors[-5:]) or "所有候选广告主均未返回可用地址"
    log_warn("素材{}在本地刷新视频地址失败：{}".format(
        material_id,
        error_summary,
    ))
    return None, error_summary

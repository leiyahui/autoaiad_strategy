"""根据素材实际投放账户刷新千川视频临时地址。"""

import time
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from chuan_api import get_video_materials
from chuan_api.rate_limiter import rate_limit_app_context
from plan_logger import log_info, log_warn
from utils.token_refresh import check_auth_status, refresh_token_threadsafe


INVALID_URL_MARKERS = ("素材所属主体与开发者主体不一致", "无法获取URL")


def _load_plan_candidates(conn, cohort: str, material_id: str) -> List[dict]:
    """按素材在计划中的GMV贡献排序候选广告主。"""
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
    """优先读取用户授权表中的最新过期时间和Token。"""
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


def _get_usable_auth(candidate: dict) -> Tuple[Optional[dict], str]:
    """检查数据库过期时间，必要时复用统一刷新逻辑获取有效Token。"""
    is_valid, status = check_auth_status(
        candidate.get("refresh_token_expire_at"),
        candidate.get("token_expire_at"),
    )
    if status in ("auth_expired", "auth_expired_no_field"):
        return None, "auth_expired"
    if is_valid and status != "token_expiring":
        return candidate, ""

    success, error_type, token_info = refresh_token_threadsafe(
        str(candidate["advertiser_id"]),
        candidate,
    )
    if not success or not token_info:
        return None, error_type or "token_refresh_failed"
    refreshed = dict(candidate)
    refreshed.update(token_info)
    return refreshed, ""


def _valid_video_url(value: object) -> bool:
    """过滤千川用URL字段返回的业务错误文本。"""
    url = str(value or "").strip()
    if not url.startswith(("http://", "https://")):
        return False
    return not any(marker in url for marker in INVALID_URL_MARKERS)


def resolve_material_url(
    conn,
    cohort: str,
    material_id: str,
    auth_cache: Optional[Dict[Tuple[int, str, str], dict]] = None,
    usable_auth_cache: Optional[
        Dict[Tuple[int, str, str], Tuple[Optional[dict], str]]
    ] = None,
) -> Tuple[Optional[Dict[str, object]], str]:
    """依次尝试素材实际投放过的广告主，返回最新可下载地址。"""
    try:
        numeric_material_id = int(material_id)
    except (TypeError, ValueError):
        return None, "素材ID不是纯数字，无法调用千川素材接口"

    candidates = _load_plan_candidates(conn, cohort, material_id)
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
        cache_needs_refresh = bool(
            cached_auth
            and cached_auth[0]
            and int(cached_auth[0].get("token_expire_at") or 0) - int(time.time()) < 300
        )
        if cached_auth is not None and not cache_needs_refresh:
            usable_auth, error_type = cached_auth
        else:
            usable_auth, error_type = _get_usable_auth(candidate)
            if usable_auth_cache is not None:
                usable_auth_cache[cache_key] = (usable_auth, error_type)

        advertiser_id = str(candidate["advertiser_id"])
        if not usable_auth:
            errors.append("广告主{}:{}".format(advertiser_id, error_type))
            continue

        app_key = str(usable_auth.get("app_key") or "default")
        try:
            with rate_limit_app_context(app_key):
                video_list, page_info = get_video_materials(
                    access_token=str(usable_auth["access_token"]),
                    advertiser_id=advertiser_id,
                    material_ids=[numeric_material_id],
                    page=1,
                    page_size=1,
                )
        except Exception as exc:
            errors.append("广告主{}:接口异常{}".format(advertiser_id, exc))
            continue
        if not video_list:
            error_code = str(page_info.get("_error_code") or "")
            if error_code in ("40100", "40110", "40130", "local_rate_limit"):
                errors.append("广告主{}:接口临时限流".format(advertiser_id))
            elif error_code in ("40001", "request_timeout", "request_error", "unknown_error"):
                errors.append("广告主{}:接口临时异常{}".format(advertiser_id, error_code))
            elif error_code in ("40002", "40003"):
                errors.append("广告主{}:账户无权限".format(advertiser_id))
            elif error_code:
                errors.append("广告主{}:接口临时异常{}".format(advertiser_id, error_code))
            else:
                errors.append("广告主{}:未返回素材".format(advertiser_id))
            continue
        detail = dict(video_list[0])
        if not _valid_video_url(detail.get("url")):
            errors.append("广告主{}:视频地址不可用".format(advertiser_id))
            continue

        detail["user_id"] = usable_auth.get("user_id")
        detail["advertiser_id"] = advertiser_id
        detail["app_key"] = app_key
        log_info("素材{}刷新视频地址成功，广告主{}".format(material_id, advertiser_id))
        return detail, ""

    error_summary = "；".join(errors[-5:]) or "所有候选广告主均未返回可用地址"
    log_warn("素材{}刷新视频地址失败：{}".format(material_id, error_summary))
    return None, error_summary

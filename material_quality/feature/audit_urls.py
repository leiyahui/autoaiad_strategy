#!/usr/bin/env python3
"""随机抽样审计素材视频地址的实时可获取率。"""

import argparse
from typing import Dict, List

from sqlalchemy import text

from plan_logger import log_info, set_process_name

from material_quality.database import engine
from material_quality.feature.url_resolver import resolve_material_url


DEFAULT_COHORT = "food_v1"


def _load_targets(cohort: str, sample_size: int, sample_seed: str) -> List[dict]:
    """使用素材ID哈希做可重复的近似随机抽样，避免数据库全表随机排序。"""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT material_id, label_status, label_gmv_7d
            FROM material_quality_sample
            WHERE cohort = :cohort
            ORDER BY CRC32(CONCAT(material_id, :sample_seed))
            LIMIT :sample_size
        """), {
            "cohort": cohort,
            "sample_seed": sample_seed,
            "sample_size": sample_size,
        }).mappings().all()
    return [dict(row) for row in rows]


def _classify_error(error: str) -> str:
    """把详细失败原因收敛为便于统计的类别。"""
    normalized = str(error or "")
    if "auth_expired" in normalized:
        return "授权已过期"
    if "视频地址不可用" in normalized:
        return "主体不一致或地址不可用"
    if "不是纯数字" in normalized:
        return "素材ID格式异常"
    if "没有可关联" in normalized:
        return "缺少投放账户"
    if "未返回素材" in normalized:
        return "接口未返回或账户无权限"
    if "账户无权限" in normalized:
        return "账户无权限"
    if "接口临时限流" in normalized:
        return "接口临时限流"
    if "接口异常" in normalized or "token_refresh_failed" in normalized:
        return "接口或Token刷新异常"
    return "其他失败"


def run_audit(cohort: str, sample_size: int, sample_seed: str) -> dict:
    """逐条实时刷新随机样本，只统计可获取率，不下载或保存视频。"""
    targets = _load_targets(cohort, sample_size, sample_seed)
    success_count = 0
    failure_counts: Dict[str, int] = {}
    auth_cache = {}
    usable_auth_cache = {}
    for index, target in enumerate(targets, start=1):
        with engine.connect() as conn:
            detail, error = resolve_material_url(
                conn,
                cohort,
                str(target["material_id"]),
                auth_cache=auth_cache,
                usable_auth_cache=usable_auth_cache,
            )
        if detail:
            success_count += 1
        else:
            category = _classify_error(error)
            failure_counts[category] = failure_counts.get(category, 0) + 1
        if index % 20 == 0 or index == len(targets):
            log_info(
                "地址可获取率审计进度：{}/{}，当前成功{}".format(
                    index,
                    len(targets),
                    success_count,
                )
            )

    total_count = len(targets)
    success_rate = (
        float(success_count) / float(total_count) if total_count else 0.0
    )
    result = {
        "sample_count": total_count,
        "success_count": success_count,
        "failure_count": total_count - success_count,
        "success_rate": success_rate,
        "failure_counts": failure_counts,
        "sample_seed": sample_seed,
        "checked_auth_count": len(usable_auth_cache),
    }
    log_info("地址可获取率审计完成：{}".format(result))
    return result


def parse_args() -> argparse.Namespace:
    """读取审计命令行参数。"""
    parser = argparse.ArgumentParser(description="随机抽样审计素材视频地址可获取率")
    parser.add_argument("--cohort", default=DEFAULT_COHORT, help="样本范围版本")
    parser.add_argument("--sample-size", type=int, default=200, help="随机抽样数量")
    parser.add_argument("--sample-seed", default="url_audit_v1", help="可重复抽样种子")
    return parser.parse_args()


def main() -> int:
    """审计任务入口。"""
    set_process_name("material_quality_url_audit")
    args = parse_args()
    if args.sample_size <= 0:
        log_info("随机抽样数量必须大于0")
        return 2
    result = run_audit(args.cohort, args.sample_size, args.sample_seed)
    return 0 if result["sample_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""素材特征候选范围的纯计算工具。"""

import argparse
import datetime
from typing import Dict, Optional, Tuple


def parse_date(value: str) -> datetime.date:
    """解析命令行中的自然日，避免日期范围静默失效。"""
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("日期必须使用YYYY-MM-DD格式")


def build_first_date_filter(
    first_date_from: Optional[datetime.date],
    first_date_to: Optional[datetime.date],
) -> Tuple[str, Dict[str, object]]:
    """构造参数化日期条件，禁止把命令行值直接拼入SQL。"""
    conditions = []
    params: Dict[str, object] = {}
    if first_date_from:
        conditions.append("s.first_stat_date >= :first_date_from")
        params["first_date_from"] = first_date_from
    if first_date_to:
        conditions.append("s.first_stat_date <= :first_date_to")
        params["first_date_to"] = first_date_to
    if not conditions:
        return "", params
    return " AND " + " AND ".join(conditions), params

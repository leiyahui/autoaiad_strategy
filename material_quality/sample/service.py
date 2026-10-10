"""素材质量样本构造的纯数据逻辑。"""

import datetime
import math
from typing import Dict, Iterable, List, Optional


DEFAULT_FOOD_INDUSTRIES = {
    "生活食材",
    "休闲零食",
    "休闲食品",
    "饮料及冲调",
    "水饮冲调",
    "粮油干货/方便速食",
    "水果蔬菜",
    "生鲜",
    "烟酒",
    "传统滋补",
    "营养保健/特医食品",
}

FOOD_INDUSTRY_KEYWORDS = (
    "食材",
    "零食",
    "食品",
    "饮料",
    "冲调",
    "粮油",
    "干货",
    "速食",
    "水果",
    "蔬菜",
    "生鲜",
    "烟酒",
    "滋补",
)


def validate_cohort_scope(cohort: str, include_all_industries: bool) -> None:
    """阻止全行业任务误写食品专用样本版本。"""
    normalized = str(cohort or "").strip()
    if not normalized:
        raise ValueError("样本范围版本不能为空")
    if include_all_industries and normalized == "food_v1":
        raise ValueError("全行业任务必须使用独立样本范围版本")


def is_food_industry(industry: Optional[str], exact_industries: Optional[Iterable[str]] = None) -> bool:
    """判断计划行业是否属于第一版食品样本范围。"""
    normalized = str(industry or "").strip()
    if not normalized:
        return False
    allowed = set(exact_industries or DEFAULT_FOOD_INDUSTRIES)
    if normalized in allowed:
        return True
    return any(keyword in normalized for keyword in FOOD_INDUSTRY_KEYWORDS)


def choose_asset_metadata(rows: Iterable[dict]) -> Dict[str, dict]:
    """每个素材选择信息最完整且更新较新的素材库记录。"""
    selected: Dict[str, dict] = {}
    for raw in rows:
        row = dict(raw)
        material_id = str(row.get("material_id") or "").strip()
        if not material_id:
            continue
        score = (
            1 if row.get("video_url") else 0,
            1 if row.get("thumbnail_url") else 0,
            float(row.get("upload_time") or 0),
            int(row.get("id") or 0),
        )
        current = selected.get(material_id)
        if current is None or score > current["_selection_score"]:
            row["material_id"] = material_id
            row["_selection_score"] = score
            selected[material_id] = row
    for row in selected.values():
        row.pop("_selection_score", None)
    return selected


def build_samples_from_daily_rows(
    rows: Iterable[dict],
    as_of_date: datetime.date,
    window_days: int = 7,
) -> List[dict]:
    """按素材首次出现日期聚合前N个自然日，生成一素材一样本。"""
    if window_days <= 0:
        raise ValueError("样本窗口天数必须大于0")

    grouped: Dict[str, List[dict]] = {}
    for raw in rows:
        row = dict(raw)
        material_id = str(row.get("material_id") or "").strip()
        stat_date = row.get("stat_date")
        if not material_id or not isinstance(stat_date, datetime.date):
            continue
        has_activity = any(float(row.get(field) or 0) != 0 for field in (
            "gmv", "cost", "orders", "send", "click"
        ))
        if not has_activity:
            continue
        grouped.setdefault(material_id, []).append(row)

    samples = []
    for material_id, material_rows in grouped.items():
        first_stat_date = min(row["stat_date"] for row in material_rows)
        window_end_date = first_stat_date + datetime.timedelta(days=window_days)
        window_rows = [
            row for row in material_rows
            if first_stat_date <= row["stat_date"] < window_end_date
        ]
        gmv = sum(float(row.get("gmv") or 0) for row in window_rows)
        cost = sum(float(row.get("cost") or 0) for row in window_rows)
        orders = sum(float(row.get("orders") or 0) for row in window_rows)
        send = sum(float(row.get("send") or 0) for row in window_rows)
        click = sum(float(row.get("click") or 0) for row in window_rows)
        plan_ids = {
            int(row["account_plan_id"])
            for row in window_rows
            if row.get("account_plan_id") is not None
        }
        delivery_day_count = len({row["stat_date"] for row in window_rows})
        gmv_per_delivery_day = (
            gmv / float(delivery_day_count) if delivery_day_count > 0 else 0.0
        )
        is_window_closed = as_of_date >= window_end_date
        if not is_window_closed:
            label_status = "window_open"
        elif gmv < 0:
            label_status = "invalid_negative_gmv"
        elif delivery_day_count < 2:
            label_status = "insufficient_delivery_days"
        else:
            label_status = "ready"
        samples.append({
            "material_id": material_id,
            "first_stat_date": first_stat_date,
            "window_end_date": window_end_date,
            "label_gmv_7d": gmv,
            "label_log_gmv_7d": math.log1p(max(gmv, 0.0)),
            "label_gmv_per_delivery_day": gmv_per_delivery_day,
            "label_log_gmv_per_delivery_day": math.log1p(
                max(gmv_per_delivery_day, 0.0)
            ),
            "cost_7d": cost,
            "orders_7d": orders,
            "send_7d": send,
            "click_7d": click,
            "plan_count": len(plan_ids),
            "source_day_count": len(window_rows),
            "delivery_day_count": delivery_day_count,
            "is_window_closed": is_window_closed,
            "label_status": label_status,
        })
    return sorted(samples, key=lambda item: item["material_id"])

"""素材质量样本构造逻辑测试。"""

import datetime
import math
import unittest

from material_quality.sample.refresh_local_labels import effective_label_date
from material_quality.sample.service import (
    build_samples_from_daily_rows,
    choose_asset_metadata,
    is_food_industry,
    validate_cohort_scope,
)


class MaterialQualitySampleServiceTest(unittest.TestCase):
    """验证一素材一样本和7日窗口边界。"""

    def test_food_industry_scope(self):
        self.assertTrue(is_food_industry("生活食材"))
        self.assertTrue(is_food_industry("地方休闲食品"))
        self.assertTrue(is_food_industry("粮油干货/方便速食"))
        self.assertFalse(is_food_industry("服装"))
        self.assertFalse(is_food_industry(None))

    def test_requires_separate_all_industry_cohort(self):
        with self.assertRaisesRegex(ValueError, "独立样本范围"):
            validate_cohort_scope("food_v1", True)

        validate_cohort_scope("all_industries_v1", True)

    def test_caps_local_label_date_at_complete_daily_coverage(self):
        maximum_stat_date = datetime.date(2026, 8, 18)

        resolved = effective_label_date(
            datetime.date(2026, 8, 21),
            maximum_stat_date,
        )

        self.assertEqual(resolved, datetime.date(2026, 8, 19))

    def test_choose_asset_metadata_prefers_usable_url(self):
        rows = [
            {
                "id": 2,
                "material_id": "1001",
                "video_url": None,
                "thumbnail_url": None,
                "upload_time": 200,
            },
            {
                "id": 1,
                "material_id": "1001",
                "video_url": "https://example.com/video.mp4",
                "thumbnail_url": "https://example.com/cover.jpg",
                "upload_time": 100,
            },
            {
                "id": 3,
                "material_id": "1002",
                "video_url": None,
                "thumbnail_url": None,
                "upload_time": 300,
            },
        ]

        result = choose_asset_metadata(rows)

        self.assertEqual(result["1001"]["id"], 1)
        self.assertEqual(result["1002"]["id"], 3)
        self.assertNotIn("_selection_score", result["1001"])

    def test_builds_one_material_sample_across_plans(self):
        first_date = datetime.date(2026, 1, 1)
        rows = [
            {
                "account_plan_id": 10,
                "material_id": "m1",
                "stat_date": first_date,
                "gmv": 100,
                "cost": 20,
                "orders": 1,
                "send": 1000,
                "click": 50,
            },
            {
                "account_plan_id": 11,
                "material_id": "m1",
                "stat_date": first_date + datetime.timedelta(days=6),
                "gmv": 200,
                "cost": 40,
                "orders": 2,
                "send": 2000,
                "click": 80,
            },
            {
                "account_plan_id": 10,
                "material_id": "m1",
                "stat_date": first_date + datetime.timedelta(days=7),
                "gmv": 999,
                "cost": 99,
                "orders": 9,
                "send": 9999,
                "click": 999,
            },
        ]

        result = build_samples_from_daily_rows(
            rows,
            as_of_date=first_date + datetime.timedelta(days=8),
        )

        self.assertEqual(len(result), 1)
        sample = result[0]
        self.assertEqual(sample["label_gmv_7d"], 300)
        self.assertAlmostEqual(sample["label_log_gmv_7d"], math.log1p(300))
        self.assertEqual(sample["delivery_day_count"], 2)
        self.assertEqual(sample["label_gmv_per_delivery_day"], 150)
        self.assertAlmostEqual(
            sample["label_log_gmv_per_delivery_day"],
            math.log1p(150),
        )
        self.assertEqual(sample["cost_7d"], 60)
        self.assertEqual(sample["orders_7d"], 3)
        self.assertEqual(sample["plan_count"], 2)
        self.assertEqual(sample["source_day_count"], 2)
        self.assertTrue(sample["is_window_closed"])
        self.assertEqual(sample["label_status"], "ready")

    def test_ignores_zero_rows_when_selecting_first_date(self):
        rows = [
            {
                "account_plan_id": 1,
                "material_id": "m1",
                "stat_date": datetime.date(2026, 1, 1),
                "gmv": 0,
                "cost": 0,
                "orders": 0,
                "send": 0,
                "click": 0,
            },
            {
                "account_plan_id": 1,
                "material_id": "m1",
                "stat_date": datetime.date(2026, 1, 3),
                "gmv": 50,
                "cost": 10,
                "orders": 1,
                "send": 100,
                "click": 5,
            },
        ]

        result = build_samples_from_daily_rows(
            rows,
            as_of_date=datetime.date(2026, 1, 5),
        )

        self.assertEqual(result[0]["first_stat_date"], datetime.date(2026, 1, 3))
        self.assertFalse(result[0]["is_window_closed"])
        self.assertEqual(result[0]["label_status"], "window_open")

    def test_rejects_invalid_window(self):
        with self.assertRaisesRegex(ValueError, "样本窗口天数必须大于0"):
            build_samples_from_daily_rows([], datetime.date(2026, 1, 1), window_days=0)

    def test_marks_closed_negative_gmv_as_invalid(self):
        rows = [{
            "account_plan_id": 1,
            "material_id": "m1",
            "stat_date": datetime.date(2026, 1, 1),
            "gmv": -10,
            "cost": 20,
            "orders": 0,
            "send": 100,
            "click": 5,
        }]

        result = build_samples_from_daily_rows(
            rows,
            as_of_date=datetime.date(2026, 1, 8),
        )

        self.assertTrue(result[0]["is_window_closed"])
        self.assertEqual(result[0]["label_gmv_7d"], -10)
        self.assertEqual(result[0]["label_log_gmv_7d"], 0)
        self.assertEqual(result[0]["label_status"], "invalid_negative_gmv")

    def test_marks_one_delivery_day_as_insufficient(self):
        first_date = datetime.date(2026, 1, 1)
        rows = [{
            "account_plan_id": 1,
            "material_id": "m1",
            "stat_date": first_date,
            "gmv": 100,
            "cost": 20,
            "orders": 1,
            "send": 100,
            "click": 5,
        }]

        result = build_samples_from_daily_rows(
            rows,
            as_of_date=first_date + datetime.timedelta(days=7),
        )

        self.assertEqual(result[0]["delivery_day_count"], 1)
        self.assertEqual(result[0]["label_gmv_per_delivery_day"], 100)
        self.assertEqual(
            result[0]["label_status"],
            "insufficient_delivery_days",
        )

    def test_counts_delivery_dates_once_across_plans(self):
        first_date = datetime.date(2026, 1, 1)
        rows = [
            {
                "account_plan_id": plan_id,
                "material_id": "m1",
                "stat_date": first_date,
                "gmv": 50,
                "cost": 10,
                "orders": 1,
                "send": 100,
                "click": 5,
            }
            for plan_id in (1, 2)
        ]
        rows.append({
            "account_plan_id": 1,
            "material_id": "m1",
            "stat_date": first_date + datetime.timedelta(days=1),
            "gmv": 20,
            "cost": 5,
            "orders": 1,
            "send": 50,
            "click": 2,
        })

        result = build_samples_from_daily_rows(
            rows,
            as_of_date=first_date + datetime.timedelta(days=7),
        )

        self.assertEqual(result[0]["source_day_count"], 3)
        self.assertEqual(result[0]["delivery_day_count"], 2)
        self.assertEqual(result[0]["label_gmv_per_delivery_day"], 60)


if __name__ == "__main__":
    unittest.main()

"""验证刷新样本的窗口、去重和历史首次日期保护。"""

import tempfile
import unittest
from pathlib import Path

from material_quality.sample.refresh_training_data import build_candidates, initialize_local


class RefreshTrainingDataTest(unittest.TestCase):
    """用本地数据库验证跨计划汇总和标签边界。"""

    def test_merges_plans_and_excludes_outside_window(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = initialize_local(Path(directory) / "source.sqlite3")
            connection.execute("INSERT INTO plans VALUES (1,'食品')")
            connection.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?)", [
                (1,"new","2026-09-01",10,1,1,10,1),
                (2,"new","2026-09-01",20,2,1,10,1),
                (1,"new","2026-09-02",30,3,1,10,1),
                (1,"new","2026-09-08",999,3,1,10,1),
            ])
            result = build_candidates(connection,"2026-09-08")
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]["label_gmv_7d"],60)
            self.assertEqual(result[0]["label_gmv_per_delivery_day"],30)
            self.assertEqual(result[0]["label_status"],"ready")
            self.assertEqual(result[0]["plan_ids"],[1,2])

    def test_preserves_first_date_and_rejects_open_or_short_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = initialize_local(Path(directory) / "source.sqlite3")
            connection.execute("INSERT INTO first_seen VALUES ('old','2026-07-01')")
            connection.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?)", [
                (1,"old","2026-09-01",10,1,1,10,1),
                (1,"open","2026-10-05",0,1,0,10,1),
                (1,"short","2026-09-01",0,1,0,10,1),
            ])
            results = {row["material_id"]:row for row in build_candidates(connection,"2026-10-10")}
            self.assertNotIn("old",results)
            self.assertEqual(results["open"]["label_status"],"window_open")
            self.assertEqual(results["short"]["label_status"],"insufficient_delivery_days")

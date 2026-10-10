"""验证采集顺序与脚本用量审计，避免标签排序和重复计费统计。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from material_quality.feature.codex_script_supervisor import read_codex_usage
from material_quality.feature.refresh_collection import stratified_targets
from material_quality.feature import refresh_collection


class RefreshCollectionTest(unittest.TestCase):
    """检查行业轮转和实际用量事件解析。"""

    def test_rotates_industries_without_label_priority(self):
        targets = [
            {"material_id":str(index),"first_stat_date":"2026-09-01",
             "primary_industry":industry,"label_gmv_7d":label}
            for index,industry,label in [(1,"食品",0),(2,"食品",100),(3,"家具",0),(4,"家具",100)]
        ]
        ordered = stratified_targets(targets)
        self.assertNotEqual(ordered[0]["primary_industry"],ordered[1]["primary_industry"])
        self.assertEqual({t["material_id"] for t in ordered},{"1","2","3","4"})
        changed = [{**target,"label_gmv_7d":999-target["label_gmv_7d"]} for target in targets]
        self.assertEqual([t["material_id"] for t in ordered],
                         [t["material_id"] for t in stratified_targets(changed)])

    def test_sums_only_completed_usage_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"codex.log"
            path.write_text("非结构化启动日志\n"+"\n".join(json.dumps(event) for event in [
                {"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}},
                {"type":"turn.completed","usage":{"input_tokens":30,"cached_input_tokens":10}},
                {"type":"item.completed","usage":{"input_tokens":999}},
            ]))
            self.assertEqual(read_codex_usage(str(path)),
                             {"input_tokens":130,"output_tokens":20,"cached_input_tokens":10})

    def test_retries_rate_limit_before_marking_material_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = {"runtime":directory,"temp_root":directory,"slots":directory,"raw_root":directory}
            auth = {1:{"user_id":1,"advertiser_id":"1","app_key":"default","access_token":"测试占位值"}}
            store = MagicMock()
            store.has_content.return_value = True
            responses = [
                {"error":"接口临时限流","request_id":"请求一"},
                {"detail":{"url":"https://example.com/video.mp4"},"request_id":"请求二"},
            ]
            with patch.object(refresh_collection,"_settings",settings), \
                    patch.object(refresh_collection,"load_auth",return_value=auth), \
                    patch.object(refresh_collection,"_auth_status",return_value="ready"), \
                    patch.object(refresh_collection,"_request_video",side_effect=responses) as request, \
                    patch.object(refresh_collection,"download_video",return_value={"content_sha256":"a"*64}), \
                    patch.object(refresh_collection,"LocalRawFeatureStore",return_value=store), \
                    patch.object(refresh_collection.time,"sleep"), \
                    patch.object(refresh_collection.shutil,"disk_usage",return_value=type("DiskUsage",(),{"free":100*1024**3})()):
                result = refresh_collection.collect_one({"asset_id":1,"material_id":"123","plan_ids":[1]})
            self.assertEqual(result["status"],"ready")
            self.assertEqual(request.call_count,2)
            self.assertEqual(result["api_retry_count"],1)
            self.assertEqual(result["external_request_ids"],["请求一","请求二"])

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

    def test_permission_cache_expires_and_accepts_new_token(self):
        candidate = {"user_id":1,"advertiser_id":"1","app_key":"default","access_token":"测试令牌甲"}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(refresh_collection,"_settings",{"runtime":directory}), \
                patch.object(refresh_collection,"_request_video",return_value={"error":"账户无权限"}) as request, \
                patch.object(refresh_collection.time,"time",return_value=1000) as now:
            refresh_collection.request_account_video(candidate,100)
            self.assertTrue(refresh_collection.request_account_video(candidate,101)["account_cache_hit"])
            self.assertEqual(request.call_count,1)
            now.return_value = 1901
            refresh_collection.request_account_video(candidate,102)
            self.assertEqual(request.call_count,2)
            refresh_collection.request_account_video({**candidate,"access_token":"测试令牌乙"},103)
            self.assertEqual(request.call_count,3)
            states = "".join(p.read_text() for p in Path(directory).rglob("*.json"))
            self.assertNotIn("测试令牌",states)

    def test_permission_cache_does_not_block_other_accounts_or_apps(self):
        candidate = {"user_id":1,"advertiser_id":"1","app_key":"default","access_token":"测试令牌"}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(refresh_collection,"_settings",{"runtime":directory}), \
                patch.object(refresh_collection,"_request_video",return_value={"error":"账户无权限"}) as request:
            refresh_collection.request_account_video(candidate,100)
            for changed in [{"user_id":2},{"advertiser_id":"2"},{"app_key":"其他应用"}]:
                refresh_collection.request_account_video({**candidate,**changed},101)
            self.assertEqual(request.call_count,4)

    def test_material_errors_and_rate_limits_do_not_block_account(self):
        candidate = {"user_id":1,"advertiser_id":"1","app_key":"default","access_token":"测试令牌"}
        responses = [{"error":"未返回素材"},{"error":"接口临时限流"},
                     {"detail":{"url":"素材所属主体与开发者主体不一致，无法获取URL"}}]
        for response in responses:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory, \
                    patch.object(refresh_collection,"_settings",{"runtime":directory}), \
                    patch.object(refresh_collection,"_request_video",return_value=response) as request:
                refresh_collection.request_account_video(candidate,100)
                refresh_collection.request_account_video(candidate,101)
                self.assertEqual(request.call_count,2)

    def test_cached_account_failure_still_tries_other_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = {"runtime":directory,"temp_root":directory,"slots":directory,"raw_root":directory}
            auth = {i:{"user_id":1,"advertiser_id":str(i),"app_key":"default","access_token":"测试令牌"}
                    for i in [1,2]}
            store = MagicMock()
            store.has_content.return_value = True
            with patch.object(refresh_collection,"_settings",settings), \
                    patch.object(refresh_collection,"load_auth",return_value=auth), \
                    patch.object(refresh_collection,"_auth_status",return_value="ready"), \
                    patch.object(refresh_collection,"_request_video",side_effect=[
                        {"error":"账户无权限"},{"detail":{"url":"https://example.com/video.mp4"}}]) as request, \
                    patch.object(refresh_collection,"download_video",return_value={"content_sha256":"b"*64}), \
                    patch.object(refresh_collection,"LocalRawFeatureStore",return_value=store), \
                    patch.object(refresh_collection.shutil,"disk_usage",return_value=type("DiskUsage",(),{"free":100*1024**3})()):
                refresh_collection.request_account_video(auth[1],100)
                result = refresh_collection.collect_one({"asset_id":1,"material_id":"123","plan_ids":[1,2]})
            self.assertEqual(result["status"],"ready")
            self.assertEqual(result["account_cache_skips"],1)
            self.assertEqual(request.call_count,2)

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

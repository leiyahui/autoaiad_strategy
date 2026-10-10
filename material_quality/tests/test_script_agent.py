"""线上千问脚本理解、严格结构校验和Sidecar测试。"""

import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from material_quality.feature.script_agent_features import (
    SCRIPT_ANALYSIS_SCHEMA,
    build_dashscope_script_request,
    canonical_script_text,
    has_script_feature,
    parse_dashscope_script_analysis,
    save_script_feature,
    select_evidence_frames,
)


def _analysis_payload():
    """构造满足严格字段约束的最小脚本结果。"""
    return {
        "objective_summary": "展示床垫结构并引导进入直播间咨询",
        "content_category": "床垫",
        "presentation_style": "商品展示",
        "opening_hook": {
            "hook_type": "痛点",
            "visual": "床垫近景",
            "spoken_or_text": "买床垫最怕躺久腰酸",
            "estimated_end_second": 3.0,
            "evidence": "ASR开场与前三张代表帧",
        },
        "storyline": [{
            "start_second": 0.0,
            "end_second": 8.0,
            "stage": "卖点讲解",
            "visual_action": "展示床垫内部材料",
            "narration": "介绍支撑棉和黄麻",
            "onscreen_text": "高密支撑棉",
            "communication_purpose": "说明商品结构",
        }],
        "reconstructed_script": "买床垫最怕躺久腰酸，这张床垫使用高密支撑棉。",
        "selling_points": ["高密支撑棉"],
        "pain_points": ["久躺腰酸"],
        "proof_elements": ["内部材料展示"],
        "promotion_offers": [],
        "calls_to_action": ["进入直播间咨询"],
        "characters": [],
        "settings": ["商品展示场景"],
        "transitions": [],
        "pacing": {
            "tempo": "中等",
            "scene_count": 2,
            "opening_seconds": 3.0,
            "average_scene_seconds": 4.0,
            "density_notes": "语音连续",
        },
        "evidence_limitations": [],
    }


def _content_manifest(temporary_directory):
    """创建不含投放标签的代表帧和内容档案。"""
    frame_directory = os.path.join(temporary_directory, "frames")
    os.makedirs(frame_directory)
    frame_records = []
    for index in range(3):
        filename = "frame_{}.webp".format(index)
        Image.new("RGB", (32, 48), color=(index * 20, 0, 0)).save(
            os.path.join(frame_directory, filename),
            format="WEBP",
        )
        frame_records.append({
            "timestamp": float(index),
            "file": filename,
            "reasons": ["opening"],
        })
    Image.new("RGB", (64, 64)).save(
        os.path.join(temporary_directory, "sheet.jpg"),
        format="JPEG",
    )
    manifest_path = os.path.join(temporary_directory, "content.json")
    with open(manifest_path, "w", encoding="utf-8") as output:
        json.dump({
            "content_sha256": "a" * 64,
            "feature_version": "source_v4",
            "duration": 10,
            "frame_timestamps": [0, 1, 2],
            "scene_change_timestamps": [1.5],
            "ocr_text": "屏幕文字",
            "asr_text": "语音内容",
            "asr_segments": [{"start": 0, "end": 2, "text": "语音内容"}],
            "contact_sheet_file": "sheet.jpg",
            "keyframe_directory": "frames",
            "keyframe_files": frame_records,
        }, output, ensure_ascii=False)
    return manifest_path


class ScriptAgentFeatureTest(unittest.TestCase):
    """验证线上请求不携带标签且响应不能绕过结构约束。"""

    def test_selects_opening_and_full_timeline_frames(self):
        frames = [
            {"timestamp": float(index), "file": "{}.webp".format(index)}
            for index in range(20)
        ]

        selected = select_evidence_frames(frames, max_frames=8)

        self.assertEqual(len(selected), 8)
        self.assertEqual(
            [item["timestamp"] for item in selected[:4]],
            [0.0, 1.0, 2.0, 3.0],
        )
        self.assertEqual(selected[-1]["timestamp"], 19.0)

    def test_builds_dashscope_label_free_multimodal_request(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = _content_manifest(temporary_directory)

            request, audit = build_dashscope_script_request(
                manifest_path,
                max_frames=3,
            )
            serialized = json.dumps(request, ensure_ascii=False)

            self.assertEqual(request["model"], "qwen-vl-max")
            self.assertEqual(request["response_format"], {"type": "json_object"})
            self.assertFalse(request["enable_thinking"])
            self.assertEqual(request["messages"][0]["role"], "system")
            self.assertIn("image_url", serialized)
            self.assertIn("语音内容", serialized)
            self.assertNotIn("gmv", serialized.lower())
            self.assertNotIn("asset_id", serialized)
            self.assertEqual(len(audit["selected_frames"]), 3)
            self.assertEqual(audit["provider"], "dashscope")

    def test_parses_dashscope_json_and_rejects_schema_drift(self):
        analysis = _analysis_payload()
        response = {
            "choices": [{
                "message": {
                    "content": json.dumps(analysis, ensure_ascii=False),
                },
            }],
        }

        parsed = parse_dashscope_script_analysis(response)

        self.assertEqual(parsed["content_category"], "床垫")
        invalid = dict(analysis)
        invalid.pop("storyline")
        with self.assertRaisesRegex(ValueError, "缺少字段"):
            parse_dashscope_script_analysis({
                "choices": [{
                    "message": {
                        "content": json.dumps(invalid, ensure_ascii=False),
                    },
                }],
            })

    def test_rejects_nested_schema_drift(self):
        analysis = _analysis_payload()
        analysis["opening_hook"] = "以痛点口播开场"

        with self.assertRaisesRegex(ValueError, "opening_hook"):
            parse_dashscope_script_analysis({
                "choices": [{
                    "message": {
                        "content": json.dumps(analysis, ensure_ascii=False),
                    },
                }],
            })

    def test_normalizes_generic_proof_stage_from_dashscope(self):
        """千问偶发返回“证明”时应归一到稳定训练枚举。"""
        analysis = _analysis_payload()
        analysis["storyline"][0]["stage"] = "证明"

        parsed = parse_dashscope_script_analysis({
            "choices": [{
                "message": {
                    "content": json.dumps(analysis, ensure_ascii=False),
                },
            }],
        })

        self.assertEqual(parsed["storyline"][0]["stage"], "对比证明")

    def test_builds_canonical_text_and_saves_sidecar(self):
        analysis = _analysis_payload()
        canonical = canonical_script_text(analysis)

        self.assertIn("高密支撑棉", canonical)
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = {
                "content_sha256": "b" * 64,
                "embedding_dim": 16,
                "analysis": analysis,
            }
            save_script_feature(
                temporary_directory,
                "script_test_v1",
                result,
                np.ones(16, dtype=np.float32),
            )
            self.assertTrue(has_script_feature(
                temporary_directory,
                "script_test_v1",
                "b" * 64,
            ))


if __name__ == "__main__":
    unittest.main()

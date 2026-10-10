"""素材质量线上原始特征存储与清理测试。"""

import datetime
import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from material_quality.feature.local_artifact_store import LocalRawFeatureStore
from material_quality.feature.semantic_features import build_keyframe_timestamps
from material_quality.feature.service import pack_float32


class MaterialQualityLocalPipelineTest(unittest.TestCase):
    """验证线上特征档案完整保留模型输入并可安全清理媒体。"""

    def test_keyframe_timestamps_cover_opening_and_full_timeline(self):
        timestamps = build_keyframe_timestamps(30.0, 24, [8.0, 18.0])

        self.assertEqual(len(timestamps), 24)
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertTrue(all(
            value in timestamps
            for value in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
        ))
        self.assertTrue(all(value in timestamps for value in (7.8, 8.0, 17.8, 18.0)))
        self.assertGreater(timestamps[-1], 28.0)
        self.assertLess(timestamps[-1], 30.0)

    def test_raw_feature_store_preserves_per_frame_and_text_embeddings(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            frame_paths = []
            for frame_index, color in enumerate(((255, 0, 0), (0, 255, 0))):
                frame_path = os.path.join(
                    temporary_dir,
                    "frame_{}.jpg".format(frame_index),
                )
                Image.new("RGB", (32, 48), color=color).save(frame_path)
                frame_paths.append(frame_path)

            store = LocalRawFeatureStore(
                os.path.join(temporary_dir, "features"),
                "raw_v1",
            )
            target = {
                "asset_id": 10,
                "material_id": "2001",
                "cohort": "all_industries_v1",
                "label_status": "ready",
                "label_gmv_7d": 300,
                "label_gmv_per_delivery_day": 100,
                "delivery_day_count": 3,
                "first_stat_date": datetime.date(2026, 7, 1),
                "primary_industry": "休闲食品",
                "industry_count": 1,
            }
            feature_info = {
                "raw_visual_embeddings": np.ones((2, 4), dtype=np.float32),
                "raw_ocr_chunk_embeddings": np.full((2, 4), 2, dtype=np.float32),
                "raw_asr_chunk_embeddings": np.full((1, 4), 3, dtype=np.float32),
                "raw_frame_paths": frame_paths,
                "raw_frame_timestamps": [0.0, 1.5],
                "raw_frame_reasons": [["opening"], ["timeline"]],
                "raw_scene_change_timestamps": [],
                "raw_ocr_chunks": ["限时", "优惠"],
                "raw_asr_chunks": ["现在下单"],
                "visual_vector": pack_float32([1.0, 2.0]),
                "visual_vector_dim": 2,
                "visual_temporal_vector": pack_float32([0.1, 0.2]),
                "visual_temporal_dim": 2,
                "text_vector": pack_float32([3.0, 4.0]),
                "text_vector_dim": 2,
                "ocr_stats_vector": pack_float32([5.0, 6.0]),
                "ocr_stats_dim": 2,
                "asr_text_vector": pack_float32([7.0, 8.0]),
                "asr_text_vector_dim": 2,
                "audio_stats_vector": pack_float32([9.0, 10.0]),
                "audio_stats_dim": 2,
                "visual_model_name": "测试视觉模型",
                "asr_model_name": "测试语音模型",
                "asr_language": "zh",
                "asr_language_probability": 0.9,
                "duration": 3.0,
                "width": 1080,
                "height": 1920,
                "fps": 30.0,
                "bit_rate": 1000,
                "has_audio": True,
                "sample_frame_count": 2,
                "ocr_text": "限时优惠",
                "ocr_json": "[]",
                "asr_text": "现在下单",
                "asr_json": "[]",
                "raw_probe": {},
                "extractor_config": {"keyframe_count": 2},
            }
            content_sha256 = "b" * 64
            source_video_path = os.path.join(temporary_dir, "source.mp4")
            with open(source_video_path, "wb") as output_file:
                output_file.write(b"test-video-content")

            feature_info["has_audio"] = False
            content_directory = store.save(
                target,
                content_sha256,
                feature_info,
                source_video_path=source_video_path,
            )

            vector_path = os.path.join(content_directory, content_sha256 + ".npz")
            with np.load(vector_path) as arrays:
                self.assertEqual(arrays["visual_frames"].shape, (2, 4))
                self.assertEqual(arrays["ocr_chunks"].shape, (2, 4))
                self.assertEqual(arrays["asr_chunks"].shape, (1, 4))
                self.assertEqual(arrays["frame_timestamps"].tolist(), [0.0, 1.5])

            manifest_path = os.path.join(content_directory, content_sha256 + ".json")
            with open(manifest_path, "r", encoding="utf-8") as input_file:
                manifest = json.load(input_file)
            self.assertEqual(manifest["ocr_chunks"], ["限时", "优惠"])
            self.assertTrue(os.path.isfile(
                os.path.join(content_directory, content_sha256 + ".jpg")
            ))
            frame_directory = os.path.join(
                content_directory,
                content_sha256 + ".frames",
            )
            self.assertEqual(len(os.listdir(frame_directory)), 2)
            self.assertEqual(len(manifest["keyframe_files"]), 2)
            self.assertTrue(all(
                item["file"].endswith(".webp")
                for item in manifest["keyframe_files"]
            ))
            self.assertEqual(
                manifest["keyframe_files"][0]["reasons"],
                ["opening"],
            )
            self.assertIsNone(manifest["audio_archive_file"])
            self.assertTrue(store.has_content(content_sha256))
            self.assertTrue(os.path.isfile(os.path.join(
                store.asset_root,
                "10.json",
            )))


if __name__ == "__main__":
    unittest.main()

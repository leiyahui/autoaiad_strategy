"""在线素材媒体探测的回归测试。"""

import json
import subprocess
import unittest
from unittest.mock import patch


class MediaFeatureTests(unittest.TestCase):
    """验证在线特征提取依赖的媒体信息接口。"""

    def test_probe_video_returns_online_extractor_fields(self):
        """模块重构后仍应提供在线提取器需要的完整媒体信息。"""
        try:
            from material_quality.feature.media_features import probe_video
        except ModuleNotFoundError as exc:
            if exc.name != "material_quality.feature.media_features":
                raise
            self.fail("在线媒体探测模块无法导入：{}".format(exc))

        raw_probe = {
            "format": {
                "duration": "12.5",
                "size": "2048",
                "bit_rate": "300000",
            },
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "width": 1080,
                    "height": 1920,
                    "avg_frame_rate": "30000/1001",
                    "r_frame_rate": "30/1",
                    "bit_rate": "280000",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "sample_rate": "48000",
                    "channels": 2,
                },
            ],
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(raw_probe).encode("utf-8"),
            stderr=b"",
        )

        with patch(
            "material_quality.feature.media_features.subprocess.run",
            return_value=completed,
        ):
            result = probe_video("/tmp/material.mp4")

        self.assertEqual(result["duration"], 12.5)
        self.assertEqual(result["width"], 1080)
        self.assertEqual(result["height"], 1920)
        self.assertAlmostEqual(result["fps"], 30000.0 / 1001.0)
        self.assertEqual(result["bit_rate"], 300000)
        self.assertEqual(result["file_size"], 2048)
        self.assertTrue(result["has_audio"])
        self.assertEqual(result["raw_probe"], raw_probe)


if __name__ == "__main__":
    unittest.main()

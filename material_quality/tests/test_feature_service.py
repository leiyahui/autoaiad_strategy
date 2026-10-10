"""素材质量特征纯计算逻辑测试。"""

import os
import struct
import tempfile
import unittest

import numpy as np

from material_quality.feature.service import (
    VISUAL_VECTOR_DIM,
    build_visual_vector,
    pack_float32,
    parse_fraction,
)
from material_quality.feature.semantic_features import (
    AUDIO_STATS_DIM,
    FileSlotPool,
    OCR_STATS_DIM,
    VISUAL_TEMPORAL_DIM,
    SemanticFeatureExtractor,
    aggregate_frame_embeddings,
    build_audio_stats,
    build_ocr_features,
    build_visual_temporal_features,
    find_audio_split_sample,
    merge_asr_results,
    normalize_ocr_text,
    parse_rapidocr_output,
    split_text_chunks,
)


class MaterialQualityFeatureServiceTest(unittest.TestCase):
    """验证线上抽取使用的帧、文本和音频特征。"""

    def test_parses_ffprobe_fraction(self):
        self.assertAlmostEqual(parse_fraction("30000/1001"), 29.97002997)
        self.assertEqual(parse_fraction("25"), 25.0)
        self.assertEqual(parse_fraction("0/0"), 0.0)
        self.assertEqual(parse_fraction("异常"), 0.0)

    def test_builds_fixed_visual_vector(self):
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        vector, frame_count = build_visual_vector(black.tobytes() + white.tobytes())

        self.assertEqual(frame_count, 2)
        self.assertEqual(len(vector), VISUAL_VECTOR_DIM)
        self.assertGreater(vector[-2], 0)

    def test_rejects_incomplete_frame_bytes(self):
        with self.assertRaisesRegex(ValueError, "长度不完整"):
            build_visual_vector(b"123")

    def test_packs_float32_vector(self):
        packed = pack_float32([1.0, 2.5, -3.0])
        self.assertEqual(len(packed), 12)
        self.assertEqual(struct.unpack("<3f", packed), (1.0, 2.5, -3.0))

    def test_aggregates_semantic_frame_embeddings(self):
        embeddings = np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)

        vector = aggregate_frame_embeddings(embeddings)

        self.assertEqual(len(vector), 4)
        self.assertTrue(np.isfinite(vector).all())

    def test_normalizes_and_deduplicates_ocr_features(self):
        output = [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], " 限时 折扣 ", 0.9),
            ([[0, 30], [50, 30], [50, 50], [0, 50]], "限时折扣", 0.8),
        ]

        records = parse_rapidocr_output(output, 0, 100, 100)
        text_value, stats, normalized_records = build_ocr_features(records, 2)

        self.assertEqual(normalize_ocr_text(" 限时 折扣 "), "限时折扣")
        self.assertEqual(text_value, "限时 折扣")
        self.assertEqual(len(normalized_records), 2)
        self.assertEqual(len(stats), OCR_STATS_DIM)
        self.assertEqual(stats[0], 2)
        self.assertEqual(stats[1], 1)

    def test_parses_numpy_rapidocr_output(self):
        class RapidOutput:
            """模拟RapidOCR 2返回的NumPy数组字段。"""

            boxes = np.asarray([[[0, 0], [20, 0], [20, 10], [0, 10]]])
            txts = np.asarray(["立即抢购"])
            scores = np.asarray([0.95])

        records = parse_rapidocr_output(RapidOutput(), 0, 100, 100)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["normalized_text"], "立即抢购")
        self.assertAlmostEqual(records[0]["score"], 0.95)

    def test_splits_long_text_without_losing_tail(self):
        chunks = split_text_chunks("一二三四五六七八九十", chunk_chars=4, max_chunks=3)

        self.assertEqual(chunks, ["一二三四", "五六七八", "九十"])

    def test_finds_low_energy_audio_split_near_midpoint(self):
        waveform = np.ones(16000 * 10, dtype=np.float32)
        waveform[16000 * 5 - 2000:16000 * 5 + 2000] = 0.0

        split_sample = find_audio_split_sample(
            waveform,
            sample_rate=16000,
            minimum_chunk_seconds=2.0,
            search_seconds=2.0,
            window_seconds=0.25,
        )

        self.assertIsNotNone(split_sample)
        self.assertLess(abs(int(split_sample) - 16000 * 5), 5000)

    def test_merges_asr_chunks_with_timeline_offsets(self):
        merged = merge_asr_results([
            (0.0, 10.0, {
                "text": "第一段",
                "segments": [{"start": 1.0, "end": 2.0, "text": "第一段"}],
                "language": "zh",
                "language_probability": 0.8,
            }),
            (10.0, 10.0, {
                "text": "第二段",
                "segments": [{"start": 0.5, "end": 1.5, "text": "第二段"}],
                "language": "zh",
                "language_probability": 1.0,
            }),
        ])

        self.assertEqual(merged["text"], "第一段\n第二段")
        self.assertEqual(merged["segments"][1]["start"], 10.5)
        self.assertAlmostEqual(merged["language_probability"], 0.9)

    def test_file_slot_pool_limits_cross_instance_parallelism(self):
        with tempfile.TemporaryDirectory() as directory:
            first_pool = FileSlotPool(directory, "测试", 1)
            second_pool = FileSlotPool(directory, "测试", 1)
            lease = first_pool.acquire()

            self.assertIsNotNone(lease)
            self.assertIsNone(second_pool.acquire())
            lease.release()
            next_lease = second_pool.acquire()
            self.assertIsNotNone(next_lease)
            next_lease.release()

    def test_parallel_ocr_preserves_frame_order(self):
        class FakeEngine:
            """记录分配到各OCR引擎的帧文件。"""

            def __init__(self):
                self.calls = []

            def __call__(self, frame_path):
                self.calls.append(os.path.basename(frame_path))
                return [(
                    [[0, 0], [20, 0], [20, 10], [0, 10]],
                    os.path.basename(frame_path),
                    0.9,
                )]

        from PIL import Image

        extractor = SemanticFeatureExtractor.__new__(SemanticFeatureExtractor)
        extractor.ocr_engines = [FakeEngine(), FakeEngine()]
        with tempfile.TemporaryDirectory() as directory:
            frame_paths = []
            for frame_index in range(4):
                frame_path = os.path.join(
                    directory,
                    "frame_{:03d}.jpg".format(frame_index),
                )
                Image.new("RGB", (32, 32)).save(frame_path)
                frame_paths.append(frame_path)
            records = extractor._extract_ocr_records(
                frame_paths,
                [0.0, 1.0, 2.0, 3.0],
            )

        self.assertEqual(
            [record["frame_index"] for record in records],
            [0, 1, 2, 3],
        )
        self.assertEqual(len(extractor.ocr_engines[0].calls), 2)
        self.assertEqual(len(extractor.ocr_engines[1].calls), 2)

    def test_builds_visual_temporal_features(self):
        embeddings = np.asarray([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        features = build_visual_temporal_features(embeddings, [])

        self.assertEqual(len(features), VISUAL_TEMPORAL_DIM)
        self.assertEqual(features[0], 1.0)

    def test_builds_audio_stats(self):
        sample_rate = 16000
        timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
        waveform = 0.1 * np.sin(2 * np.pi * 440 * timeline)
        segments = [{
            "start": 0.1,
            "end": 0.9,
            "avg_logprob": -0.2,
            "no_speech_prob": 0.1,
        }]

        features = build_audio_stats(
            waveform,
            sample_rate,
            segments,
            "今天限时优惠",
        )

        self.assertEqual(len(features), AUDIO_STATS_DIM)
        self.assertAlmostEqual(features[0], 1.0)
        self.assertAlmostEqual(features[8], 0.8)

if __name__ == "__main__":
    unittest.main()

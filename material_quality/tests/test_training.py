"""素材质量训练数据和评估纯计算逻辑测试。"""

import datetime
import json
import os
import tempfile
import unittest

import numpy as np

from material_quality.training.dataset import (
    FEATURE_NAMES,
    VISUAL_FEATURE_DIM,
    build_feature_row,
    build_population_weights,
)
from material_quality.training.evaluation import evaluate_top_fraction, restore_gmv_from_log
from material_quality.training.semantic_dataset import (
    ASR_SEMANTIC_DIM,
    AUDIO_STATS_DIM,
    FEATURE_GROUP_RANGES,
    FEATURE_NAMES as SEMANTIC_FEATURE_NAMES,
    OCR_SEMANTIC_DIM,
    OCR_STATS_DIM,
    VISUAL_SEMANTIC_DIM,
    VISUAL_TEMPORAL_DIM,
    build_semantic_feature_row,
    load_local_semantic_dataset,
)
from material_quality.training.train_semantic_baseline import build_time_masks
from material_quality.training.train_regularized_classifier import (
    build_rolling_time_folds,
    regularized_selection_score,
    select_conservative_candidate,
)
from material_quality.training.train_late_fusion_classifier import (
    select_fusion_candidate,
    weighted_probability,
)
from material_quality.training.evaluate_account_spend import (
    build_account_relative_labels,
)


class MaterialQualityTrainingTest(unittest.TestCase):
    """验证训练特征维度、抽样权重和头部指标。"""

    def test_builds_combined_material_features(self):
        visual = np.linspace(0.0, 1.0, VISUAL_FEATURE_DIM, dtype="<f4")
        row = {
            "visual_vector_dim": VISUAL_FEATURE_DIM,
            "visual_vector": visual.tobytes(),
            "duration": 30,
            "file_size": 1024,
            "bit_rate": 2000000,
            "fps": 30,
            "width": 1080,
            "height": 1920,
        }

        features = build_feature_row(row)

        self.assertEqual(features.size, len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[-2], 1.0)

    def test_reweights_month_and_label_strata(self):
        july = datetime.date(2026, 7, 1).toordinal()
        dates = np.asarray([july, july, july, july], dtype=np.int32)
        labels = np.asarray([1.0, 1.0, 0.0, 0.0])
        population = {("2026-07", 1): 20, ("2026-07", 0): 80}

        weights = build_population_weights(dates, labels, population)

        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertGreater(weights[2], weights[0])
        self.assertAlmostEqual(weights[2] / weights[0], 4.0)

    def test_evaluates_top_fraction_lift(self):
        actual = np.asarray([100.0, 20.0, 0.0, 0.0, 0.0])
        predicted = np.asarray([0.9, 0.8, 0.3, 0.2, 0.1])

        metrics = evaluate_top_fraction(actual, predicted, 0.4)

        self.assertEqual(metrics["selected_count"], 2)
        self.assertGreater(metrics["positive_rate_lift"], 1.0)
        self.assertEqual(metrics["gmv_capture"], 1.0)

    def test_clips_log_prediction_before_restoring_gmv(self):
        predicted = restore_gmv_from_log(np.asarray([1.0, 1000.0]), 10.0)

        self.assertTrue(np.isfinite(predicted).all())
        self.assertAlmostEqual(predicted[1], np.expm1(10.0))

    def test_builds_multimodal_semantic_features(self):
        def vector(dim, value):
            return np.full(dim, value, dtype="<f4").tobytes()

        row = {
            "visual_vector": vector(VISUAL_SEMANTIC_DIM, 1.0),
            "visual_vector_dim": VISUAL_SEMANTIC_DIM,
            "visual_temporal_vector": vector(VISUAL_TEMPORAL_DIM, 2.0),
            "visual_temporal_dim": VISUAL_TEMPORAL_DIM,
            "text_vector": vector(OCR_SEMANTIC_DIM, 3.0),
            "text_vector_dim": OCR_SEMANTIC_DIM,
            "ocr_stats_vector": vector(OCR_STATS_DIM, 4.0),
            "ocr_stats_dim": OCR_STATS_DIM,
            "asr_text_vector": vector(ASR_SEMANTIC_DIM, 5.0),
            "asr_text_vector_dim": ASR_SEMANTIC_DIM,
            "audio_stats_vector": vector(AUDIO_STATS_DIM, 6.0),
            "audio_stats_dim": AUDIO_STATS_DIM,
            "duration": 30,
            "file_size": 1024,
            "bit_rate": 2000000,
            "fps": 30,
            "width": 1080,
            "height": 1920,
            "has_audio": 1,
        }

        features = build_semantic_feature_row(row)

        media_start, media_end = FEATURE_GROUP_RANGES["media"]
        self.assertEqual(features.size, len(SEMANTIC_FEATURE_NAMES))
        self.assertEqual(media_end, len(SEMANTIC_FEATURE_NAMES))
        self.assertEqual(features[media_start + 8], 1.0)
        self.assertEqual(features[media_start + 9], 1.0)
        self.assertEqual(features[media_start + 10], 1.0)
        self.assertTrue(np.isfinite(features).all())

    def test_zero_fills_missing_text_modalities(self):
        def vector(dim, value):
            return np.full(dim, value, dtype="<f4").tobytes()

        row = {
            "visual_vector": vector(VISUAL_SEMANTIC_DIM, 1.0),
            "visual_vector_dim": VISUAL_SEMANTIC_DIM,
            "visual_temporal_vector": vector(VISUAL_TEMPORAL_DIM, 2.0),
            "visual_temporal_dim": VISUAL_TEMPORAL_DIM,
            "text_vector": None,
            "text_vector_dim": 0,
            "ocr_stats_vector": vector(OCR_STATS_DIM, 4.0),
            "ocr_stats_dim": OCR_STATS_DIM,
            "asr_text_vector": None,
            "asr_text_vector_dim": 0,
            "audio_stats_vector": vector(AUDIO_STATS_DIM, 6.0),
            "audio_stats_dim": AUDIO_STATS_DIM,
        }

        features = build_semantic_feature_row(row)

        ocr_start, _ = FEATURE_GROUP_RANGES["ocr"]
        asr_start, _ = FEATURE_GROUP_RANGES["asr"]
        media_start, _ = FEATURE_GROUP_RANGES["media"]
        self.assertTrue(np.all(features[ocr_start:ocr_start + OCR_SEMANTIC_DIM] == 0))
        self.assertTrue(np.all(features[asr_start:asr_start + ASR_SEMANTIC_DIM] == 0))
        self.assertEqual(features[media_start + 9], 0.0)
        self.assertEqual(features[media_start + 10], 0.0)

    def test_builds_chronological_semantic_splits(self):
        dates = np.asarray([
            datetime.date(2026, 7, 20).toordinal(),
            datetime.date(2026, 7, 25).toordinal(),
            datetime.date(2026, 7, 31).toordinal(),
            datetime.date(2026, 8, 1).toordinal(),
        ], dtype=np.int32)

        masks = build_time_masks(dates, datetime.date(2026, 8, 1), 7)

        self.assertEqual(masks["train"].tolist(), [True, False, False, False])
        self.assertEqual(masks["validation"].tolist(), [False, True, True, False])
        self.assertEqual(masks["test"].tolist(), [False, False, False, True])

    def test_builds_multiple_rolling_time_folds_without_future_leakage(self):
        start = datetime.date(2026, 7, 1)
        dates = []
        labels = []
        for offset in range(35):
            current = start + datetime.timedelta(days=offset)
            for item_index in range(10):
                dates.append(current.toordinal())
                labels.append(item_index % 2)
        date_values = np.asarray(dates, dtype=np.int32)
        binary = np.asarray(labels, dtype=np.int32)

        folds = build_rolling_time_folds(
            date_values,
            datetime.date(2026, 8, 5),
            7,
            3,
            binary,
            minimum_train_count=50,
            minimum_validation_count=20,
        )

        self.assertEqual(len(folds), 3)
        for fold in folds:
            train_dates = date_values[fold["train"]]
            validation_dates = date_values[fold["validation"]]
            self.assertLess(int(train_dates.max()), int(validation_dates.min()))
            self.assertLess(int(validation_dates.max()), datetime.date(2026, 8, 5).toordinal())

    def test_regularized_score_penalizes_instability_and_fit_gap(self):
        stable = regularized_selection_score(0.65, 0.01, 0.05)
        unstable = regularized_selection_score(0.65, 0.08, 0.20)

        self.assertGreater(stable, unstable)

    def test_selects_simpler_candidate_when_scores_are_close(self):
        candidates = [
            {
                "selection_score": 0.600,
                "profile": {"pca_components": 32, "c": 0.003},
            },
            {
                "selection_score": 0.594,
                "profile": {"pca_components": 8, "c": 0.001},
            },
        ]

        selected = select_conservative_candidate(candidates)

        self.assertEqual(selected["profile"]["pca_components"], 8)

    def test_combines_tower_probabilities_with_normalized_weights(self):
        probabilities = {
            "visual": np.asarray([0.2, 0.8]),
            "script": np.asarray([0.6, 0.4]),
            "asr": np.asarray([0.5, 0.5]),
        }

        combined = weighted_probability(
            probabilities,
            {"visual": 4.0, "script": 4.0, "asr": 2.0},
        )

        np.testing.assert_allclose(combined, np.asarray([0.42, 0.58]))

    def test_selects_simpler_fusion_when_validation_scores_are_close(self):
        candidates = [
            {
                "selection_score": 0.600,
                "weights": {"visual": 0.4, "script": 0.4, "asr": 0.2},
            },
            {
                "selection_score": 0.596,
                "weights": {"visual": 0.5, "script": 0.5},
            },
        ]

        selected = select_fusion_candidate(candidates)

        self.assertEqual(selected["weights"], {"visual": 0.5, "script": 0.5})

    def test_builds_balanced_labels_within_each_account(self):
        accounts = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
        costs = np.asarray([0.0, 10.0, 20.0, 30.0, 0.0, 0.0, 5.0, 15.0])
        candidates = np.ones(costs.shape, dtype=bool)

        labels, weights, valid = build_account_relative_labels(
            accounts,
            costs,
            candidates,
            minimum_materials=4,
        )

        self.assertTrue(valid.all())
        for account_id in ("a", "b"):
            account_mask = accounts == account_id
            self.assertAlmostEqual(float(weights[account_mask].sum()), 1.0)
            self.assertEqual(set(labels[account_mask].tolist()), {0, 1})

    def test_loads_mature_local_v4_snapshot(self):
        """本地快照只读取成熟标签，并复用V4聚合向量。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            version = "multimodal_test_v4"
            version_root = os.path.join(temporary_directory, version)
            asset_root = os.path.join(version_root, "assets")
            content_sha256 = "a" * 64
            content_root = os.path.join(
                version_root,
                "content",
                content_sha256[:2],
            )
            os.makedirs(asset_root)
            os.makedirs(content_root)
            reference = {
                "asset_id": 12,
                "material_id": "material-12",
                "cohort": "all_industries_v1",
                "feature_version": version,
                "content_sha256": content_sha256,
                "label_status": "ready",
                "label_gmv_per_delivery_day": 88.5,
                "delivery_day_count": 3,
                "first_stat_date": "2026-07-20",
                "primary_industry": "食品",
            }
            with open(
                os.path.join(asset_root, "12.json"),
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(reference, output, ensure_ascii=False)
            content_base = os.path.join(content_root, content_sha256)
            with open(content_base + ".json", "w", encoding="utf-8") as output:
                json.dump({
                    "content_sha256": content_sha256,
                    "duration": 20,
                    "width": 1080,
                    "height": 1920,
                    "fps": 30,
                    "bit_rate": 2000000,
                    "has_audio": True,
                    "probe": {"format": {"size": "4096"}},
                }, output, ensure_ascii=False)
            np.savez_compressed(
                content_base + ".npz",
                visual_aggregate=np.ones(VISUAL_SEMANTIC_DIM, dtype=np.float32),
                visual_temporal=np.ones(VISUAL_TEMPORAL_DIM, dtype=np.float32),
                ocr_aggregate=np.ones(OCR_SEMANTIC_DIM, dtype=np.float32),
                ocr_stats=np.ones(OCR_STATS_DIM, dtype=np.float32),
                asr_aggregate=np.ones(ASR_SEMANTIC_DIM, dtype=np.float32),
                audio_stats=np.ones(AUDIO_STATS_DIM, dtype=np.float32),
            )

            dataset, manifest = load_local_semantic_dataset(
                temporary_directory,
                "all_industries_v1",
                version,
            )

            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["feature_dim"], len(SEMANTIC_FEATURE_NAMES))
            self.assertEqual(dataset["labels"].tolist(), [88.5])
            self.assertEqual(dataset["primary_industries"].tolist(), ["食品"])


if __name__ == "__main__":
    unittest.main()

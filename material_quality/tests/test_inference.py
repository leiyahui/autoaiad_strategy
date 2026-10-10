"""素材质量正式模型包和推理入口测试。"""

import hashlib
import json
import os
import tempfile
import unittest

import joblib
import numpy as np
import sklearn

from material_quality.inference.model import (
    MODEL_SCHEMA_VERSION,
    MaterialQualityPredictor,
    build_serving_tower_matrices,
)
from material_quality.inference.feature_schema import (
    FEATURE_GROUP_RANGES,
    FEATURE_NAMES,
)


class FixedProbabilityModel:
    """以输入均值生成稳定概率，避免单元测试依赖训练过程。"""

    def predict_proba(self, matrix):
        values = np.asarray(matrix, dtype=np.float64)
        positive = np.clip(values.mean(axis=1), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


def _file_sha256(path):
    """计算测试模型摘要。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        digest.update(source.read())
    return digest.hexdigest()


class MaterialQualityInferenceTest(unittest.TestCase):
    """验证锁定推理布局，并拒绝损坏模型包。"""

    def test_serving_tower_layout_matches_locked_feature_schema(self):
        base = np.arange(
            3 * len(FEATURE_NAMES),
            dtype=np.float32,
        ).reshape(3, -1)
        script = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
        serving = build_serving_tower_matrices(base, script, 8)
        visual_start, visual_end = FEATURE_GROUP_RANGES["visual"]
        media_start, media_end = FEATURE_GROUP_RANGES["media"]
        expected_visual = np.hstack([
            base[:, visual_start:visual_end],
            base[:, media_start:media_end],
        ])

        np.testing.assert_array_equal(serving["visual"], expected_visual)
        np.testing.assert_array_equal(serving["script"], script)

    def test_loads_package_and_replays_smoke_predictions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = np.full((2, len(FEATURE_NAMES)), 0.2, dtype=np.float32)
            script = np.full((2, 8), 0.6, dtype=np.float32)
            package = {
                "schema_version": MODEL_SCHEMA_VERSION,
                "model_version": "test-v1",
                "models": {
                    "visual": FixedProbabilityModel(),
                    "script": FixedProbabilityModel(),
                },
                "fusion_weights": {"visual": 0.5, "script": 0.5},
                "script_feature_dim": 8,
                "reference_scores": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            }
            model_path = os.path.join(temporary_directory, "model.joblib")
            joblib.dump(package, model_path)
            expected = np.full(2, 0.4, dtype=np.float64)
            np.savez_compressed(
                os.path.join(temporary_directory, "smoke_test.npz"),
                base_features=base,
                script_features=script,
                positive_probability=expected,
            )
            with open(
                os.path.join(temporary_directory, "manifest.json"),
                "w",
                encoding="utf-8",
            ) as output:
                json.dump({
                    "schema_version": MODEL_SCHEMA_VERSION,
                    "model_version": "test-v1",
                    "model_sha256": _file_sha256(model_path),
                    "runtime": {
                        "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__,
                    },
                }, output)

            predictor = MaterialQualityPredictor.load(temporary_directory)
            result = predictor.predict_one(base[0], script[0])
            smoke_ok, maximum_error = predictor.verify_smoke_test()

            self.assertEqual(result["model_version"], "test-v1")
            self.assertAlmostEqual(result["positive_probability"], 0.4)
            self.assertEqual(result["quality_level"], "高")
            self.assertTrue(smoke_ok)
            self.assertAlmostEqual(maximum_error, 0.0)

    def test_rejects_model_file_with_wrong_digest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = os.path.join(temporary_directory, "model.joblib")
            joblib.dump({}, model_path)
            with open(
                os.path.join(temporary_directory, "manifest.json"),
                "w",
                encoding="utf-8",
            ) as output:
                json.dump({
                    "schema_version": MODEL_SCHEMA_VERSION,
                    "model_version": "test-v1",
                    "model_sha256": "0" * 64,
                    "runtime": {
                        "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__,
                    },
                }, output)

            with self.assertRaisesRegex(RuntimeError, "摘要"):
                MaterialQualityPredictor.load(temporary_directory)

    def test_bundled_model_replays_smoke_predictions(self):
        """仓库随附模型必须能在锁定依赖环境重放固定预测。"""
        model_dir = os.path.realpath(os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "mq-quality-v1-20260824-py38",
        ))
        if not all(os.path.isfile(os.path.join(model_dir, name)) for name in (
            "model.joblib", "smoke_test.npz",
        )):
            self.skipTest("公开仓库不附带模型权重和固定测试特征")
        predictor = MaterialQualityPredictor.load(model_dir)
        smoke_ok, maximum_error = predictor.verify_smoke_test()

        self.assertEqual(
            predictor.manifest["model_version"],
            "mq-quality-v1-20260824-py38",
        )
        self.assertTrue(smoke_ok)
        self.assertAlmostEqual(maximum_error, 0.0)

if __name__ == "__main__":
    unittest.main()

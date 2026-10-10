"""加载锁定模型包，并对画面和脚本特征进行晚融合评分。"""

import hashlib
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from material_quality.inference.feature_schema import (
    FEATURE_GROUP_RANGES,
    FEATURE_NAMES,
)


MODEL_SCHEMA_VERSION = "material_quality_serving_v1"
DEFAULT_FUSION_WEIGHTS = {"visual": 0.5, "script": 0.5}


def _file_sha256(path: str) -> str:
    """计算文件摘要，阻止模型和清单错配。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _as_feature_matrix(
    values: object,
    expected_dim: int,
    description: str,
) -> np.ndarray:
    """把单条或批量输入统一为二维有限浮点矩阵。"""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
        raise ValueError("{}维度异常，期望{}，实际{}".format(
            description,
            expected_dim,
            matrix.shape,
        ))
    if not np.isfinite(matrix).all():
        raise ValueError("{}包含非有限数值".format(description))
    return matrix


def build_serving_tower_matrices(
    base_features: object,
    script_features: object,
    script_dim: int,
) -> Dict[str, np.ndarray]:
    """按照训练时固定顺序构造画面和脚本两个子模型输入。"""
    base = _as_feature_matrix(base_features, len(FEATURE_NAMES), "多模态特征")
    script = _as_feature_matrix(script_features, int(script_dim), "脚本特征")
    if base.shape[0] != script.shape[0]:
        raise ValueError("多模态特征与脚本特征数量不一致")
    visual_start, visual_end = FEATURE_GROUP_RANGES["visual"]
    media_start, media_end = FEATURE_GROUP_RANGES["media"]
    visual = np.hstack([
        base[:, visual_start:visual_end],
        base[:, media_start:media_end],
    ]).astype(np.float32, copy=False)
    return {"visual": visual, "script": script}


class MaterialQualityPredictor:
    """提供版本校验、批量评分和训练分布百分位。"""

    def __init__(self, model_dir: str, package: dict, manifest: dict):
        self.model_dir = os.path.realpath(model_dir)
        self.package = package
        self.manifest = manifest
        self.models = package["models"]
        self.fusion_weights = package["fusion_weights"]
        self.script_feature_dim = int(package["script_feature_dim"])
        self.reference_scores = np.asarray(
            package["reference_scores"],
            dtype=np.float64,
        )

    @classmethod
    def load(
        cls,
        model_dir: str,
        strict_versions: bool = True,
    ) -> "MaterialQualityPredictor":
        """加载模型并校验文件摘要、结构版本和核心依赖版本。"""
        import joblib
        import sklearn

        resolved = os.path.realpath(model_dir)
        manifest_path = os.path.join(resolved, "manifest.json")
        model_path = os.path.join(resolved, "model.joblib")
        try:
            with open(manifest_path, "r", encoding="utf-8") as source:
                manifest = json.load(source)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("模型清单读取失败，原因：{}".format(exc))
        if manifest.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("模型结构版本不受支持")
        actual_sha256 = _file_sha256(model_path)
        if actual_sha256 != manifest.get("model_sha256"):
            raise RuntimeError("模型文件摘要与清单不一致")
        runtime = manifest.get("runtime") or {}
        if strict_versions:
            expected_numpy = str(runtime.get("numpy") or "")
            expected_sklearn = str(runtime.get("scikit_learn") or "")
            if np.__version__ != expected_numpy:
                raise RuntimeError("NumPy版本不一致，期望{}，实际{}".format(
                    expected_numpy,
                    np.__version__,
                ))
            if sklearn.__version__ != expected_sklearn:
                raise RuntimeError("scikit-learn版本不一致，期望{}，实际{}".format(
                    expected_sklearn,
                    sklearn.__version__,
                ))
        package = joblib.load(model_path)
        if package.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ValueError("模型包结构版本不受支持")
        required_models = set(package.get("fusion_weights") or {})
        if required_models != {"visual", "script"}:
            raise ValueError("正式模型必须包含画面和脚本两个子模型")
        if required_models != set(package.get("models") or {}):
            raise ValueError("模型包中的子模型与融合权重不一致")
        reference_scores = np.asarray(package.get("reference_scores"))
        if (
            reference_scores.ndim != 1
            or reference_scores.size == 0
            or not np.isfinite(reference_scores).all()
            or np.any(np.diff(reference_scores) < 0)
        ):
            raise ValueError("模型包中的训练分布参考分无效")
        return cls(resolved, package, manifest)

    def predict_arrays(
        self,
        base_features: object,
        script_features: object,
    ) -> Dict[str, np.ndarray]:
        """返回融合概率、训练分布百分位和两个子模型概率。"""
        towers = build_serving_tower_matrices(
            base_features,
            script_features,
            self.script_feature_dim,
        )
        tower_probabilities = {}
        combined = np.zeros(towers["visual"].shape[0], dtype=np.float64)
        total_weight = float(sum(self.fusion_weights.values()))
        if total_weight <= 0:
            raise ValueError("模型融合权重总和必须大于0")
        for tower_name, weight in self.fusion_weights.items():
            probability = self.models[tower_name].predict_proba(
                towers[tower_name]
            )[:, 1]
            tower_probabilities[tower_name] = probability
            combined += probability * (float(weight) / total_weight)
        combined = np.clip(combined, 0.0, 1.0)
        ranks = np.searchsorted(
            self.reference_scores,
            combined,
            side="right",
        )
        percentiles = ranks.astype(np.float64) / float(self.reference_scores.size)
        quality_scores = np.clip(percentiles * 100.0, 0.0, 100.0)
        levels = np.where(
            quality_scores >= 90.0,
            "优",
            np.where(quality_scores >= 70.0, "高", np.where(
                quality_scores >= 40.0,
                "中",
                "低",
            )),
        )
        return {
            "positive_probability": combined,
            "quality_percentile": percentiles,
            "quality_score": quality_scores,
            "quality_level": levels,
            "visual_probability": tower_probabilities["visual"],
            "script_probability": tower_probabilities["script"],
        }

    def predict_one(
        self,
        base_features: object,
        script_features: object,
        content_sha256: Optional[str] = None,
    ) -> dict:
        """以便于API序列化的字典返回单条素材评分。"""
        predictions = self.predict_arrays(base_features, script_features)
        if predictions["positive_probability"].size != 1:
            raise ValueError("单条评分接口只接受一条素材")
        result = {
            "model_version": self.manifest["model_version"],
            "quality_score": round(float(predictions["quality_score"][0]), 4),
            "quality_level": str(predictions["quality_level"][0]),
            "positive_probability": round(
                float(predictions["positive_probability"][0]),
                8,
            ),
            "quality_percentile": round(
                float(predictions["quality_percentile"][0]),
                8,
            ),
            "visual_probability": round(
                float(predictions["visual_probability"][0]),
                8,
            ),
            "script_probability": round(
                float(predictions["script_probability"][0]),
                8,
            ),
        }
        if content_sha256:
            result["content_sha256"] = str(content_sha256)
        return result

    def verify_smoke_test(self, tolerance: float = 1e-7) -> Tuple[bool, float]:
        """用模型包随附样例验证序列化前后预测完全一致。"""
        smoke_path = os.path.join(self.model_dir, "smoke_test.npz")
        with np.load(smoke_path, allow_pickle=False) as source:
            predictions = self.predict_arrays(
                source["base_features"],
                source["script_features"],
            )
            expected = np.asarray(source["positive_probability"], dtype=np.float64)
        maximum_error = float(np.max(np.abs(
            predictions["positive_probability"] - expected
        )))
        return maximum_error <= float(tolerance), maximum_error

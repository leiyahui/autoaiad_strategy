"""从内容哈希档案读取正式模型所需的两类输入特征。"""

import json
import os
from typing import Tuple

import numpy as np

from material_quality.feature.script_agent_features import script_feature_paths
from material_quality.inference.feature_schema import load_local_semantic_feature


def load_script_feature_vector(
    script_feature_root: str,
    script_feature_version: str,
    content_sha256: str,
) -> Tuple[np.ndarray, dict]:
    """读取并严格校验一条内容级脚本向量。"""
    content_hash = str(content_sha256 or "").lower()
    paths = script_feature_paths(
        script_feature_root,
        script_feature_version,
        content_hash,
    )
    if not os.path.isfile(paths["manifest"]) or not os.path.isfile(paths["vectors"]):
        raise FileNotFoundError("脚本特征不存在：{}".format(content_hash))
    try:
        with open(paths["manifest"], "r", encoding="utf-8") as source:
            manifest = json.load(source)
        with np.load(paths["vectors"], allow_pickle=False) as vectors:
            embedding = np.asarray(
                vectors["script_embedding"],
                dtype=np.float32,
            )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("脚本特征读取失败，内容{}，原因：{}".format(
            content_hash,
            exc,
        ))
    if manifest.get("content_sha256") != content_hash:
        raise ValueError("脚本特征内容哈希不一致：{}".format(content_hash))
    if manifest.get("script_feature_version") != script_feature_version:
        raise ValueError("脚本特征版本不一致：{}".format(content_hash))
    if embedding.ndim != 1 or embedding.size == 0:
        raise ValueError("脚本向量维度无效：{}".format(content_hash))
    if not np.isfinite(embedding).all():
        raise ValueError("脚本向量包含非有限数值：{}".format(content_hash))
    if int(manifest.get("embedding_dim") or 0) != int(embedding.size):
        raise ValueError("脚本向量维度与清单不一致：{}".format(content_hash))
    return embedding, manifest


def load_content_features(
    content_sha256: str,
    raw_feature_root: str,
    raw_feature_version: str,
    script_feature_root: str,
    script_feature_version: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """读取一条素材的多模态特征和版本化脚本向量。"""
    base_feature, raw_manifest = load_local_semantic_feature(
        raw_feature_root,
        raw_feature_version,
        content_sha256,
    )
    script_feature, script_manifest = load_script_feature_vector(
        script_feature_root,
        script_feature_version,
        content_sha256,
    )
    metadata = {
        "content_sha256": str(content_sha256).lower(),
        "raw_feature_version": raw_feature_version,
        "script_feature_version": script_feature_version,
        "script_model": script_manifest.get("script_model"),
        "embedding_model": script_manifest.get("embedding_model"),
        "duration": raw_manifest.get("duration"),
    }
    return base_feature, script_feature, metadata

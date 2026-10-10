"""把内容级脚本Sidecar与既有冻结快照按内容哈希严格对齐。"""

import hashlib
import os
from typing import Dict, Tuple

import numpy as np

from material_quality.feature.script_agent_features import script_feature_paths
from material_quality.inference.features import load_script_feature_vector
from material_quality.training.semantic_dataset import load_semantic_snapshot


def load_script_augmented_dataset(
    base_snapshot_dir: str,
    script_feature_root: str,
    script_feature_version: str,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """要求成熟样本全部存在脚本向量，避免因API成败改变测试人群。"""
    dataset, base_manifest = load_semantic_snapshot(base_snapshot_dir)
    embeddings = []
    missing_hashes = []
    embedding_dim = None
    prompt_hashes = set()
    script_models = set()
    embedding_models = set()
    for content_sha256 in dataset["content_sha256"].tolist():
        content_hash = str(content_sha256)
        paths = script_feature_paths(
            script_feature_root,
            script_feature_version,
            content_hash,
        )
        if not os.path.isfile(paths["manifest"]) or not os.path.isfile(paths["vectors"]):
            missing_hashes.append(content_hash)
            continue
        try:
            embedding, script_manifest = load_script_feature_vector(
                script_feature_root,
                script_feature_version,
                content_hash,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError("脚本特征读取失败，内容{}，原因：{}".format(
                content_hash,
                exc,
            ))
        if embedding_dim is None:
            embedding_dim = int(embedding.size)
        elif embedding.size != embedding_dim:
            raise ValueError("不同素材的脚本向量维度不一致")
        embeddings.append(embedding)
        prompt_hashes.add(str((script_manifest.get("audit") or {}).get("prompt_sha256") or ""))
        script_models.add(str(script_manifest.get("script_model") or ""))
        embedding_models.add(str(script_manifest.get("embedding_model") or ""))
    if missing_hashes:
        raise RuntimeError("成熟样本缺少脚本特征{}条，示例：{}".format(
            len(missing_hashes),
            ",".join(missing_hashes[:5]),
        ))
    script_features = np.vstack(embeddings).astype(np.float32)
    augmented = dict(dataset)
    augmented["script_features"] = script_features
    digest = hashlib.sha256()
    digest.update(str(base_manifest["dataset_sha256"]).encode("utf-8"))
    digest.update(script_feature_version.encode("utf-8"))
    digest.update(np.ascontiguousarray(script_features).tobytes())
    manifest = dict(base_manifest)
    manifest.update({
        "base_dataset_sha256": base_manifest["dataset_sha256"],
        "script_feature_version": script_feature_version,
        "script_embedding_dim": int(script_features.shape[1]),
        "script_coverage_count": int(script_features.shape[0]),
        "script_models": sorted(script_models),
        "script_embedding_models": sorted(embedding_models),
        "script_prompt_sha256": sorted(prompt_hashes),
        "script_dataset_sha256": digest.hexdigest(),
    })
    return augmented, manifest

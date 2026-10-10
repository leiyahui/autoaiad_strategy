-- 素材质量特征流水线数据表
-- 原始视频仅用于临时处理，表中持久化媒体信息、低阶视觉向量和版本信息。

USE qianchuan_db;

CREATE TABLE IF NOT EXISTS `material_quality_feature` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `asset_id` BIGINT NOT NULL,
  `feature_version` VARCHAR(64) NOT NULL,
  `content_sha256` VARCHAR(64) NOT NULL,
  `duration` DOUBLE NOT NULL DEFAULT 0,
  `width` INT NOT NULL DEFAULT 0,
  `height` INT NOT NULL DEFAULT 0,
  `fps` DOUBLE NOT NULL DEFAULT 0,
  `bit_rate` BIGINT NOT NULL DEFAULT 0,
  `file_size` BIGINT NOT NULL DEFAULT 0,
  `has_audio` TINYINT(1) NOT NULL DEFAULT 0,
  `sample_frame_count` INT NOT NULL DEFAULT 0,
  `visual_vector` MEDIUMBLOB NOT NULL,
  `visual_vector_dim` INT NOT NULL DEFAULT 0,
  `visual_vector_dtype` VARCHAR(16) NOT NULL DEFAULT 'float32',
  `probe_json` JSON NULL,
  `extraction_seconds` DOUBLE NOT NULL DEFAULT 0,
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_material_quality_feature_version` (`asset_id`, `feature_version`),
  KEY `idx_material_quality_feature_version` (`feature_version`, `asset_id`),
  CONSTRAINT `fk_material_quality_feature_asset`
    FOREIGN KEY (`asset_id`) REFERENCES `material_quality_asset` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='素材质量模型的版本化特征';

-- 素材质量多模态语义特征表
-- 原始视频和关键帧只在临时目录中处理，完成后立即删除。

USE qianchuan_db;

CREATE TABLE IF NOT EXISTS `material_quality_semantic_feature` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `asset_id` BIGINT NOT NULL,
  `feature_version` VARCHAR(64) NOT NULL,
  `content_sha256` VARCHAR(64) NOT NULL,
  `visual_model_name` VARCHAR(255) NOT NULL,
  `visual_vector` MEDIUMBLOB NOT NULL,
  `visual_vector_dim` INT NOT NULL DEFAULT 0,
  `visual_vector_dtype` VARCHAR(16) NOT NULL DEFAULT 'float32',
  `visual_temporal_vector` BLOB NULL,
  `visual_temporal_dim` INT NOT NULL DEFAULT 0,
  `text_model_name` VARCHAR(255) NOT NULL,
  `text_vector` MEDIUMBLOB NULL,
  `text_vector_dim` INT NOT NULL DEFAULT 0,
  `text_vector_dtype` VARCHAR(16) NOT NULL DEFAULT 'float32',
  `ocr_engine_name` VARCHAR(128) NOT NULL,
  `ocr_text` MEDIUMTEXT NULL,
  `ocr_json` JSON NULL,
  `ocr_stats_vector` BLOB NOT NULL,
  `ocr_stats_dim` INT NOT NULL DEFAULT 0,
  `asr_model_name` VARCHAR(255) NULL,
  `asr_language` VARCHAR(32) NULL,
  `asr_language_probability` DOUBLE NOT NULL DEFAULT 0,
  `asr_text` MEDIUMTEXT NULL,
  `asr_json` JSON NULL,
  `asr_text_vector` MEDIUMBLOB NULL,
  `asr_text_vector_dim` INT NOT NULL DEFAULT 0,
  `asr_text_vector_dtype` VARCHAR(16) NOT NULL DEFAULT 'float32',
  `audio_stats_vector` BLOB NULL,
  `audio_stats_dim` INT NOT NULL DEFAULT 0,
  `duration` DOUBLE NOT NULL DEFAULT 0,
  `width` INT NOT NULL DEFAULT 0,
  `height` INT NOT NULL DEFAULT 0,
  `fps` DOUBLE NOT NULL DEFAULT 0,
  `bit_rate` BIGINT NOT NULL DEFAULT 0,
  `file_size` BIGINT NOT NULL DEFAULT 0,
  `has_audio` TINYINT(1) NOT NULL DEFAULT 0,
  `sample_frame_count` INT NOT NULL DEFAULT 0,
  `probe_json` JSON NULL,
  `extractor_config_json` JSON NULL,
  `extraction_seconds` DOUBLE NOT NULL DEFAULT 0,
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_material_quality_semantic_feature_version` (`asset_id`, `feature_version`),
  KEY `idx_material_quality_semantic_feature_version` (`feature_version`, `asset_id`),
  CONSTRAINT `fk_material_quality_semantic_feature_asset`
    FOREIGN KEY (`asset_id`) REFERENCES `material_quality_asset` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='素材质量模型的版本化多模态语义特征';

CREATE TABLE IF NOT EXISTS `material_quality_semantic_task` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `asset_id` BIGINT NOT NULL,
  `feature_version` VARCHAR(64) NOT NULL,
  `status` VARCHAR(32) NOT NULL DEFAULT 'processing',
  `last_phase` VARCHAR(32) NULL,
  `attempt_count` INT NOT NULL DEFAULT 0,
  `last_error` TEXT NULL,
  `lease_until` DATETIME NULL,
  `last_attempt_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_material_quality_semantic_task_version` (`asset_id`, `feature_version`),
  KEY `idx_material_quality_semantic_task_status` (`feature_version`, `status`, `lease_until`),
  CONSTRAINT `fk_material_quality_semantic_task_asset`
    FOREIGN KEY (`asset_id`) REFERENCES `material_quality_asset` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='素材质量语义特征的版本化尝试和失败状态';

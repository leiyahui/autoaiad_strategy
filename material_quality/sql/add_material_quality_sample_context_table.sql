-- 为存量素材质量样本补充行业归属与训练分层上下文。

USE qianchuan_db;

CREATE TABLE IF NOT EXISTS `material_quality_sample_context` (
  `sample_id` BIGINT NOT NULL,
  `cohort` VARCHAR(32) NOT NULL,
  `material_id` VARCHAR(100) NOT NULL,
  `label_version` VARCHAR(64) NOT NULL,
  `primary_industry` VARCHAR(100) NOT NULL DEFAULT '未标注',
  `industry_count` INT NOT NULL DEFAULT 0,
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`sample_id`),
  KEY `idx_material_quality_context_scope` (
    `cohort`, `label_version`, `primary_industry`
  ),
  CONSTRAINT `fk_material_quality_context_sample`
    FOREIGN KEY (`sample_id`) REFERENCES `material_quality_sample` (`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='素材质量样本的行业归属与训练分层上下文';

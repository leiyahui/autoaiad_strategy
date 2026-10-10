-- 素材质量样本流水线数据表
-- 第一版按素材首次产生有效投放数据的自然日开始，聚合前7个自然日的GMV作为主标签。

USE qianchuan_db;

CREATE TABLE IF NOT EXISTS `material_quality_daily_stat` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `cohort` VARCHAR(32) NOT NULL COMMENT '样本范围版本，例如food_v1',
  `account_plan_id` INT NOT NULL,
  `material_id` VARCHAR(100) NOT NULL,
  `stat_date` DATE NOT NULL,
  `gmv` DOUBLE NOT NULL DEFAULT 0,
  `cost` DOUBLE NOT NULL DEFAULT 0,
  `orders` DOUBLE NOT NULL DEFAULT 0,
  `send` DOUBLE NOT NULL DEFAULT 0,
  `click` DOUBLE NOT NULL DEFAULT 0,
  `source_row_count` INT NOT NULL DEFAULT 0,
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_material_quality_daily` (`cohort`, `account_plan_id`, `material_id`, `stat_date`),
  KEY `idx_material_quality_daily_material` (`cohort`, `material_id`, `stat_date`),
  KEY `idx_material_quality_daily_plan` (`account_plan_id`, `cohort`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='素材质量样本的按日中间汇总数据';

CREATE TABLE IF NOT EXISTS `material_quality_asset` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `material_id` VARCHAR(100) NOT NULL,
  `material_for_edit_id` INT NULL,
  `user_id` INT NULL,
  `advertiser_id` VARCHAR(32) NULL,
  `material_name` VARCHAR(255) NOT NULL DEFAULT '',
  `video_url` TEXT NULL,
  `thumbnail_url` TEXT NULL,
  `duration` DOUBLE NOT NULL DEFAULT 0,
  `file_size` BIGINT NOT NULL DEFAULT 0,
  `source_status` VARCHAR(32) NOT NULL DEFAULT 'pending_url' COMMENT 'pending_url/url_ready/unavailable',
  `download_status` VARCHAR(32) NOT NULL DEFAULT 'pending',
  `feature_status` VARCHAR(32) NOT NULL DEFAULT 'pending',
  `content_sha256` VARCHAR(64) NULL,
  `feature_version` VARCHAR(64) NULL,
  `attempt_count` INT NOT NULL DEFAULT 0,
  `lease_until` DATETIME NULL,
  `last_error` TEXT NULL,
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_material_quality_asset_material` (`material_id`),
  KEY `idx_material_quality_asset_source_status` (`source_status`, `id`),
  KEY `idx_material_quality_asset_feature_status` (`feature_status`, `id`),
  KEY `idx_material_quality_asset_lease` (`lease_until`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='素材质量样本的媒体资产与后续特征处理状态';

CREATE TABLE IF NOT EXISTS `material_quality_sample` (
  `id` BIGINT NOT NULL AUTO_INCREMENT,
  `asset_id` BIGINT NOT NULL,
  `cohort` VARCHAR(32) NOT NULL,
  `material_id` VARCHAR(100) NOT NULL,
  `label_version` VARCHAR(64) NOT NULL,
  `window_days` INT NOT NULL DEFAULT 7,
  `first_stat_date` DATE NOT NULL,
  `window_end_date` DATE NOT NULL COMMENT '左闭右开窗口的结束日期',
  `label_gmv_7d` DOUBLE NOT NULL DEFAULT 0,
  `label_log_gmv_7d` DOUBLE NOT NULL DEFAULT 0,
  `label_gmv_per_delivery_day` DOUBLE NOT NULL DEFAULT 0,
  `label_log_gmv_per_delivery_day` DOUBLE NOT NULL DEFAULT 0,
  `cost_7d` DOUBLE NOT NULL DEFAULT 0,
  `orders_7d` DOUBLE NOT NULL DEFAULT 0,
  `send_7d` DOUBLE NOT NULL DEFAULT 0,
  `click_7d` DOUBLE NOT NULL DEFAULT 0,
  `plan_count` INT NOT NULL DEFAULT 0,
  `source_day_count` INT NOT NULL DEFAULT 0,
  `delivery_day_count` INT NOT NULL DEFAULT 0,
  `is_window_closed` TINYINT(1) NOT NULL DEFAULT 0,
  `label_status` VARCHAR(32) NOT NULL DEFAULT 'window_open' COMMENT 'window_open/ready/invalid_negative_gmv/insufficient_delivery_days',
  `create_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_material_quality_sample` (`cohort`, `material_id`, `label_version`),
  KEY `idx_material_quality_sample_ready` (`cohort`, `label_version`, `label_status`),
  KEY `idx_material_quality_sample_asset` (`asset_id`),
  CONSTRAINT `fk_material_quality_sample_asset`
    FOREIGN KEY (`asset_id`) REFERENCES `material_quality_asset` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='一条素材一条记录的7日GMV质量样本';

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

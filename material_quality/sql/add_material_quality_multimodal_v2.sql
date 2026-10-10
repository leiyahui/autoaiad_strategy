-- 为素材质量样本补充按实际投放天数归一化的标签，并扩展一次性多模态特征字段。

USE qianchuan_db;

ALTER TABLE `material_quality_sample`
  ADD COLUMN `label_gmv_per_delivery_day` DOUBLE NOT NULL DEFAULT 0
    AFTER `label_log_gmv_7d`,
  ADD COLUMN `label_log_gmv_per_delivery_day` DOUBLE NOT NULL DEFAULT 0
    AFTER `label_gmv_per_delivery_day`,
  ADD COLUMN `delivery_day_count` INT NOT NULL DEFAULT 0
    AFTER `source_day_count`;

ALTER TABLE `material_quality_semantic_feature`
  ADD COLUMN `visual_temporal_vector` BLOB NULL
    AFTER `visual_vector_dtype`,
  ADD COLUMN `visual_temporal_dim` INT NOT NULL DEFAULT 0
    AFTER `visual_temporal_vector`,
  ADD COLUMN `asr_model_name` VARCHAR(255) NULL
    AFTER `ocr_stats_dim`,
  ADD COLUMN `asr_language` VARCHAR(32) NULL
    AFTER `asr_model_name`,
  ADD COLUMN `asr_language_probability` DOUBLE NOT NULL DEFAULT 0
    AFTER `asr_language`,
  ADD COLUMN `asr_text` MEDIUMTEXT NULL
    AFTER `asr_language_probability`,
  ADD COLUMN `asr_json` JSON NULL
    AFTER `asr_text`,
  ADD COLUMN `asr_text_vector` MEDIUMBLOB NULL
    AFTER `asr_json`,
  ADD COLUMN `asr_text_vector_dim` INT NOT NULL DEFAULT 0
    AFTER `asr_text_vector`,
  ADD COLUMN `asr_text_vector_dtype` VARCHAR(16) NOT NULL DEFAULT 'float32'
    AFTER `asr_text_vector_dim`,
  ADD COLUMN `audio_stats_vector` BLOB NULL
    AFTER `asr_text_vector_dtype`,
  ADD COLUMN `audio_stats_dim` INT NOT NULL DEFAULT 0
    AFTER `audio_stats_vector`,
  ADD COLUMN `extractor_config_json` JSON NULL
    AFTER `probe_json`;

UPDATE `material_quality_sample` AS s
INNER JOIN (
  SELECT
    sample.id AS sample_id,
    COUNT(DISTINCT daily.stat_date) AS delivery_day_count
  FROM `material_quality_sample` AS sample
  INNER JOIN `material_quality_daily_stat` AS daily
    ON daily.cohort = sample.cohort
   AND daily.material_id = sample.material_id
   AND daily.stat_date >= sample.first_stat_date
   AND daily.stat_date < sample.window_end_date
  GROUP BY sample.id
) AS counts ON counts.sample_id = s.id
SET
  s.delivery_day_count = counts.delivery_day_count,
  s.label_gmv_per_delivery_day =
    s.label_gmv_7d / GREATEST(counts.delivery_day_count, 1),
  s.label_log_gmv_per_delivery_day = LN(1 + GREATEST(
    s.label_gmv_7d / GREATEST(counts.delivery_day_count, 1),
    0
  )),
  s.label_status = CASE
    WHEN s.is_window_closed = 0 THEN 'window_open'
    WHEN s.label_gmv_7d < 0 THEN 'invalid_negative_gmv'
    WHEN counts.delivery_day_count < 2 THEN 'insufficient_delivery_days'
    ELSE 'ready'
  END,
  s.label_version = 'gmv_per_delivery_day_7d_v2',
  s.update_time = NOW();

UPDATE `material_quality_sample_context` AS context
INNER JOIN `material_quality_sample` AS sample ON sample.id = context.sample_id
SET
  context.label_version = sample.label_version,
  context.update_time = NOW();

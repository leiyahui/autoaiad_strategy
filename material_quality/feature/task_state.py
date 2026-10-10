"""素材质量特征任务的数据库状态更新。"""

from sqlalchemy import text

from material_quality.database import engine


def mark_url_success(asset_id: int, detail: dict) -> None:
    """保存刷新后的地址和媒体元数据。"""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE material_quality_asset
            SET user_id = :user_id,
                advertiser_id = :advertiser_id,
                material_name = :material_name,
                video_url = :video_url,
                thumbnail_url = :thumbnail_url,
                duration = :duration,
                file_size = :file_size,
                source_status = 'url_ready',
                last_error = NULL,
                update_time = NOW()
            WHERE id = :asset_id
        """), {
            "asset_id": asset_id,
            "user_id": detail.get("user_id"),
            "advertiser_id": detail.get("advertiser_id"),
            "material_name": str(detail.get("filename") or detail.get("name") or "")[:255],
            "video_url": detail.get("url"),
            "thumbnail_url": detail.get("poster_url"),
            "duration": float(detail.get("duration") or 0),
            "file_size": int(detail.get("size") or 0),
        })


def mark_processing(asset_id: int) -> None:
    """标记素材已被当前离线任务领取。"""
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE material_quality_asset
            SET download_status = 'downloading',
                feature_status = 'processing',
                lease_until = DATE_ADD(NOW(), INTERVAL 30 MINUTE),
                attempt_count = attempt_count + 1,
                last_error = NULL,
                update_time = NOW()
            WHERE id = :asset_id
        """), {"asset_id": asset_id})


def mark_failure(asset_id: int, phase: str, error: str) -> None:
    """记录可重试错误并释放任务租约。"""
    with engine.begin() as conn:
        if phase == "url":
            retryable_markers = (
                "接口临时限流",
                "接口临时异常",
                "接口异常",
                "token_refresh_failed",
                "token_expiring",
                "access_token_missing",
            )
            source_status = (
                "pending_url"
                if any(marker in error for marker in retryable_markers)
                else "unavailable"
            )
            conn.execute(text("""
                UPDATE material_quality_asset
                SET source_status = :source_status,
                    last_error = :last_error,
                    lease_until = NULL,
                    update_time = NOW()
                WHERE id = :asset_id
            """), {
                "asset_id": asset_id,
                "source_status": source_status,
                "last_error": error[-2000:],
            })
        else:
            conn.execute(text("""
                UPDATE material_quality_asset
                SET download_status = 'pending',
                    feature_status = 'retry',
                    last_error = :last_error,
                    lease_until = NULL,
                    update_time = NOW()
                WHERE id = :asset_id
            """), {"asset_id": asset_id, "last_error": error[-2000:]})

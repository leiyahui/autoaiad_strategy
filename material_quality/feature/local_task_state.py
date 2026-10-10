"""本地素材特征任务状态，避免离线抽取写入生产数据库。"""

import os
import sqlite3
import time
from typing import Dict, Iterable, List, Optional


class LocalTaskState:
    """使用SQLite原子领取任务并记录本地重试状态。"""

    def __init__(self, database_path: str):
        self.database_path = os.path.realpath(database_path)
        directory = os.path.dirname(self.database_path)
        if not directory:
            raise ValueError("本地任务数据库目录不能为空")
        os.makedirs(directory, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """创建启用WAL和忙等待的短连接，支持多个抽取进程。"""
        connection = sqlite3.connect(self.database_path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def _initialize(self) -> None:
        """幂等创建本地任务表和常用索引。"""
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS material_feature_task (
                    asset_id INTEGER NOT NULL,
                    feature_version TEXT NOT NULL,
                    material_id TEXT NOT NULL,
                    cohort TEXT NOT NULL,
                    label_status TEXT,
                    first_stat_date TEXT,
                    primary_industry TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    phase TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    content_sha256 TEXT,
                    artifact_path TEXT,
                    next_retry_at REAL,
                    lease_until REAL,
                    create_time REAL NOT NULL,
                    update_time REAL NOT NULL,
                    PRIMARY KEY (asset_id, feature_version)
                );
                CREATE INDEX IF NOT EXISTS idx_material_feature_task_status
                ON material_feature_task(feature_version, status, next_retry_at);
                CREATE INDEX IF NOT EXISTS idx_material_feature_task_material
                ON material_feature_task(material_id, feature_version);
            """)

    def upsert_targets(
        self,
        targets: Iterable[Dict[str, object]],
        feature_version: str,
        cohort: str,
    ) -> int:
        """写入候选清单，但不重置已完成和失败状态。"""
        now = time.time()
        rows = []
        for target in targets:
            rows.append({
                "asset_id": int(target["asset_id"]),
                "feature_version": feature_version,
                "material_id": str(target.get("material_id") or ""),
                "cohort": cohort,
                "label_status": str(target.get("label_status") or ""),
                "first_stat_date": str(target.get("first_stat_date") or ""),
                "primary_industry": str(target.get("primary_industry") or ""),
                "create_time": now,
                "update_time": now,
            })
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany("""
                INSERT INTO material_feature_task (
                    asset_id, feature_version, material_id, cohort,
                    label_status, first_stat_date, primary_industry,
                    status, create_time, update_time
                ) VALUES (
                    :asset_id, :feature_version, :material_id, :cohort,
                    :label_status, :first_stat_date, :primary_industry,
                    'pending', :create_time, :update_time
                )
                ON CONFLICT(asset_id, feature_version) DO UPDATE SET
                    material_id = excluded.material_id,
                    cohort = excluded.cohort,
                    label_status = excluded.label_status,
                    first_stat_date = excluded.first_stat_date,
                    primary_industry = excluded.primary_industry,
                    update_time = excluded.update_time
            """, rows)
        return len(rows)

    def defer_assets(
        self,
        asset_errors: Dict[int, str],
        feature_version: str,
    ) -> int:
        """按账户级授权结果前置延后素材，不把它记为永久不可用。"""
        if not asset_errors:
            return 0
        now = time.time()
        rows = [{
            "asset_id": int(asset_id),
            "feature_version": feature_version,
            "last_error": str(error or "账户授权暂不可用")[-2000:],
            "update_time": now,
        } for asset_id, error in asset_errors.items()]
        with self._connect() as connection:
            cursor = connection.executemany("""
                UPDATE material_feature_task
                SET status = 'url_deferred',
                    phase = 'auth',
                    last_error = :last_error,
                    lease_until = NULL,
                    next_retry_at = NULL,
                    update_time = :update_time
                WHERE asset_id = :asset_id
                  AND feature_version = :feature_version
                  AND status NOT IN ('ready', 'url_unavailable')
            """, rows)
        return int(cursor.rowcount or 0)

    def reactivate_assets(
        self,
        asset_ids: Iterable[int],
        feature_version: str,
    ) -> int:
        """授权恢复后把延后任务重新放回待处理队列。"""
        now = time.time()
        rows = [{
            "asset_id": int(asset_id),
            "feature_version": feature_version,
            "update_time": now,
        } for asset_id in asset_ids]
        if not rows:
            return 0
        with self._connect() as connection:
            cursor = connection.executemany("""
                UPDATE material_feature_task
                SET status = 'pending',
                    phase = NULL,
                    last_error = NULL,
                    next_retry_at = NULL,
                    update_time = :update_time
                WHERE asset_id = :asset_id
                  AND feature_version = :feature_version
                  AND status IN ('url_deferred', 'url_retry')
            """, rows)
        return int(cursor.rowcount or 0)

    def claim(
        self,
        asset_id: int,
        feature_version: str,
        lease_seconds: int = 3600,
    ) -> bool:
        """原子领取一条任务，崩溃后的过期租约允许其他进程接管。"""
        now = time.time()
        connection = self._connect()
        try:
            connection.isolation_level = None
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute("""
                UPDATE material_feature_task
                SET status = 'processing',
                    phase = 'url',
                    attempt_count = attempt_count + 1,
                    last_error = NULL,
                    next_retry_at = NULL,
                    lease_until = :lease_until,
                    update_time = :update_time
                WHERE asset_id = :asset_id
                  AND feature_version = :feature_version
                  AND (
                      status = 'pending'
                      OR (
                          status IN ('url_retry', 'feature_retry')
                          AND COALESCE(next_retry_at, 0) <= :update_time
                      )
                      OR (
                          status = 'processing'
                          AND COALESCE(lease_until, 0) <= :update_time
                      )
                  )
            """, {
                "asset_id": int(asset_id),
                "feature_version": feature_version,
                "lease_until": now + max(int(lease_seconds), 60),
                "update_time": now,
            })
            connection.execute("COMMIT")
            return int(cursor.rowcount or 0) == 1
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def filter_unfinished_targets(
        self,
        targets: Iterable[Dict[str, object]],
        feature_version: str,
        include_unavailable: bool = False,
    ) -> List[Dict[str, object]]:
        """过滤本地已完成任务，避免每次重新检查相同账户和素材。"""
        target_list = list(targets)
        if not target_list:
            return []
        asset_ids = [int(target["asset_id"]) for target in target_list]
        placeholders = ",".join("?" for _ in asset_ids)
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT asset_id, status
                FROM material_feature_task
                WHERE feature_version = ?
                  AND asset_id IN ({})
            """.format(placeholders), [feature_version] + asset_ids).fetchall()
        status_by_asset = {
            int(row["asset_id"]): str(row["status"])
            for row in rows
        }
        excluded_statuses = {"ready"}
        if not include_unavailable:
            excluded_statuses.add("url_unavailable")
        return [
            target for target in target_list
            if status_by_asset.get(int(target["asset_id"])) not in excluded_statuses
        ]

    def mark_ready(
        self,
        asset_id: int,
        feature_version: str,
        content_sha256: str,
        artifact_path: str,
    ) -> None:
        """记录本地特征完成位置并释放租约。"""
        with self._connect() as connection:
            connection.execute("""
                UPDATE material_feature_task
                SET status = 'ready',
                    phase = 'complete',
                    last_error = NULL,
                    content_sha256 = :content_sha256,
                    artifact_path = :artifact_path,
                    next_retry_at = NULL,
                    lease_until = NULL,
                    update_time = :update_time
                WHERE asset_id = :asset_id AND feature_version = :feature_version
            """, {
                "asset_id": int(asset_id),
                "feature_version": feature_version,
                "content_sha256": content_sha256,
                "artifact_path": artifact_path,
                "update_time": time.time(),
            })

    def mark_failure(
        self,
        asset_id: int,
        feature_version: str,
        phase: str,
        error: str,
        retryable: bool,
        retry_delay_seconds: int = 1800,
    ) -> None:
        """区分临时重试和永久URL不可用，防止无效循环。"""
        if phase == "url":
            status = "url_retry" if retryable else "url_unavailable"
        else:
            status = "feature_retry"
        now = time.time()
        next_retry_at: Optional[float] = None
        if retryable or phase != "url":
            next_retry_at = now + max(int(retry_delay_seconds), 60)
        with self._connect() as connection:
            connection.execute("""
                UPDATE material_feature_task
                SET status = :status,
                    phase = :phase,
                    last_error = :last_error,
                    next_retry_at = :next_retry_at,
                    lease_until = NULL,
                    update_time = :update_time
                WHERE asset_id = :asset_id AND feature_version = :feature_version
            """, {
                "asset_id": int(asset_id),
                "feature_version": feature_version,
                "status": status,
                "phase": phase,
                "last_error": str(error or "未知错误")[-2000:],
                "next_retry_at": next_retry_at,
                "update_time": now,
            })

    def get(self, asset_id: int, feature_version: str) -> Optional[Dict[str, object]]:
        """读取单条任务，供测试和审计使用。"""
        with self._connect() as connection:
            row = connection.execute("""
                SELECT * FROM material_feature_task
                WHERE asset_id = ? AND feature_version = ?
            """, (int(asset_id), feature_version)).fetchone()
        return dict(row) if row else None

    def summary(self, feature_version: str) -> List[Dict[str, object]]:
        """返回本地各状态数量。"""
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT status, COUNT(*) AS task_count
                FROM material_feature_task
                WHERE feature_version = ?
                GROUP BY status
                ORDER BY status
            """, (feature_version,)).fetchall()
        return [dict(row) for row in rows]

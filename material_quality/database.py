"""素材质量离线任务的隔离数据库连接。"""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


def _build_engine():
    """仅从独立配置创建连接，避免训练仓库依赖业务后端的数据库模块。"""
    database_url = os.getenv("MATERIAL_QUALITY_DATABASE_URL", "").strip()
    if not database_url:
        return create_engine("sqlite:///:memory:")

    resolved_url = make_url(database_url)
    if resolved_url.get_backend_name() == "sqlite":
        return create_engine(resolved_url)

    return create_engine(
        resolved_url,
        echo=False,
        pool_size=4,
        max_overflow=2,
        pool_timeout=20,
        pool_recycle=300,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 10,
            "read_timeout": 120,
            "write_timeout": 120,
            "charset": "utf8mb4",
        },
    )


engine = _build_engine()

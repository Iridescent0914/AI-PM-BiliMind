"""
Bilibili RAG 知识库系统

数据库管理模块
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from contextlib import asynccontextmanager
from app.config import settings
from app.models import Base
import os


# 确保数据目录存在
os.makedirs("data", exist_ok=True)

# 创建异步引擎
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True
)

# 创建异步会话工厂
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """初始化数据库（创建表）"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_video_cache(conn)


async def _migrate_video_cache(conn) -> None:
    """轻量补齐旧库缺失字段，避免手动迁移。"""
    result = await conn.execute(text("PRAGMA table_info(video_cache)"))
    columns = {row[1] for row in result.fetchall()}

    required_columns = {
        "transcript_json": "ALTER TABLE video_cache ADD COLUMN transcript_json JSON",
        "summary_json": "ALTER TABLE video_cache ADD COLUMN summary_json JSON",
        "knowledge_graph_json": "ALTER TABLE video_cache ADD COLUMN knowledge_graph_json JSON",
    }

    for column, sql in required_columns.items():
        if column not in columns:
            await conn.execute(text(sql))


async def get_db() -> AsyncSession:
    """获取数据库会话（用于 FastAPI 依赖注入）"""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """获取数据库会话（用于上下文管理器）"""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

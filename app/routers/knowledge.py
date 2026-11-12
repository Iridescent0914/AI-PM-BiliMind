"""
Bilibili RAG 知识库系统

知识库路由 - 构建和管理知识库
"""
from datetime import datetime
import re
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from loguru import logger
from typing import List, Optional, Callable
from pydantic import BaseModel
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, get_db_context
from app.models import FavoriteFolder, FavoriteVideo, VideoCache, UserSession, ContentSource, VideoContent
from app.services.bilibili import BilibiliService
from app.services.content_fetcher import ContentFetcher
from app.services.asr import ASRService
from app.services.rag import RAGService
from app.routers.auth import get_session

router = APIRouter(prefix="/knowledge", tags=["知识库"])

# 全局 RAG 服务实例
_rag_service: Optional[RAGService] = None

# 构建任务状态
build_tasks = {}


def get_rag_service() -> RAGService:
    """获取 RAG 服务实例"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


class BuildRequest(BaseModel):
    """知识库构建请求"""
    folder_ids: List[int]  # 要处理的收藏夹 ID 列表
    exclude_bvids: Optional[List[str]] = None  # 排除的视频


class BuildStatus(BaseModel):
    """构建状态"""
    task_id: str
    status: str  # pending / running / completed / failed
    progress: int  # 0-100
    current_step: str
    total_videos: int
    processed_videos: int
    message: str


class FolderStatus(BaseModel):
    """收藏夹入库状态"""
    media_id: int
    indexed_count: int
    media_count: Optional[int] = None
    last_sync_at: Optional[datetime] = None


class SyncRequest(BaseModel):
    """同步请求"""
    folder_ids: Optional[List[int]] = None


class SyncResult(BaseModel):
    """同步结果"""
    folder_id: int
    total: int
    added: int
    removed: int
    indexed: int
    message: str
    last_sync_at: Optional[datetime] = None


class VideoArtifactItem(BaseModel):
    bvid: str
    title: str
    url: str
    source: Optional[str] = None
    summary: Optional[str] = None
    key_points: List[str] = []
    entities: List[dict] = []
    relations: List[dict] = []
    transcript_segment_count: int = 0
    updated_at: Optional[datetime] = None


def _extract_key_points_from_summary(summary_text: str) -> List[str]:
    key_points: List[str] = []
    for line in (summary_text or "").splitlines():
        cleaned = line.strip().lstrip("-•0123456789. ")
        if cleaned and cleaned not in key_points:
            key_points.append(cleaned)
        if len(key_points) >= 5:
            break
    return key_points


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _dedupe_key_points(summary_text: str, key_points: List[str]) -> List[str]:
    summary_norm = _normalize_text(summary_text)
    deduped: List[str] = []
    seen: set[str] = set()

    for item in key_points or []:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lstrip("-•123456789. ")
        if len(cleaned) < 6:
            continue
        normalized = _normalize_text(cleaned)
        if not normalized or normalized in seen:
            continue
        if summary_norm and (normalized in summary_norm or summary_norm in normalized):
            continue
        seen.add(normalized)
        deduped.append(cleaned)
        if len(deduped) >= 5:
            break

    return deduped


def _normalize_graph(knowledge_graph_json: dict) -> tuple[List[dict], List[dict]]:
    entities_raw = (knowledge_graph_json or {}).get("entities") or []
    relations_raw = (knowledge_graph_json or {}).get("relations") or []

    entities: List[dict] = []
    entity_names: set[str] = set()
    for item in entities_raw:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        key = _normalize_text(name)
        if key in entity_names:
            continue
        entity_names.add(key)
        entities.append({
            "name": name,
            "type": (item.get("type") or "").strip(),
            "description": (item.get("description") or "").strip(),
        })

    relations: List[dict] = []
    relation_keys: set[tuple[str, str, str]] = set()
    for item in relations_raw:
        if not isinstance(item, dict):
            continue
        source = (item.get("source") or "").strip()
        relation = (item.get("relation") or "").strip()
        target = (item.get("target") or "").strip()
        evidence = (item.get("evidence") or "").strip()
        if not source or not target:
            continue
        rel_key = (_normalize_text(source), _normalize_text(relation or "相关"), _normalize_text(target))
        if rel_key in relation_keys:
            continue
        relation_keys.add(rel_key)
        relations.append({
            "source": source,
            "relation": relation or "相关",
            "target": target,
            "evidence": evidence,
        })

        for name in (source, target):
            key = _normalize_text(name)
            if key and key not in entity_names:
                entity_names.add(key)
                entities.append({"name": name, "type": "", "description": ""})

    degree: dict[str, int] = {}
    for rel in relations:
        src = _normalize_text(rel.get("source") or "")
        tgt = _normalize_text(rel.get("target") or "")
        if src:
            degree[src] = degree.get(src, 0) + 1
        if tgt:
            degree[tgt] = degree.get(tgt, 0) + 1

    entities = sorted(
        entities,
        key=lambda item: (
            degree.get(_normalize_text(item.get("name") or ""), 0),
            1 if item.get("type") else 0,
            1 if item.get("description") else 0,
            len(item.get("name") or ""),
        ),
        reverse=True,
    )

    relations = sorted(
        relations,
        key=lambda item: (
            1 if item.get("evidence") else 0,
            degree.get(_normalize_text(item.get("source") or ""), 0)
            + degree.get(_normalize_text(item.get("target") or ""), 0),
            len(item.get("relation") or ""),
        ),
        reverse=True,
    )

    return entities[:8], relations[:8]


def _has_meaningful_summary(summary_json: dict) -> bool:
    if not isinstance(summary_json, dict):
        return False
    summary_text = (summary_json.get("summary") or "").strip()
    if len(summary_text) >= 20:
        return True
    key_points = [
        item for item in (summary_json.get("key_points") or [])
        if isinstance(item, str) and len(item.strip()) >= 6
    ]
    return len(key_points) >= 2


def _has_meaningful_graph(knowledge_graph_json: dict) -> bool:
    if not isinstance(knowledge_graph_json, dict):
        return False
    entities = [
        item for item in (knowledge_graph_json.get("entities") or [])
        if isinstance(item, dict) and (item.get("name") or "").strip()
    ]
    relations = [
        item for item in (knowledge_graph_json.get("relations") or [])
        if isinstance(item, dict)
        and (item.get("source") or "").strip()
        and (item.get("target") or "").strip()
    ]
    return len(entities) >= 2 and len(relations) >= 1


def _artifacts_missing(summary_json: dict, knowledge_graph_json: dict) -> bool:
    return (not _has_meaningful_summary(summary_json)) or (not _has_meaningful_graph(knowledge_graph_json))


def _build_summary_fallback(content_text: str, outline_json: List[dict]) -> tuple[Optional[str], List[str]]:
    points: List[str] = []

    for item in outline_json or []:
        if not isinstance(item, dict):
            continue
        item_title = (item.get("title") or "").strip()
        if item_title and item_title not in points:
            points.append(item_title)
        for point in item.get("points") or []:
            if not isinstance(point, dict):
                continue
            point_content = (point.get("content") or "").strip()
            if point_content and point_content not in points:
                points.append(point_content)
        if len(points) >= 5:
            break

    if not points and content_text:
        normalized = content_text.replace("\r", "\n")
        for chunk in normalized.split("\n"):
            text = chunk.strip().lstrip("-•0123456789. ")
            if text and text not in points:
                points.append(text)
            if len(points) >= 5:
                break

    summary = None
    if content_text:
        compact = " ".join(content_text.split())
        summary = compact[:220] + ("..." if len(compact) > 220 else "")
    elif points:
        summary = "；".join(points[:3])

    return summary, points[:5]


def _summary_has_content_overlap(summary_text: str, content_text: str, outline_json: List[dict]) -> bool:
    summary_text = (summary_text or "").strip()
    if not summary_text:
        return False

    corpus_parts: List[str] = []
    if content_text:
        corpus_parts.append(content_text)
    for item in outline_json or []:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if title:
            corpus_parts.append(title)
        for point in item.get("points") or []:
            if not isinstance(point, dict):
                continue
            point_text = (point.get("content") or "").strip()
            if point_text:
                corpus_parts.append(point_text)

    corpus = _normalize_text(" ".join(corpus_parts))
    if not corpus:
        return False

    summary_tokens = [
        token
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", summary_text)
        if len(token.strip()) >= 2
    ]
    if not summary_tokens:
        return False

    matches = sum(1 for token in summary_tokens if _normalize_text(token) in corpus)
    return matches >= max(2, min(4, len(summary_tokens) // 3 or 1))


def _pick_artifact_summary(
    source: Optional[str],
    summary_json: dict,
    content_text: str,
    outline_json: List[dict],
) -> tuple[Optional[str], List[str]]:
    summary_json = summary_json if isinstance(summary_json, dict) else {}
    summary_text = (summary_json.get("summary") or "").strip()
    key_points = [
        item for item in (summary_json.get("key_points") or [])
        if isinstance(item, str) and item.strip()
    ]

    if summary_text:
        if _summary_has_content_overlap(summary_text, content_text, outline_json):
            return summary_text, _dedupe_key_points(summary_text, key_points)
        # 非重叠场景允许保留较长摘要，避免误判导致“无摘要”。
        if len(summary_text) >= 48:
            return summary_text, _dedupe_key_points(summary_text, key_points)

    if key_points:
        merged_key_points = _dedupe_key_points(summary_text, key_points)
        if merged_key_points:
            synthetic_summary = summary_text or "；".join(merged_key_points[:3])
            return synthetic_summary, merged_key_points

    if source == ContentSource.AI_SUMMARY.value:
        fallback_summary, fallback_points = _build_summary_fallback(content_text, outline_json)
        if fallback_summary:
            return fallback_summary, _dedupe_key_points(fallback_summary, fallback_points)

    fallback_summary, fallback_points = _build_summary_fallback(content_text, outline_json)
    if fallback_summary:
        return fallback_summary, _dedupe_key_points(fallback_summary, fallback_points)

    return None, []


def _estimate_transcript_segments(transcript_json: List[dict], content_text: str, source: Optional[str]) -> int:
    if transcript_json:
        return len(transcript_json)
    if source in (ContentSource.ASR.value, ContentSource.SUBTITLE.value) and content_text:
        return max(1, min(99, len([line for line in content_text.splitlines() if line.strip()])))
    return 0


async def _get_or_create_folder(
    db: AsyncSession,
    session_id: str,
    media_id: int,
    title: Optional[str] = None,
    media_count: Optional[int] = None,
) -> FavoriteFolder:
    """获取或创建收藏夹记录"""
    result = await db.execute(
        select(FavoriteFolder).where(
            FavoriteFolder.session_id == session_id,
            FavoriteFolder.media_id == media_id,
        )
    )
    folder = result.scalar_one_or_none()

    if folder is None:
        folder = FavoriteFolder(
            session_id=session_id,
            media_id=media_id,
            title=title or "",
            media_count=media_count or 0,
            is_selected=True,
        )
        db.add(folder)
        await db.flush()
    else:
        if title:
            folder.title = title
        if media_count is not None:
            folder.media_count = media_count

    return folder


async def _get_folder_db_ids(
    db: AsyncSession,
    session_id: str,
    media_ids: Optional[List[int]] = None,
) -> List[int]:
    """按 session + media_id 获取内部收藏夹 ID，兼容同一用户多 session。"""
    result = await db.execute(
        select(UserSession.bili_mid).where(UserSession.session_id == session_id)
    )
    mid = result.scalar()

    target_session_ids = [session_id]
    if mid:
        rows = await db.execute(
            select(UserSession.session_id).where(UserSession.bili_mid == mid)
        )
        target_session_ids = [row[0] for row in rows.fetchall()]

    stmt = (
        select(FavoriteFolder.id, FavoriteFolder.media_id, FavoriteFolder.updated_at)
        .where(FavoriteFolder.session_id.in_(target_session_ids))
        .order_by(FavoriteFolder.updated_at.desc())
    )
    if media_ids:
        stmt = stmt.where(FavoriteFolder.media_id.in_(media_ids))

    rows = await db.execute(stmt)
    dedup: dict[int, int] = {}
    for folder_id, media_id, _updated_at in rows.fetchall():
        if media_id not in dedup:
            dedup[media_id] = folder_id
    return list(dedup.values())


def _extract_video_info(media: dict) -> tuple[str, str, Optional[int]]:
    """抽取视频关键信息"""
    bvid = media.get("bvid") or media.get("bv_id")
    title = media.get("title", bvid)
    cid = None
    ugc = media.get("ugc") or {}
    if ugc.get("first_cid"):
        cid = ugc.get("first_cid")
    else:
        pages = media.get("pages") or []
        first_page = pages[0] if pages and isinstance(pages[0], dict) else {}
        cid = (
            media.get("cid")
            or first_page.get("cid")
            or ugc.get("cid")
        )
    return bvid, title, cid


async def _upsert_video_cache(db: AsyncSession, bvid: str, meta: dict) -> None:
    """写入或更新视频缓存信息"""
    result = await db.execute(select(VideoCache).where(VideoCache.bvid == bvid))
    cache = result.scalar_one_or_none()

    if cache is None:
        cache = VideoCache(
            bvid=bvid,
            cid=meta.get("cid"),
            title=meta.get("title") or bvid,
            description=meta.get("intro"),
            owner_name=meta.get("owner_name"),
            owner_mid=meta.get("owner_mid"),
            duration=meta.get("duration"),
            pic_url=meta.get("cover"),
            is_processed=False,
        )
        db.add(cache)
        await db.flush()
        return

    if meta.get("cid") is not None:
        cache.cid = meta.get("cid")
    cache.title = meta.get("title") or cache.title
    if meta.get("intro") is not None:
        cache.description = meta.get("intro")
    if meta.get("owner_name") is not None:
        cache.owner_name = meta.get("owner_name")
    if meta.get("owner_mid") is not None:
        cache.owner_mid = meta.get("owner_mid")
    if meta.get("duration") is not None:
        cache.duration = meta.get("duration")
    if meta.get("cover") is not None:
        cache.pic_url = meta.get("cover")


async def _sync_folder(
    db: AsyncSession,
    bili: BilibiliService,
    rag: RAGService,
    content_fetcher: ContentFetcher,
    session_id: str,
    folder_id: int,
    exclude_bvids: Optional[set[str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """同步单个收藏夹到向量库"""
    info = {}
    try:
        info_result = await bili.get_favorite_content(folder_id, pn=1, ps=1)
        info = info_result.get("info", {})
    except Exception as e:
        logger.warning(f"获取收藏夹信息失败 [{folder_id}]: {e}")

    videos = await bili.get_all_favorite_videos(folder_id)
    total_in_folder = info.get("media_count", len(videos))

    # 保护：接口异常返回空列表时，避免误删
    if not videos:
        if total_in_folder and total_in_folder > 0:
            logger.warning(f"[{folder_id}] 收藏夹返回空列表，跳过删除逻辑")
            existing_count = await db.scalar(
                select(func.count(FavoriteVideo.bvid))
                .where(FavoriteVideo.folder_id == folder_id)
            )
            return {
                "folder_id": folder_id,
                "total": total_in_folder,
                "added": 0,
                "removed": 0,
                "indexed": existing_count or 0,
                "message": "本次同步异常：空列表，已跳过",
                "last_sync_at": datetime.utcnow(),
            }

    video_map = {}
    skipped_invalid = 0
    for media in videos:
        bvid, title, cid = _extract_video_info(media)
        if not bvid:
            continue
        if exclude_bvids and bvid in exclude_bvids:
            continue
        
        # 过滤失效视频（被删除、下架等）
        # attr 字段: 0=正常, 9=已失效, 1=私密等
        attr = media.get("attr", 0)
        if attr == 9 or title in ["已失效视频", "已删除视频"]:
            skipped_invalid += 1
            logger.debug(f"跳过失效视频: {bvid} - {title}")
            continue
        
        owner = media.get("upper") or {}
        video_map[bvid] = {
            "title": title,
            "cid": cid,
            "intro": media.get("intro"),
            "cover": media.get("cover"),
            "duration": media.get("duration"),
            "owner_name": owner.get("name"),
            "owner_mid": owner.get("mid"),
        }
    
    if skipped_invalid > 0:
        logger.info(f"[{folder_id}] 过滤了 {skipped_invalid} 个失效视频")

    # 以有效视频数作为统计口径（过滤失效视频）
    valid_count = len(video_map)
    current_bvids = set(video_map.keys())

    folder = await _get_or_create_folder(
        db,
        session_id=session_id,
        media_id=folder_id,
        title=info.get("title"),
        media_count=valid_count,
    )

    existing_rows = await db.execute(
        select(FavoriteVideo.bvid).where(FavoriteVideo.folder_id == folder.id)
    )
    existing_bvids = {row[0] for row in existing_rows.fetchall()}

    added = current_bvids - existing_bvids
    removed = existing_bvids - current_bvids

    # 写入标题/简介等信息
    for bvid, meta in video_map.items():
        await _upsert_video_cache(db, bvid, meta)

    source_priority = {
        ContentSource.BASIC_INFO.value: 1,
        ContentSource.AI_SUMMARY.value: 2,
        ContentSource.SUBTITLE.value: 3,
        ContentSource.ASR.value: 4,
    }

    def _is_better_source(new_source: str, old_source: Optional[str]) -> bool:
        return source_priority.get(new_source, 0) > source_priority.get(old_source or "", 0)

    def _should_refresh_cache(cache: Optional[VideoCache]) -> bool:
        if not cache:
            return True
        text = (cache.content or "").strip()
        if len(text) < 50:
            return True
        if cache.content_source in (None, "", ContentSource.BASIC_INFO.value):
            return True
        return False

    def _is_asr_cache_usable(cache: Optional[VideoCache]) -> bool:
        if not cache:
            return False
        if cache.content_source not in (ContentSource.ASR.value, ContentSource.SUBTITLE.value):
            return False
        text = (cache.content or "").strip()
        return len(text) >= 50

    # 需要更新的已存在视频（缓存过少或来源较弱）
    update_candidates: set[str] = set()
    for bvid in current_bvids & existing_bvids:
        if bvid in added:
            continue
        result = await db.execute(select(VideoCache).where(VideoCache.bvid == bvid))
        cache = result.scalar_one_or_none()
        meta = video_map.get(bvid) or {}
        if _should_refresh_cache(cache):
            update_candidates.add(bvid)
            continue
        if (
            cache
            and cache.content_source in (ContentSource.SUBTITLE.value, ContentSource.ASR.value)
            and not meta.get("cid")
        ):
            update_candidates.add(bvid)
            continue
        if cache and meta.get("cid") and cache.cid and int(cache.cid) != int(meta.get("cid")):
            update_candidates.add(bvid)
            continue
        if (
            cache
            and cache.content_source in (ContentSource.ASR.value, ContentSource.SUBTITLE.value)
            and isinstance(cache.transcript_json, list)
            and cache.transcript_json
            and _artifacts_missing(cache.summary_json, cache.knowledge_graph_json)
        ):
            update_candidates.add(bvid)

    # 新增/更新向量与关联
    targets = list(added) + list(update_candidates)
    total_targets = len(targets)
    processed_targets = 0
    if progress_callback:
        progress_callback("准备处理", processed_targets, total_targets)
    for bvid in targets:
        meta = video_map[bvid]

        if not meta.get("cid"):
            try:
                video_info = await bili.get_video_info(bvid)
                meta["cid"] = video_info.get("cid") or meta.get("cid")
                meta["intro"] = video_info.get("desc") or meta.get("intro")
                owner = video_info.get("owner") or {}
                meta["owner_name"] = owner.get("name") or meta.get("owner_name")
                meta["owner_mid"] = owner.get("mid") or meta.get("owner_mid")
                meta["duration"] = video_info.get("duration") or meta.get("duration")
                meta["cover"] = video_info.get("pic") or meta.get("cover")
            except Exception as e:
                logger.warning(f"[{bvid}] 补齐权威视频信息失败: {e}")
        
        # 尝试添加到向量库（可能失败，但不影响记录入库）
        try:
            global_count = await db.scalar(
                select(func.count()).select_from(FavoriteVideo).where(FavoriteVideo.bvid == bvid)
            )
            # 检查缓存内容是否缺失
            result = await db.execute(select(VideoCache).where(VideoCache.bvid == bvid))
            cache = result.scalar_one_or_none()
            if cache is None:
                await _upsert_video_cache(db, bvid, meta)
                result = await db.execute(select(VideoCache).where(VideoCache.bvid == bvid))
                cache = result.scalar_one_or_none()
            old_content = (cache.content or "").strip() if cache else ""
            old_source = cache.content_source if cache else None

            needs_fetch = _should_refresh_cache(cache)
            content = None
            should_update_cache = False
            should_reindex = False

            if (
                not needs_fetch
                and cache
                and cache.content_source in (ContentSource.ASR.value, ContentSource.SUBTITLE.value)
                and isinstance(cache.transcript_json, list)
                and cache.transcript_json
                and _artifacts_missing(cache.summary_json, cache.knowledge_graph_json)
            ):
                try:
                    artifacts = await rag.build_transcript_artifacts(
                        cache.title or meta["title"],
                        cache.transcript_json,
                    )
                    cache.summary_json = artifacts.get("summary")
                    cache.knowledge_graph_json = artifacts.get("knowledge_graph")
                    cache.is_processed = True
                    should_update_cache = True
                    should_reindex = True
                    logger.info(f"[{bvid}] 已基于缓存 transcript 回填摘要与知识图谱")
                except Exception as e:
                    logger.warning(f"[{bvid}] transcript 回填摘要/图谱失败: {e}")

            if needs_fetch:
                content = await content_fetcher.fetch_content(
                    bvid, cid=meta["cid"], title=meta["title"]
                )
                if content and content.transcript:
                    artifacts = await rag.build_transcript_artifacts(content.title, content.transcript)
                    content.summary = artifacts.get("summary")
                    content.knowledge_graph = artifacts.get("knowledge_graph")
                new_text = (content.content or "").strip() if content else ""
                new_source = content.source.value if content else None

                if not old_content:
                    should_update_cache = True
                    should_reindex = True
                elif new_source and _is_better_source(new_source, old_source):
                    should_update_cache = True
                    should_reindex = True
                elif new_text and new_text != old_content:
                    should_update_cache = True
                    should_reindex = True

                if cache and should_update_cache:
                    cache.content = content.content
                    cache.content_source = content.source.value
                    cache.outline_json = content.outline
                    cache.transcript_json = content.transcript
                    cache.summary_json = content.summary
                    cache.knowledge_graph_json = content.knowledge_graph
                    cache.is_processed = True
                    logger.info(f"[{bvid}] 已写入缓存: source={cache.content_source}")

            # 需要重建向量：新增/升级/内容变化 或 向量缺失
            if (global_count == 0) or should_reindex:
                if not content:
                    if _is_asr_cache_usable(cache):
                        content = VideoContent(
                            bvid=bvid,
                            title=meta["title"],
                            content=(cache.content or "").strip(),
                            source=ContentSource(cache.content_source),
                            outline=cache.outline_json,
                            transcript=cache.transcript_json,
                            summary=cache.summary_json,
                            knowledge_graph=cache.knowledge_graph_json,
                        )
                        cache.is_processed = True
                        logger.info(f"[{bvid}] 使用缓存 ASR 内容重建向量")
                    else:
                        content = await content_fetcher.fetch_content(
                            bvid, cid=meta["cid"], title=meta["title"]
                        )
                        if content and content.transcript:
                            artifacts = await rag.build_transcript_artifacts(content.title, content.transcript)
                            content.summary = artifacts.get("summary")
                            content.knowledge_graph = artifacts.get("knowledge_graph")
                        if cache:
                            cache.content = content.content
                            cache.content_source = content.source.value
                            cache.outline_json = content.outline
                            cache.transcript_json = content.transcript
                            cache.summary_json = content.summary
                            cache.knowledge_graph_json = content.knowledge_graph
                            cache.is_processed = True
                            logger.info(f"[{bvid}] 已写入缓存: source={cache.content_source}")
                try:
                    rag.delete_video(bvid)
                except Exception as e:
                    logger.warning(f"删除旧向量失败 [{bvid}]: {e}")
                chunks = rag.add_video_content(content)
                logger.info(f"[{bvid}] 向量化完成，块数={chunks}")
            else:
                logger.info(f"[{bvid}] 内容未变化或无需升级，跳过向量化")
        except Exception as e:
            logger.warning(f"添加向量失败 [{bvid}]: {e} (仍会记录到数据库)")
        
        # 无论向量是否添加成功，都写入 FavoriteVideo 记录
        try:
            exists_row = await db.execute(
                select(FavoriteVideo.id).where(
                    FavoriteVideo.folder_id == folder.id,
                    FavoriteVideo.bvid == bvid,
                )
            )
            if exists_row.scalar_one_or_none() is None:
                db.add(FavoriteVideo(folder_id=folder.id, bvid=bvid, is_selected=True))
            processed_targets += 1
            if progress_callback:
                progress_callback(meta["title"], processed_targets, total_targets)
        except Exception as e:
            logger.error(f"写入数据库失败 [{bvid}]: {e}")

    # 删除无效向量
    if removed:
        for bvid in removed:
            other_count = await db.scalar(
                select(func.count())
                .select_from(FavoriteVideo)
                .where(
                    FavoriteVideo.bvid == bvid,
                    FavoriteVideo.folder_id != folder.id,
                )
            )
            if other_count == 0:
                try:
                    rag.delete_video(bvid)
                except Exception as e:
                    logger.warning(f"删除向量失败 [{bvid}]: {e}")

        await db.execute(
            delete(FavoriteVideo).where(
                FavoriteVideo.folder_id == folder.id,
                FavoriteVideo.bvid.in_(removed),
            )
        )

    folder.last_sync_at = datetime.utcnow()

    await db.commit()

    indexed_count = await db.scalar(
        select(func.count(func.distinct(FavoriteVideo.bvid)))
        .select_from(FavoriteVideo)
        .where(FavoriteVideo.folder_id == folder.id)
    )

    return {
        "folder_id": folder_id,
        "total": valid_count,
        "added": len(added),
        "removed": len(removed),
        "indexed": indexed_count or 0,
        "message": "同步完成",
        "last_sync_at": folder.last_sync_at,
    }


@router.get("/stats")
async def get_knowledge_stats():
    """获取知识库统计信息"""
    try:
        rag = get_rag_service()
        stats = rag.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/artifacts", response_model=List[VideoArtifactItem])
async def get_video_artifacts(
    session_id: str = Query(..., description="会话ID"),
    folder_ids: Optional[List[int]] = Query(default=None, description="收藏夹 media_id 列表"),
    limit: int = Query(24, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """获取已入库视频的摘要与知识图谱，用于前端常驻展示。"""
    folder_db_ids = await _get_folder_db_ids(db, session_id, folder_ids)
    if not folder_db_ids:
        return []

    stmt = (
        select(
            VideoCache.bvid,
            VideoCache.title,
            VideoCache.content,
            VideoCache.outline_json,
            VideoCache.content_source,
            VideoCache.summary_json,
            VideoCache.knowledge_graph_json,
            VideoCache.transcript_json,
            VideoCache.updated_at,
        )
        .join(FavoriteVideo, FavoriteVideo.bvid == VideoCache.bvid)
        .where(FavoriteVideo.folder_id.in_(folder_db_ids))
        .order_by(VideoCache.updated_at.desc())
        .limit(limit * 3)
    )

    rows = await db.execute(stmt)
    items: List[VideoArtifactItem] = []
    seen_bvids: set[str] = set()

    for bvid, title, content, outline_json, source, summary_json, knowledge_graph_json, transcript_json, updated_at in rows.fetchall():
        if not bvid or bvid in seen_bvids:
            continue
        seen_bvids.add(bvid)

        summary_json = summary_json if isinstance(summary_json, dict) else {}
        knowledge_graph_json = knowledge_graph_json if isinstance(knowledge_graph_json, dict) else {}
        transcript_json = transcript_json if isinstance(transcript_json, list) else []
        outline_json = outline_json if isinstance(outline_json, list) else []
        content_text = (content or "").strip()

        summary_text, key_points = _pick_artifact_summary(
            source,
            summary_json,
            content_text,
            outline_json,
        )

        entities, relations = _normalize_graph(knowledge_graph_json)

        items.append(
            VideoArtifactItem(
                bvid=bvid,
                title=title or bvid,
                url=f"https://www.bilibili.com/video/{bvid}",
                source=source,
                summary=summary_text or None,
                key_points=key_points[:5],
                entities=entities[:8],
                relations=relations[:8],
                transcript_segment_count=_estimate_transcript_segments(transcript_json, content_text, source),
                updated_at=updated_at,
            )
        )

        if len(items) >= limit:
            break

    return items


@router.get("/folders/status", response_model=List[FolderStatus])
async def get_folder_status(
    session_id: str = Query(..., description="会话ID"),
    db: AsyncSession = Depends(get_db),
):
    """获取收藏夹入库状态（跨 Session 查找同一用户的数据）"""
    
    # 1. 先查当前 Session 对应的用户 MID
    result = await db.execute(
        select(UserSession.bili_mid).where(UserSession.session_id == session_id)
    )
    mid = result.scalar()
    
    target_session_ids = [session_id]
    
    if mid:
        # 2. 如果有 MID，查找该用户所有的 Session ID
        result = await db.execute(
            select(UserSession.session_id).where(UserSession.bili_mid == mid)
        )
        target_session_ids = [row[0] for row in result.fetchall()]
    
    # 3. 查询所有关联 Session 的收藏夹状态
    # 使用 group_by media_id 来去重，取最新的那个
    rows = await db.execute(
        select(FavoriteFolder.id, FavoriteFolder.media_id, FavoriteFolder.last_sync_at)
        .where(FavoriteFolder.session_id.in_(target_session_ids))
        .order_by(FavoriteFolder.updated_at.desc())
    )
    
    # 手动按 media_id 去重，保留最新的
    folders_map = {}
    for row in rows.fetchall():
        fid, media_id, last_sync = row
        if media_id not in folders_map:
            folders_map[media_id] = (fid, last_sync)
            
    if not folders_map:
        return []

    folder_ids = [v[0] for v in folders_map.values()]
    
    # 4. 统计视频数量
    counts = await db.execute(
        select(FavoriteVideo.folder_id, func.count(func.distinct(FavoriteVideo.bvid)))
        .where(FavoriteVideo.folder_id.in_(folder_ids))
        .group_by(FavoriteVideo.folder_id)
    )
    count_map = {row[0]: row[1] for row in counts.fetchall()}

    result = []
    for media_id, (folder_id, last_sync_at) in folders_map.items():
        # 读取有效视频数（过滤失效后的口径）
        folder_row = await db.execute(
            select(FavoriteFolder.media_count).where(FavoriteFolder.id == folder_id)
        )
        media_count = folder_row.scalar()
        result.append(
            FolderStatus(
                media_id=media_id,
                indexed_count=count_map.get(folder_id, 0),
                media_count=media_count,
                last_sync_at=last_sync_at,
            )
        )
    return result


@router.post("/folders/sync", response_model=List[SyncResult])
async def sync_folders(
    request: SyncRequest,
    session_id: str = Query(..., description="会话ID"),
    db: AsyncSession = Depends(get_db),
):
    """同步收藏夹到向量库"""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")

    cookies = session.get("cookies", {})
    user_info = session.get("user_info", {})

    bili = BilibiliService(
        sessdata=cookies.get("SESSDATA"),
        bili_jct=cookies.get("bili_jct"),
        dedeuserid=cookies.get("DedeUserID"),
    )
    rag = get_rag_service()
    asr_service = ASRService()
    content_fetcher = ContentFetcher(bili, asr_service)

    try:
        folder_ids = request.folder_ids or []
        if not folder_ids:
            mid = user_info.get("mid") or cookies.get("DedeUserID")
            if not mid:
                raise HTTPException(status_code=400, detail="无法获取用户信息")
            folders = await bili.get_user_favorites(mid=mid)
            folder_ids = [folder.get("id") for folder in folders if folder.get("id")]

        results: List[SyncResult] = []
        for folder_id in folder_ids:
            try:
                result = await _sync_folder(
                    db,
                    bili,
                    rag,
                    content_fetcher,
                    session_id,
                    folder_id,
                )
                results.append(SyncResult(**result))
            except Exception as e:
                logger.error(f"同步收藏夹失败 [{folder_id}]: {e}")
                results.append(
                    SyncResult(
                        folder_id=folder_id,
                        total=0,
                        added=0,
                        removed=0,
                        indexed=0,
                        message=f"同步失败: {e}",
                        last_sync_at=None,
                    )
                )

        return results
    finally:
        await bili.close()


@router.post("/build")
async def build_knowledge_base(
    request: BuildRequest,
    background_tasks: BackgroundTasks,
    session_id: str = Query(..., description="会话ID"),
):
    """构建知识库（后台任务）"""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")

    import uuid
    task_id = str(uuid.uuid4())

    build_tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "current_step": "初始化中...",
        "total_videos": 0,
        "processed_videos": 0,
        "message": "",
    }

    background_tasks.add_task(
        _build_knowledge_base_task,
        task_id,
        session_id,
        session,
        request.folder_ids,
        request.exclude_bvids or [],
    )

    return {"task_id": task_id, "message": "构建任务已启动"}


async def _build_knowledge_base_task(
    task_id: str,
    session_id: str,
    session: dict,
    folder_ids: List[int],
    exclude_bvids: List[str],
):
    """后台构建任务"""
    cookies = session.get("cookies", {})

    try:
        build_tasks[task_id]["status"] = "running"
        build_tasks[task_id]["current_step"] = "同步收藏夹..."

        bili = BilibiliService(
            sessdata=cookies.get("SESSDATA"),
            bili_jct=cookies.get("bili_jct"),
            dedeuserid=cookies.get("DedeUserID"),
        )
        asr_service = ASRService()
        content_fetcher = ContentFetcher(bili, asr_service)
        rag = get_rag_service()

        try:
            total_folders = len(folder_ids)
            if total_folders == 0:
                build_tasks[task_id]["status"] = "completed"
                build_tasks[task_id]["progress"] = 100
                build_tasks[task_id]["message"] = "没有需要处理的收藏夹"
                return

            processed = 0
            total_added = 0
            total_removed = 0

            async with get_db_context() as db:
                for idx, folder_id in enumerate(folder_ids, start=1):
                    build_tasks[task_id]["current_step"] = f"同步收藏夹 {folder_id}"

                    def progress_cb(title: str, processed_count: int = 0, total_count: int = 0):
                        build_tasks[task_id]["current_step"] = f"处理: {title}"
                        if total_count:
                            build_tasks[task_id]["total_videos"] = total_count
                        if processed_count:
                            build_tasks[task_id]["processed_videos"] = processed_count
                            if build_tasks[task_id]["total_videos"]:
                                build_tasks[task_id]["progress"] = int(
                                    (processed_count / build_tasks[task_id]["total_videos"]) * 100
                                )

                    result = await _sync_folder(
                        db,
                        bili,
                        rag,
                        content_fetcher,
                        session_id,
                        folder_id,
                        exclude_bvids=set(exclude_bvids),
                        progress_callback=progress_cb,
                    )

                    processed = idx
                    total_added += result["added"]
                    total_removed += result["removed"]

            build_tasks[task_id]["status"] = "completed"
            build_tasks[task_id]["progress"] = 100
            build_tasks[task_id]["processed_videos"] = total_folders
            build_tasks[task_id]["current_step"] = "完成"
            build_tasks[task_id]["message"] = f"同步完成：新增 {total_added}，移除 {total_removed}"

            logger.info(f"知识库构建完成: 新增 {total_added}，移除 {total_removed}")
        finally:
            await bili.close()

    except Exception as e:
        logger.error(f"构建任务失败: {e}")
        build_tasks[task_id]["status"] = "failed"
        build_tasks[task_id]["message"] = str(e)


@router.get("/build/status/{task_id}", response_model=BuildStatus)
async def get_build_status(task_id: str):
    """获取构建任务状态"""
    if task_id not in build_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = build_tasks[task_id]
    return BuildStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        current_step=task["current_step"],
        total_videos=task["total_videos"],
        processed_videos=task["processed_videos"],
        message=task["message"],
    )


@router.delete("/clear")
async def clear_knowledge_base():
    """清空知识库"""
    try:
        rag = get_rag_service()
        rag.clear_collection()
        return {"message": "知识库已清空"}
    except Exception as e:
        logger.error(f"清空知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/video/{bvid}")
async def delete_video_from_knowledge(bvid: str):
    """从知识库中删除指定视频"""
    try:
        rag = get_rag_service()
        rag.delete_video(bvid)
        return {"message": f"已删除视频 {bvid}"}
    except Exception as e:
        logger.error(f"删除视频失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
Bilibili RAG 知识库系统
对话路由 - 智能问答
"""
import re
import json
from dataclasses import dataclass
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from openai import OpenAI
from langchain.schema import Document

from app.database import get_db
from app.models import ChatRequest, ChatResponse, FavoriteFolder, FavoriteVideo, UserSession, VideoCache
from app.config import settings
from app.routers.knowledge import get_rag_service

router = APIRouter(prefix="/chat", tags=["对话"])


def _format_timestamp(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_source_payload(meta: dict) -> dict:
    bvid = meta.get("bvid", "")
    title = meta.get("title", "")
    start_sec = meta.get("start_sec")
    end_sec = meta.get("end_sec")
    timestamp_text = meta.get("timestamp_text") or _format_timestamp(start_sec)
    base_url = f"https://www.bilibili.com/video/{bvid}" if bvid else meta.get("url", "")
    url = meta.get("url") or (f"{base_url}?t={max(0, int(start_sec))}" if start_sec is not None else base_url)
    payload = {
        "bvid": bvid,
        "title": title,
        "url": url,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "timestamp": timestamp_text,
    }
    for key in ("evidence_id", "content_source", "evidence_text"):
        if meta.get(key) is not None:
            payload[key] = meta[key]
    return payload


@dataclass
class PreparedChat:
    messages: list[dict]
    sources: list[dict]
    direct_answer: Optional[str] = None
    notice: str = ""

def _get_llm_client() -> OpenAI:
    """获取 LLM 客户端"""
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="未配置 LLM API Key")
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

def _build_overview_messages(context: str, question: str) -> list[dict]:
    system = (
        "你是一个收藏夹知识库助手。用户想要了解他们收藏夹的整体内容。\n"
        "请根据以下视频信息回答用户的问题。回答要：\n"
        "1. 自然、友好、有条理\n"
        "2. 可以总结、分类、提炼要点\n"
        "3. 如果内容较多，挑选代表性的进行介绍\n\n"
        f"收藏夹内容：\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

def _build_rag_messages(context: str, question: str) -> list[dict]:
    system = (
        "你是一个严格基于视频内容作答的深度知识助手。\n"
        "你的目标不是寒暄，而是认真分析用户收藏视频里的具体内容。\n\n"
        "回答要求：\n"
        "1. 优先依据给定内容中的细节、论点、步骤、定义、因果关系来回答，不能泛泛而谈。\n"
        "2. 回答长度根据问题复杂度自然调整：简单问题就直接说明，复杂问题再展开，不要为了显得完整而堆废话，也不要过度压缩成一句话。\n"
        "3. 明确说明这些结论分别来自哪些视频或哪些片段，不要只给空洞结论。\n"
        "4. 如果检索内容不足以支持某个判断，要明确说“现有视频内容没有直接说明”。\n"
        "5. 不要重复用户问题，不要输出正确的废话，不要写通用套话。\n"
        "6. 回答时尽量保留原视频中的关键术语、方法名、观点分歧和步骤顺序。\n"
        "7. 如果问题是“为什么/怎么做/区别是什么”，优先回答机制、步骤、对比，而不是一句话概括。\n\n"
        "建议输出风格：\n"
        "先给直接结论，再按需要展开依据，最后用“来源”简短点名相关视频。\n\n"
        f"相关内容：\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

def _build_fallback_messages(context: str, question: str) -> list[dict]:
    system = (
        "你是一个收藏夹知识库助手。\n"
        "用户的问题在现有知识库中没有检索到直接内容。\n"
        "以下是用户收藏夹中的视频概览（如果为空说明用户还没入库）：\n"
        f"{context}\n\n"
        "请根据以上信息（如果有）：\n"
        "1. 尝试回答用户问题\n"
        "2. 如果没有任何视频信息，礼貌地告诉用户需要先在左侧选择收藏夹并点击「入库」或者「更新」\n"
        "3. 保持像真人助手一样的语气，不要显示这是\"备选方案\""
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

def _build_direct_messages(question: str) -> list[dict]:
    """通用回答（不查库）"""
    system = (
        "你是一个知识库问答助手。\n"
        "请直接回答用户问题，避免引入收藏夹或知识库内容。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

def _build_direct_messages_with_context(context: str, question: str) -> list[dict]:
    """带收藏夹上下文的通用回答（引导用户提问）"""
    system = (
        "你是一个知识库问答助手。\n"
        "以下是用户收藏夹的概览（收藏夹名称与视频标题）：\n"
        f"{context}\n\n"
        "请先回答用户问题，再根据收藏夹内容引导用户提出与收藏相关的问题。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

def _log_final_payload(route: str, messages: list[dict], sources: list[dict]) -> None:
    """记录最终发送给 LLM 的内容与来源"""
    logger.info(f"最终路由: {route}")
    logger.info(f"最终消息: {messages}")
    logger.info(f"最终来源数量: {len(sources)}")

def _build_db_list_messages(context: str, question: str) -> list[dict]:
    """仅用标题/简介回答列表类问题"""
    system = (
        "你是一个收藏夹知识库助手。\n"
        "用户需要清单/列表类答案，请基于以下视频标题与简介回答。\n"
        "回答要：\n"
        "1. 按收藏夹或主题分组\n"
        "2. 只输出与问题相关的条目\n"
        "3. 简洁清晰\n\n"
        f"收藏夹内容：\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

def _build_db_summary_messages(context: str, question: str) -> list[dict]:
    """仅用数据库内容回答总结类问题"""
    system = (
        "你是一个收藏夹知识库助手。\n"
        "用户需要总结/提炼，请基于以下视频内容回答。\n"
        "回答要：\n"
        "1. 提炼重点与要点，但回答长度按问题复杂度自然控制，不要机械地写得特别短\n"
        "2. 尽量总结出主题、核心观点、方法步骤、争议点或适用场景\n"
        "3. 结构清晰、可快速理解\n"
        "4. 必要时引用视频标题作为来源\n"
        "5. 如果数据库里已经有文字稿总结或知识图谱，要优先吸收进去，不要忽略\n\n"
        f"收藏夹内容：\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

def _is_list_question(question: str) -> bool:
    """列表/清单类问题"""
    list_terms = ["有哪些", "有什么", "列表", "清单", "目录", "都有哪些", "列出", "罗列", "多少个", "几个"]
    return any(term in question for term in list_terms)

def _is_summary_question(question: str) -> bool:
    """总结/概括类问题"""
    summary_terms = ["总结", "概述", "概括", "分析", "梳理", "提炼", "回顾", "复盘", "要点", "重点", "关键点", "核心", "讲了什么", "讲些什么"]
    return any(term in question for term in summary_terms)

def _is_general_question(question: str) -> bool:
    """通用闲聊/与收藏无关的问题"""
    general_terms = ["你好", "嗨", "哈喽", "hello", "hi", "在吗", "你是谁", "你能做什么", "谢谢", "晚安", "早安", "早上好"]
    cleaned = re.sub(r"[\\W_]+", "", question, flags=re.UNICODE)
    lowered = cleaned.lower()
    residual = lowered
    for term in general_terms:
        residual = residual.replace(term.lower(), "")
    return residual == ""

def _is_collection_intent(question: str) -> bool:
    """是否显式指向收藏/视频/知识库"""
    terms = ["收藏", "收藏夹", "视频", "合集", "up主", "BV", "bv", "分P", "字幕", "知识库", "入库", "同步", "向量", "检索"]
    return any(term in question for term in terms)

def _is_overview_question(question: str) -> bool:
    """概览类问题（列表或总结）"""
    return _is_list_question(question) or _is_summary_question(question)

def _route_with_rules(question: str, is_collection_intent: bool, related: bool) -> str:
    """规则路由兜底"""
    if _is_general_question(question) and not is_collection_intent:
        return "direct"
    if _is_list_question(question):
        return "db_list"
    if _is_summary_question(question):
        return "db_content"
    if not related and not is_collection_intent:
        return "direct"
    return "vector"

def _route_with_llm(question: str) -> tuple[Optional[str], str]:
    """使用 LLM 进行路由判断"""
    try:
        client = _get_llm_client()
        system = (
            "你是一个路由器，只输出以下之一：direct, db_list, db_content, vector。\n"
            "规则：\n"
            "- direct：寒暄/闲聊/与收藏无关的问题\n"
            "- db_list：清单/列表/目录/有哪些\n"
            "- db_content：明确要求“全部/所有/整体/概览/全库”内容的总结\n"
            "- vector：具体主题问题或需要“先检索再总结”的问题\n"
            "示例：\n"
            "Q: 中西方文化的差异是什么，简单总结 -> vector\n"
            "Q: 概览我收藏夹里所有王德峰相关内容 -> db_content\n"
            "只输出一个词，不要解释。"
        )
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        match = re.search(r"(direct|db_list|db_content|vector)", text)
        return (match.group(1) if match else None), text
    except Exception as e:
        logger.warning(f"LLM 路由失败: {e}")
        return None, ""

def _extract_keywords(question: str) -> List[str]:
    """提取用于过滤的关键词"""
    stopwords = {
        "什么", "怎么", "如何", "是否", "可以", "哪个", "哪些", "请问", "一下", "为什么",
        "有没有", "能不能", "能否", "是不是", "是什么", "多少", "哪里", "讲讲", "介绍",
        "总结", "概括", "分析", "解释", "说明", "评价", "区别", "内容", "视频",
    }
    keywords: List[str] = []
    for kw in re.findall(r"[\u4e00-\u9fff]{2,}", question):
        if kw not in stopwords and kw not in keywords:
            keywords.append(kw)
    for kw in re.findall(r"[A-Za-z0-9]{2,}", question):
        if kw not in keywords:
            keywords.append(kw)
    return keywords

def _filter_docs_by_keywords(docs: List[Document], question: str) -> List[Document]:
    """根据关键词过滤召回内容，减少噪声"""
    keywords = _extract_keywords(question)
    if not keywords:
        return []
    filtered: List[Document] = []
    for doc in docs:
        meta = doc.metadata or {}
        title = meta.get("title", "") or ""
        content = doc.page_content or ""
        if any(kw in title for kw in keywords) or any(kw in content for kw in keywords):
            filtered.append(doc)
    return filtered

async def _is_related_to_collection(db: AsyncSession, folder_ids: List[int], question: str) -> bool:
    """判断问题是否与收藏夹内容有关"""
    if not folder_ids:
        return False
    keywords = _extract_keywords(question)
    if not keywords:
        return False
    like_conds = []
    for kw in keywords:
        pattern = f"%{kw}%"
        like_conds.append(VideoCache.title.ilike(pattern))
        like_conds.append(VideoCache.description.ilike(pattern))
        like_conds.append(VideoCache.content.ilike(pattern))
    stmt = (
        select(func.count())
        .select_from(VideoCache)
        .join(FavoriteVideo, FavoriteVideo.bvid == VideoCache.bvid)
        .where(FavoriteVideo.folder_id.in_(folder_ids))
        .where(or_(*like_conds))
    )
    count = await db.scalar(stmt)
    return (count or 0) > 0

async def _get_folder_ids_for_session(db: AsyncSession, session_id: str, media_ids: Optional[List[int]]) -> List[int]:
    """根据 session 和 media_id 获取内部 folder_id（支持跨 session 查找同用户数据）"""
    session_result = await db.execute(
        select(UserSession).where(
            UserSession.session_id == session_id,
            UserSession.is_valid.is_(True),
        )
    )
    session = session_result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")

    mid = session.bili_mid
    target_session_ids = [session_id]
    if mid:
        # 查找该用户所有的 Session ID
        sessions_result = await db.execute(select(UserSession.session_id).where(UserSession.bili_mid == mid))
        target_session_ids = [row[0] for row in sessions_result.fetchall()]
    # 构建查询：按 media_id 去重，只保留最新的一条
    stmt = (
        select(FavoriteFolder.id, FavoriteFolder.media_id, FavoriteFolder.updated_at)
        .where(FavoriteFolder.session_id.in_(target_session_ids))
        .order_by(FavoriteFolder.updated_at.desc())
    )
    rows = await db.execute(stmt)
    dedup: dict[int, int] = {}
    for folder_id, media_id, _updated_at in rows.fetchall():
        if media_id not in dedup:
            dedup[media_id] = folder_id
    if media_ids:
        if set(media_ids) - set(dedup):
            raise HTTPException(status_code=403, detail="收藏夹不属于当前用户或尚未入库")
        return [dedup[media_id] for media_id in dict.fromkeys(media_ids)]
    return list(dedup.values())

async def _get_bvids_by_folder_ids(db: AsyncSession, folder_ids: List[int]) -> List[str]:
    """获取指定收藏夹的视频 BV 列表"""
    if not folder_ids:
        return []
    rows = await db.execute(select(FavoriteVideo.bvid).where(FavoriteVideo.folder_id.in_(folder_ids)))
    bvids = []
    seen = set()
    for (bvid,) in rows.fetchall():
        if not bvid or bvid in seen:
            continue
        seen.add(bvid)
        bvids.append(bvid)
    return bvids

async def _get_video_context(db: AsyncSession, folder_ids: List[int], include_content: bool = False, limit: Optional[int] = 50) -> tuple[str, List[dict]]:
    """获取视频上下文信息"""
    if not folder_ids:
        return "", []
    # 查询视频信息
    query = (
        select(
            FavoriteFolder.title.label("folder_title"),
            VideoCache.bvid,
            VideoCache.title,
            VideoCache.description,
            VideoCache.content if include_content else VideoCache.description,
            VideoCache.summary_json,
            VideoCache.knowledge_graph_json,
        )
        .join(FavoriteVideo, FavoriteVideo.folder_id == FavoriteFolder.id)
        .join(VideoCache, VideoCache.bvid == FavoriteVideo.bvid, isouter=True)
        .where(FavoriteFolder.id.in_(folder_ids))
    )
    if limit is not None:
        query = query.limit(limit)
    result = await db.execute(query)
    records = result.fetchall()
    if not records:
        return "", []
    # 按收藏夹分组（对 bvid 去重，避免同一视频重复出现）
    grouped = {}
    sources = []
    seen_bvids = set()
    for folder_title, bvid, title, desc, content, summary_json, knowledge_graph_json in records:
        if not bvid or not title:
            continue
        if bvid in seen_bvids:
            continue
        folder_name = folder_title or "默认收藏夹"
        if folder_name not in grouped:
            grouped[folder_name] = []
        video_info = f"- 《{title}》"
        if include_content and content:
            video_info += f"\n  摘要: {content}"
            summary_text = ((summary_json or {}).get("summary") or "").strip() if isinstance(summary_json, dict) else ""
            if summary_text:
                video_info += f"\n  文字稿总结: {summary_text}"
            relations = (knowledge_graph_json or {}).get("relations") or [] if isinstance(knowledge_graph_json, dict) else []
            if relations:
                relation_preview = "；".join(
                    f"{item.get('source', '')}-{item.get('relation', '')}-{item.get('target', '')}"
                    for item in relations[:3]
                    if item.get("source") and item.get("target")
                )
                if relation_preview:
                    video_info += f"\n  知识图谱: {relation_preview}"
        elif desc:
            short_desc = desc[:100] + "..." if len(desc) > 100 else desc
            video_info += f" ({short_desc})"
        grouped[folder_name].append(video_info)
        seen_bvids.add(bvid)
        sources.append(_build_source_payload({"bvid": bvid, "title": title, "url": f"https://www.bilibili.com/video/{bvid}"}))
    # 构建上下文文本
    context_parts = [f"【{folder_name}】\n" + "\n".join(videos) for folder_name, videos in grouped.items()]
    context = "\n\n".join(context_parts)
    return context, sources

async def _get_video_titles_context(db: AsyncSession, folder_ids: List[int], limit: int = 50) -> str:
    """获取收藏夹名称与视频标题（用于引导问题）"""
    if not folder_ids:
        return ""
    query = (
        select(FavoriteFolder.title.label("folder_title"), VideoCache.bvid, VideoCache.title)
        .join(FavoriteVideo, FavoriteVideo.folder_id == FavoriteFolder.id)
        .join(VideoCache, VideoCache.bvid == FavoriteVideo.bvid, isouter=True)
        .where(FavoriteFolder.id.in_(folder_ids))
        .limit(limit)
    )
    result = await db.execute(query)
    records = result.fetchall()
    if not records:
        return ""
    grouped = {}
    seen_bvids = set()
    for folder_title, bvid, title in records:
        if not title or not bvid:
            continue
        if bvid in seen_bvids:
            continue
        seen_bvids.add(bvid)
        folder_name = folder_title or "默认收藏夹"
        grouped.setdefault(folder_name, []).append(f"- 《{title}》")
    context_parts = [f"【{folder_name}】\n" + "\n".join(videos) for folder_name, videos in grouped.items()]
    return "\n\n".join(context_parts)


def _resolve_task(request: ChatRequest) -> str:
    if request.task != "auto":
        return request.task
    question = request.question.lower()
    if any(term in question for term in ("笔记", "复习提纲", "整理成文字", "文字版整理")):
        return "note"
    if any(term in question for term in ("找回", "找", "哪个视频", "哪些视频", "哪条视频", "出现在哪", "定位片段")):
        return "find"
    return "general"


def _build_find_messages(context: str, question: str) -> list[dict]:
    system = (
        "你帮助用户从已收藏的视频中找回相关内容。只能使用下方编号证据，不能补充外部知识。\n"
        "输出格式：\n"
        "## 找到的收藏视频\n"
        "逐条写视频标题、与问题相关的具体内容、证据编号 [E#]；只有证据给出时间时才写片段时间。"
        "相同视频的多个片段可以合并，但保留对应编号。\n"
        "## 范围与限制\n"
        "说明证据不足、只有标题简介或没有时间戳的情况。若证据与问题无关，明确说未找到可靠命中。\n"
        "不得编造视频、BV 号、时间戳或证据编号；仅标题/简介来源必须标明“仅元数据线索”。\n\n"
        f"编号证据：\n{context}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def _build_note_messages(context: str, question: str, included: int, total: int) -> list[dict]:
    partial = included < total
    title = "部分内容笔记草稿" if partial else "笔记草稿"
    system = (
        "你根据用户指定视频的字幕或 ASR 文字稿，写可核对的笔记草稿。只能使用下方编号片段；"
        "不要根据标题或常识补全未提供的内容。\n"
        f"本次提供 {included}/{total} 个按时间顺序切分的文字稿片段。\n"
        f"标题必须包含“{title}”。\n"
        "输出格式：核心概览；按内容顺序整理的要点；步骤/例子/观点差异（仅有依据时写）；证据与局限。\n"
        "每个关键事实或观点就近标注真实的 [E#]；多视频时标明各观点所属视频，"
        "不要把不同视频的观点当成同一个结论。\n"
        "没有依据的部分省略并在局限中说明；不得编造 BV 号、时间戳或证据编号。\n"
        + ("片段覆盖不完整，必须在局限中明确说明这是部分内容，不能称为完整视频笔记。\n" if partial else "")
        + f"\n编号文字稿片段：\n{context}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def _group_transcript_segments(transcript: list[dict], max_chars: int = 1500) -> list[dict]:
    chunks: list[dict] = []
    texts: list[str] = []
    start_sec = None
    end_sec = None
    length = 0
    for segment in transcript:
        content = (segment.get("text") or "").strip()
        if not content:
            continue
        if texts and length + len(content) > max_chars:
            chunks.append({"text": "\n".join(texts), "start_sec": start_sec, "end_sec": end_sec})
            texts, start_sec, end_sec, length = [], None, None, 0
        if not texts:
            start_sec = segment.get("start")
        texts.append(content)
        end_sec = segment.get("end")
        length += len(content)
    if texts:
        chunks.append({"text": "\n".join(texts), "start_sec": start_sec, "end_sec": end_sec})
    return chunks


def _sample_chunks(chunks: list[dict], limit: int) -> list[dict]:
    if len(chunks) <= limit:
        return chunks
    if limit == 1:
        return [chunks[0]]
    indices = sorted({round(i * (len(chunks) - 1) / (limit - 1)) for i in range(limit)})
    return [chunks[index] for index in indices]


async def _prepare_find_task(request: ChatRequest, db: AsyncSession) -> PreparedChat:
    if not request.session_id:
        raise HTTPException(status_code=401, detail="找回收藏视频需要先登录")
    folder_ids = await _get_folder_ids_for_session(db, request.session_id, request.folder_ids)
    bvids = await _get_bvids_by_folder_ids(db, folder_ids)
    if not bvids:
        return PreparedChat([], [], "当前范围没有已入库的视频，请先选择并入库收藏夹。")

    docs = get_rag_service().search(request.question, k=8, bvids=bvids)
    allowed_bvids = set(bvids)
    context_parts: list[str] = []
    sources: list[dict] = []
    seen: set[tuple] = set()
    for doc in docs:
        meta = doc.metadata or {}
        bvid = meta.get("bvid")
        content = (doc.page_content or "").strip()
        source_key = (bvid, meta.get("start_sec"), meta.get("end_sec"))
        if bvid not in allowed_bvids or not content or source_key in seen:
            continue
        seen.add(source_key)
        evidence_id = f"E{len(sources) + 1}"
        source = _build_source_payload({
            **meta,
            "evidence_id": evidence_id,
            "content_source": meta.get("source") or "unknown",
            "evidence_text": content,
        })
        sources.append(source)
        time_label = f"，约 {source['timestamp']}" if source.get("timestamp") else "，无片段时间"
        context_parts.append(
            f"[{evidence_id}] 《{source['title']}》来源={source['content_source']}{time_label}\n{content}"
        )
    if not sources:
        return PreparedChat([], [], "当前范围没有找到可核对的视频片段；只有基础信息或尚未建立索引的视频可能无法定位。")
    return PreparedChat(_build_find_messages("\n\n".join(context_parts), request.question), sources)


async def _prepare_note_task(request: ChatRequest, db: AsyncSession) -> PreparedChat:
    if not request.session_id:
        raise HTTPException(status_code=401, detail="生成收藏视频笔记需要先登录")
    folder_ids = await _get_folder_ids_for_session(db, request.session_id, request.folder_ids)
    allowed_bvids = set(await _get_bvids_by_folder_ids(db, folder_ids))
    requested_bvids = request.video_bvids or re.findall(r"(?i:BV)[0-9A-Za-z]{10}", request.question)
    selected = list(dict.fromkeys("BV" + bvid[2:] for bvid in requested_bvids))
    if not selected and len(allowed_bvids) == 1:
        selected = list(allowed_bvids)
    if not selected:
        return PreparedChat([], [], "请先指定要整理的 1–3 条收藏视频（可在问题中写 BV 号）；当前不会把整个收藏夹误作单视频笔记。")
    if len(selected) > 3:
        raise HTTPException(status_code=422, detail="一次最多整理 3 条视频")
    if set(selected) - allowed_bvids:
        raise HTTPException(status_code=403, detail="指定视频不属于当前收藏夹范围")

    result = await db.execute(select(VideoCache).where(VideoCache.bvid.in_(selected)))
    caches = {video.bvid: video for video in result.scalars().all()}
    grouped: list[tuple[VideoCache, list[dict]]] = []
    missing: list[str] = []
    for bvid in selected:
        video = caches.get(bvid)
        if not video or video.content_source not in ("subtitle", "asr") or not isinstance(video.transcript_json, list):
            missing.append(bvid)
            continue
        chunks = _group_transcript_segments(video.transcript_json)
        if not chunks:
            missing.append(bvid)
            continue
        grouped.append((video, chunks))
    if missing:
        return PreparedChat([], [], f"这些视频没有可用的带时间轴文字稿，暂不能生成可核对的笔记：{', '.join(missing)}。请先检查字幕或 ASR 入库状态。")

    per_video_limit = max(1, 12 // len(grouped))
    total = sum(len(chunks) for _, chunks in grouped)
    sources: list[dict] = []
    context_parts: list[str] = []
    for video, chunks in grouped:
        for chunk in _sample_chunks(chunks, per_video_limit):
            evidence_id = f"E{len(sources) + 1}"
            source = _build_source_payload({
                "bvid": video.bvid,
                "title": video.title,
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "evidence_id": evidence_id,
                "content_source": video.content_source,
                "evidence_text": chunk["text"],
            })
            sources.append(source)
            time_label = source.get("timestamp") or "无片段时间"
            context_parts.append(f"[{evidence_id}] 《{video.title}》约 {time_label}\n{chunk['text']}")
    notice = (
        f"> 仅基于 {len(sources)}/{total} 个文字稿片段生成，以下为部分内容笔记草稿；可能遗漏未抽取片段。"
        if len(sources) < total else ""
    )
    return PreparedChat(
        _build_note_messages("\n\n".join(context_parts), request.question, len(sources), total),
        sources,
        notice=notice,
    )


async def _prepare_chat(request: ChatRequest, db: AsyncSession) -> PreparedChat:
    task = _resolve_task(request)
    if task == "find":
        return await _prepare_find_task(request, db)
    if task == "note":
        return await _prepare_note_task(request, db)
    messages, sources, _ = await _prepare_messages(request, db)
    return PreparedChat(messages, sources)


async def _prepare_messages(request: ChatRequest, db: AsyncSession) -> tuple[list[dict], List[dict], str]:
    """准备 LLM 消息与来源信息"""
    question = request.question.strip()
    if request.folder_ids and not request.session_id:
        raise HTTPException(status_code=401, detail="指定收藏夹需要先登录")
    folder_ids = []
    if request.session_id:
        folder_ids = await _get_folder_ids_for_session(db, request.session_id, request.folder_ids)
    bvids = await _get_bvids_by_folder_ids(db, folder_ids) if folder_ids else []
    has_data = len(bvids) > 0
    is_collection_intent = _is_collection_intent(question)
    is_general = _is_general_question(question)
    if request.folder_ids:
        is_collection_intent = True
    # 1) LLM 路由优先，失败时降级规则路由
    logger.info(f"路由输入: question={question} folder_ids={folder_ids} has_data={has_data} is_collection_intent={is_collection_intent}")
    route, route_raw = _route_with_llm(question)
    route_source = "LLM"
    related: Optional[bool] = None
    if not route:
        related = await _is_related_to_collection(db, folder_ids, question)
        route = _route_with_rules(question, is_collection_intent, related)
        route_source = "RULE"
    logger.info(f"路由策略: {route_source} => {route}")
    # 纠偏
    if is_general:
        route = "direct"
    # 2) 无数据时处理
    if not has_data:
        if is_collection_intent:
            context, sources = await _get_video_context(db, folder_ids, include_content=False, limit=50)
            if not context:
                context = "（暂无已入库的视频信息，请提醒用户可能需要先进行入库操作）"
            messages = _build_fallback_messages(context, question)
            return messages, sources, question
        messages = _build_direct_messages(question)
        return messages, [], question
    # 3) 直接回答
    if route == "direct":
        title_context = await _get_video_titles_context(db, folder_ids, limit=50)
        messages = _build_direct_messages_with_context(title_context, question) if title_context else _build_direct_messages(question)
        return messages, [], question
    # 4) 列表类问题
    if route == "db_list":
        if related is None:
            related = await _is_related_to_collection(db, folder_ids, question)
        if not related and not is_collection_intent:
            return _build_direct_messages(question), [], question
        context, sources = await _get_video_context(db, folder_ids, include_content=False, limit=50)
        if not context:
            return _build_fallback_messages("（暂无信息，请入库）", question), sources, question
        return _build_db_list_messages(context, question), sources, question
    # 5) 总结类问题
    if route == "db_content":
        if related is None:
            related = await _is_related_to_collection(db, folder_ids, question)
        if not related and not is_collection_intent:
            return _build_direct_messages(question), [], question
        context, sources = await _get_video_context(db, folder_ids, include_content=True, limit=None)
        if not context:
            return _build_fallback_messages("（暂无信息，请入库）", question), sources, question
        return _build_db_summary_messages(context, question), sources, question
    # 6) 检查相关性
    if related is None:
        related = await _is_related_to_collection(db, folder_ids, question)
    if not related and not is_collection_intent:
        return _build_direct_messages(question), [], question
    # 7) 向量检索
    docs = []
    try:
        if bvids:
            docs = get_rag_service().search(question, k=8, bvids=bvids)
    except Exception as e:
        logger.warning(f"向量检索失败: {e}")
    docs = [doc for doc in docs if (doc.metadata or {}).get("bvid") in set(bvids)]
    if docs:
        filtered_docs = _filter_docs_by_keywords(docs, question)
        docs = filtered_docs if filtered_docs else docs
        context_parts, sources, seen_source_keys = [], [], set()
        for doc in docs:
            meta = doc.metadata or {}
            bvid, title, content = meta.get("bvid", ""), meta.get("title", ""), doc.page_content.strip()
            timestamp = meta.get("timestamp_text")
            header = f"【{title}】"
            if timestamp:
                header += f"（约 {timestamp}）"
            if content:
                context_parts.append(f"{header}\n{content}")
            source_key = (bvid, meta.get("start_sec"), meta.get("end_sec"))
            if bvid and source_key not in seen_source_keys:
                seen_source_keys.add(source_key)
                sources.append(_build_source_payload(meta))
        return _build_rag_messages("\n\n---\n\n".join(context_parts), question), sources, question
    # 兜底
    context, sources = await _get_video_context(db, folder_ids, include_content=False, limit=50)
    return _build_fallback_messages(context or "（暂无入库信息）", question), sources, question

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """智能问答"""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        prepared = await _prepare_chat(request, db)
        if prepared.direct_answer is not None:
            return ChatResponse(answer=prepared.direct_answer, sources=prepared.sources)
        client = _get_llm_client()
        response = client.chat.completions.create(model=settings.llm_model, messages=prepared.messages, temperature=0.2)
        answer = response.choices[0].message.content or ""
        if prepared.notice:
            answer = f"{prepared.notice}\n\n{answer}"
        return ChatResponse(answer=answer, sources=prepared.sources)
    except HTTPException: raise
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

@router.post("/ask/stream")
async def ask_question_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """流式问答"""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        prepared = await _prepare_chat(request, db)
        if prepared.direct_answer is not None:
            def generate_direct():
                yield prepared.direct_answer
                yield "\n[[SOURCES_JSON]][]"
            return StreamingResponse(generate_direct(), media_type="text/plain; charset=utf-8")
        client = _get_llm_client()
        def generate():
            if prepared.notice:
                yield f"{prepared.notice}\n\n"
            stream = client.chat.completions.create(model=settings.llm_model, messages=prepared.messages, temperature=0.2, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content: yield delta.content
            yield f"\n[[SOURCES_JSON]]{json.dumps(prepared.sources, ensure_ascii=False)}"
        return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
    except HTTPException: raise
    except Exception as e:
        logger.error(f"流式问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"流式问答失败: {str(e)}")

@router.post("/search")
async def search_videos(
    query: str = Query(...),
    session_id: str = Query(..., description="登录会话 ID"),
    k: int = Query(5, ge=1, le=20),
    folder_ids: Optional[List[int]] = Query(None, description="可重复传入的收藏夹 media_id"),
    db: AsyncSession = Depends(get_db),
):
    """搜索相关视频片段"""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")
    folder_db_ids = await _get_folder_ids_for_session(db, session_id, folder_ids)
    bvids = await _get_bvids_by_folder_ids(db, folder_db_ids)
    if not bvids:
        return {"results": []}
    try:
        rag = get_rag_service()
        docs = rag.search(query, k=k, bvids=bvids)
        allowed_bvids = set(bvids)
        results, seen_sources = [], set()
        for doc in docs:
            meta = doc.metadata or {}
            bvid = meta.get("bvid", "")
            if bvid not in allowed_bvids:
                continue
            source_key = (bvid, meta.get("start_sec"), meta.get("end_sec"))
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            source = _build_source_payload(meta)
            results.append({
                "bvid": bvid,
                "title": source["title"],
                "url": source["url"],
                "timestamp": source["timestamp"],
                "start_sec": source["start_sec"],
                "end_sec": source["end_sec"],
                "content_source": meta.get("source"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        return {"results": results}
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

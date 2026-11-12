"""
Bilibili RAG service.

This module handles:
1. Vector storage and retrieval
2. Q&A over indexed video content
3. Transcript summary and knowledge graph generation
"""

import asyncio
import json
import re
from typing import Any, List, Optional

from loguru import logger
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings
from app.models import VideoContent


class RAGService:
    """Core service for retrieval and transcript artifact generation."""

    def __init__(self, collection_name: str = "bilibili_videos"):
        self.collection_name = collection_name

        try:
            from langchain_community.embeddings import DashScopeEmbeddings

            self.embeddings = DashScopeEmbeddings(
                dashscope_api_key=settings.openai_api_key,
                model=settings.embedding_model,
            )
            logger.info("Initialized embeddings with DashScopeEmbeddings")
        except ImportError:
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.embedding_model,
                check_embedding_ctx_length=False,
            )

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_directory,
        )

        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=0.5,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "，", "；", ".", "!", "?", " "],
        )

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个知识库助手，专门基于用户收藏的 B 站视频内容回答问题。

请遵循以下规则：
1. 根据提供的视频内容回答问题
2. 回答要自然、友好、有条理
3. 可以引用相关的视频标题作为来源
4. 如果多个视频涉及同一主题，请综合它们的内容

视频内容：
{context}
""",
                ),
                ("human", "{question}"),
            ]
        )

        self.fallback_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个友好的助手。用户在使用一个 B 站收藏夹知识库系统。

当前情况：知识库中没有找到与用户问题相关的内容。

请：
1. 友好地回应用户的问题
2. 如果能根据常识简单回答，可以简要回答
3. 建议用户构建更多收藏夹内容，或者换个问法
4. 保持自然，不要生硬
""",
                ),
                ("human", "{question}"),
            ]
        )

        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个视频内容分析助手。请阅读一个视频文字稿片段，并输出 JSON。

要求：
1. 仅输出 JSON，不要输出额外解释
2. JSON 结构必须为 {{"summary":"","key_points":[""]}}
3. summary 用 2-4 句话概括该片段的主要内容
4. key_points 保留 3-5 条最关键事实或观点
5. 不要编造片段中没有出现的信息
""",
                ),
                ("human", "{content}"),
            ]
        )

        self.summary_merge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个视频内容总结合并助手。现在会提供同一个视频多个片段的摘要结果，请整合为全局总结并输出 JSON。

要求：
1. 仅输出 JSON
2. JSON 结构必须为 {{"summary":"","key_points":[""]}}
3. summary 写成 120-220 字的整体摘要，覆盖视频主线
4. key_points 保留 4-6 条不重复的核心要点
5. 优先保留跨片段重复出现或对整体主题最关键的信息
6. 不要加入输入中不存在的新观点
""",
                ),
                ("human", "{content}"),
            ]
        )

        self.kg_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个知识图谱抽取助手。请基于给定视频文字稿片段抽取关键实体、概念和关系，并输出 JSON。

要求：
1. 仅输出 JSON
2. JSON 结构必须为 {{"entities":[{{"name":"","type":"","description":""}}],"relations":[{{"source":"","relation":"","target":"","evidence":""}}]}}
3. 只保留片段中最重要、最明确的实体和关系
4. entities 控制在 3-8 个，relations 控制在 2-8 条
5. evidence 尽量短，且要来自原文
6. 不要编造没有明确依据的关系
""",
                ),
                ("human", "{content}"),
            ]
        )

        self.kg_merge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个知识图谱合并助手。现在会给你同一个视频多个片段抽取出的实体和关系，请去重、归并并输出最终 JSON。

要求：
1. 仅输出 JSON
2. JSON 结构必须为 {{"entities":[{{"name":"","type":"","description":""}}],"relations":[{{"source":"","relation":"","target":"","evidence":""}}]}}
3. 合并同名或含义相近的实体，保留更清晰的 type 和 description
4. 删除重复、过弱或无依据的关系
5. 最终 entities 控制在 6-12 个，relations 控制在 4-12 条
6. 只保留对整段视频最有代表性的结构
""",
                ),
                ("human", "{content}"),
            ]
        )

    def _format_time(self, seconds: Optional[float]) -> Optional[str]:
        if seconds is None:
            return None
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _build_timestamped_url(self, bvid: str, start_sec: Optional[float]) -> str:
        base = f"https://www.bilibili.com/video/{bvid}"
        if start_sec is None:
            return base
        return f"{base}?t={max(0, int(start_sec))}"

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).lower()

    def _safe_trim(self, text: str, max_length: int) -> str:
        text = (text or "").strip()
        if len(text) <= max_length:
            return text
        return text[:max_length].rstrip() + "\n...(truncated)"

    def _chunk_transcript(self, transcript: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        current_segments: list[dict] = []
        current_length = 0

        for segment in transcript or []:
            text = (segment.get("text") or "").strip()
            if not text:
                continue

            projected = current_length + len(text) + (1 if current_segments else 0)
            if current_segments and projected > self.text_splitter._chunk_size:
                chunk_text = "\n".join(item["text"] for item in current_segments).strip()
                if chunk_text:
                    chunks.append(
                        {
                            "text": chunk_text,
                            "start_sec": current_segments[0].get("start"),
                            "end_sec": current_segments[-1].get("end"),
                        }
                    )
                overlap_segments = (
                    current_segments[-2:] if len(current_segments) > 2 else current_segments[-1:]
                )
                current_segments = list(overlap_segments)
                current_length = sum(len(item["text"]) for item in current_segments)

            current_segments.append(segment)
            current_length += len(text) + (1 if current_segments else 0)

        if current_segments:
            chunk_text = "\n".join(item["text"] for item in current_segments).strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "start_sec": current_segments[0].get("start"),
                        "end_sec": current_segments[-1].get("end"),
                    }
                )

        return chunks

    def _format_knowledge_graph(self, knowledge_graph: Optional[dict]) -> Optional[str]:
        if not knowledge_graph:
            return None

        entities = knowledge_graph.get("entities") or []
        relations = knowledge_graph.get("relations") or []
        parts: list[str] = []

        if entities:
            parts.append("## 关键实体")
            for entity in entities[:10]:
                name = entity.get("name") or ""
                entity_type = entity.get("type") or "概念"
                description = entity.get("description") or ""
                line = f"- {name}（{entity_type}）"
                if description:
                    line += f"：{description}"
                parts.append(line)

        if relations:
            parts.append("\n## 关键关系")
            for relation in relations[:12]:
                source = relation.get("source") or ""
                rel = relation.get("relation") or "相关"
                target = relation.get("target") or ""
                evidence = relation.get("evidence") or ""
                line = f"- {source} --{rel}--> {target}"
                if evidence:
                    line += f"（证据：{evidence}）"
                parts.append(line)

        return "\n".join(parts).strip() or None

    def _parse_json_object(self, raw: str) -> dict:
        raw = (raw or "").strip()
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(raw[start : end + 1])
        except Exception as e:
            logger.warning(f"Failed to parse JSON payload: {e}")
        return {}

    def _normalize_summary_result(self, payload: Optional[dict]) -> dict:
        payload = payload if isinstance(payload, dict) else {}
        summary = (payload.get("summary") or "").strip()
        key_points_raw = payload.get("key_points") or []
        key_points: list[str] = []
        seen: set[str] = set()
        summary_norm = self._normalize_text(summary)

        for item in key_points_raw:
            if not isinstance(item, str):
                continue
            cleaned = item.strip().lstrip("-* 0123456789.").strip()
            normalized = self._normalize_text(cleaned)
            if len(cleaned) < 4 or not normalized or normalized in seen:
                continue
            if summary_norm and (normalized in summary_norm or summary_norm in normalized):
                continue
            seen.add(normalized)
            key_points.append(cleaned)
            if len(key_points) >= 6:
                break

        return {"summary": summary, "key_points": key_points}

    def _merge_summary_candidates(self, candidates: list[dict]) -> dict:
        merged_summary = ""
        merged_points: list[str] = []
        seen: set[str] = set()

        for item in candidates:
            normalized = self._normalize_summary_result(item)
            if not merged_summary and normalized["summary"]:
                merged_summary = normalized["summary"]
            for point in normalized["key_points"]:
                key = self._normalize_text(point)
                if key and key not in seen:
                    seen.add(key)
                    merged_points.append(point)
                if len(merged_points) >= 6:
                    break
            if merged_summary and len(merged_points) >= 5:
                break

        if not merged_summary and merged_points:
            merged_summary = "；".join(merged_points[:3])

        return {"summary": merged_summary, "key_points": merged_points[:6]}

    def _normalize_graph_result(self, payload: Optional[dict]) -> dict:
        payload = payload if isinstance(payload, dict) else {}
        entities_raw = payload.get("entities") or []
        relations_raw = payload.get("relations") or []
        generic_relations = {
            "related",
            "相关",
            "有关",
            "提到",
            "说明",
            "介绍",
            "涉及",
            "包含",
            "包括",
            "联系",
        }
        generic_entities = {
            "视频",
            "内容",
            "作者",
            "观众",
            "大家",
            "东西",
            "方法",
            "问题",
            "过程",
            "结果",
            "信息",
            "数据",
            "部分",
            "这个",
            "那个",
            "我们",
            "你们",
            "他们",
        }

        def is_valid_entity_name(name: str) -> bool:
            if not name:
                return False
            if len(name) < 2 or len(name) > 24:
                return False
            normalized = self._normalize_text(name)
            if normalized in generic_entities:
                return False
            if re.search(r"[，。！？；：,.!?;]", name):
                return False
            if name.count(" ") > 2:
                return False
            return True

        entities: list[dict] = []
        entity_seen: set[str] = set()
        for item in entities_raw:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip()
            if not is_valid_entity_name(name):
                continue
            key = self._normalize_text(name)
            if key in entity_seen:
                continue
            entity_seen.add(key)
            entities.append(
                {
                    "name": name,
                    "type": (item.get("type") or "").strip(),
                    "description": (item.get("description") or "").strip(),
                }
            )

        relations: list[dict] = []
        relation_seen: set[tuple[str, str, str]] = set()
        for item in relations_raw:
            if not isinstance(item, dict):
                continue
            source = (item.get("source") or "").strip()
            relation = (item.get("relation") or "").strip()
            target = (item.get("target") or "").strip()
            evidence = (item.get("evidence") or "").strip()
            if not source or not target:
                continue
            if not is_valid_entity_name(source) or not is_valid_entity_name(target):
                continue
            if self._normalize_text(source) == self._normalize_text(target):
                continue
            if len(relation) <= 1 or len(relation) > 20:
                continue
            if re.search(r"[，。！？；：,.!?;]", relation):
                continue
            if self._normalize_text(relation) in generic_relations:
                continue
            relation_key = (
                self._normalize_text(source),
                self._normalize_text(relation or "related"),
                self._normalize_text(target),
            )
            if relation_key in relation_seen:
                continue
            relation_seen.add(relation_key)
            relations.append(
                {
                    "source": source,
                    "relation": relation or "related",
                    "target": target,
                    "evidence": evidence,
                }
            )

        return {"entities": entities, "relations": relations}

    def _merge_graph_candidates(self, candidates: list[dict]) -> dict:
        entity_best: dict[str, dict[str, Any]] = {}
        relation_best: dict[tuple[str, str, str], dict[str, Any]] = {}

        for candidate in candidates:
            normalized = self._normalize_graph_result(candidate)
            for entity in normalized["entities"]:
                key = self._normalize_text(entity["name"])
                if not key:
                    continue
                existing = entity_best.get(key)
                score = (1 if entity.get("type") else 0) + (1 if entity.get("description") else 0)
                if existing is None or score > existing["score"]:
                    entity_best[key] = {"score": score, "entity": entity}
            for relation in normalized["relations"]:
                rel_key = (
                    self._normalize_text(relation["source"]),
                    self._normalize_text(relation["relation"]),
                    self._normalize_text(relation["target"]),
                )
                if not all(rel_key):
                    continue
                bucket = relation_best.setdefault(
                    rel_key,
                    {"relation": relation, "count": 0, "score": 0},
                )
                bucket["count"] += 1
                bucket["score"] = bucket["count"] * 10 + (3 if relation.get("evidence") else 0)
                if relation.get("evidence") and not bucket["relation"].get("evidence"):
                    bucket["relation"] = relation

        entities = [
            item["entity"]
            for item in sorted(
                entity_best.values(),
                key=lambda item: (item["score"], len(item["entity"].get("name") or "")),
                reverse=True,
            )
        ]
        relations = [
            item["relation"]
            for item in sorted(
                relation_best.values(),
                key=lambda item: (item["score"], len(item["relation"].get("relation") or "")),
                reverse=True,
            )
        ]

        return {"entities": entities[:10], "relations": relations[:8]}

    async def _invoke_json_prompt(self, prompt: ChatPromptTemplate, content: str) -> dict:
        chain = (
            {"content": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        raw = await chain.ainvoke(content)
        return self._parse_json_object(raw)

    async def _summarize_chunk(self, content: str) -> dict:
        payload = await self._invoke_json_prompt(self.summary_prompt, self._safe_trim(content, 8000))
        return self._normalize_summary_result(payload)

    async def _merge_chunk_summaries(self, title: str, chunk_summaries: list[dict]) -> dict:
        if not chunk_summaries:
            return {"summary": "", "key_points": []}
        if len(chunk_summaries) == 1:
            return self._normalize_summary_result(chunk_summaries[0])

        lines: list[str] = [f"视频标题：{title}"]
        for idx, item in enumerate(chunk_summaries, start=1):
            summary = (item.get("summary") or "").strip()
            key_points = item.get("key_points") or []
            lines.append(f"\n片段{idx}摘要：{summary}")
            for point in key_points[:5]:
                lines.append(f"- {point}")

        payload = await self._invoke_json_prompt(
            self.summary_merge_prompt,
            self._safe_trim("\n".join(lines), 12000),
        )
        merged = self._normalize_summary_result(payload)
        if not merged["summary"] or not merged["key_points"]:
            fallback = self._merge_summary_candidates(chunk_summaries)
            if not merged["summary"]:
                merged["summary"] = fallback["summary"]
            if not merged["key_points"]:
                merged["key_points"] = fallback["key_points"]
        return merged

    async def _extract_chunk_knowledge_graph(self, content: str) -> dict:
        payload = await self._invoke_json_prompt(self.kg_prompt, self._safe_trim(content, 9000))
        return self._normalize_graph_result(payload)

    async def _merge_chunk_graphs(self, title: str, graphs: list[dict]) -> dict:
        if not graphs:
            return {"entities": [], "relations": []}
        if len(graphs) == 1:
            return self._normalize_graph_result(graphs[0])

        merged = self._merge_graph_candidates(graphs)
        lines: list[str] = [f"视频标题：{title}", "候选实体："]

        for entity in merged["entities"][:20]:
            line = f'- {entity["name"]}'
            if entity.get("type"):
                line += f' | {entity["type"]}'
            if entity.get("description"):
                line += f' | {entity["description"]}'
            lines.append(line)

        lines.append("\n候选关系：")
        for relation in merged["relations"][:24]:
            line = f'- {relation["source"]} --{relation["relation"]}--> {relation["target"]}'
            if relation.get("evidence"):
                line += f' | {relation["evidence"]}'
            lines.append(line)

        payload = await self._invoke_json_prompt(
            self.kg_merge_prompt,
            self._safe_trim("\n".join(lines), 12000),
        )
        normalized = self._normalize_graph_result(payload)
        if not normalized["entities"] and not normalized["relations"]:
            normalized = merged
        return normalized

    def add_video_content(self, video: VideoContent) -> int:
        title = video.title or "未知标题"
        content_parts: List[str] = []

        if video.content and video.content.strip():
            content_parts.append(video.content.strip())

        if video.summary:
            summary_text = (video.summary.get("summary") or "").strip()
            if summary_text:
                content_parts.append(f"## 内容总结\n{summary_text}")

            key_points = video.summary.get("key_points") or []
            if key_points:
                points_text = "\n".join(f"- {point}" for point in key_points if point)
                if points_text:
                    content_parts.append(f"## 核心要点\n{points_text}")

        knowledge_graph_text = self._format_knowledge_graph(video.knowledge_graph)
        if knowledge_graph_text:
            content_parts.append(knowledge_graph_text)

        if video.outline:
            outline_text = "\n## 内容提纲\n"
            for item in video.outline:
                item_title = item.get("title", "") or ""
                outline_text += f"\n### {item_title}\n"
                for point in item.get("points", []):
                    point_content = point.get("content", "") or ""
                    if point_content:
                        outline_text += f"- {point_content}\n"
            if outline_text.strip() != "## 内容提纲":
                content_parts.append(outline_text)

        full_content = "\n\n".join(content_parts).strip()
        if not full_content or len(full_content.strip()) < 10:
            logger.warning(f"[{video.bvid}] content too short, skip indexing")
            return 0

        chunk_entries: list[dict] = []
        if video.transcript:
            chunk_entries = self._chunk_transcript(video.transcript)

        if not chunk_entries:
            chunks = self.text_splitter.split_text(full_content)
            chunk_entries = [
                {"text": chunk.strip(), "start_sec": None, "end_sec": None}
                for chunk in chunks
            ]

        if not chunk_entries:
            logger.warning(f"[{video.bvid}] no chunks generated")
            return 0

        valid_chunks = [c for c in chunk_entries if c.get("text") and len(c["text"].strip()) > 5]
        if not valid_chunks:
            logger.warning(f"[{video.bvid}] no valid chunks generated")
            return 0

        documents = []
        for i, chunk in enumerate(valid_chunks):
            start_sec = chunk.get("start_sec")
            end_sec = chunk.get("end_sec")
            doc = Document(
                page_content=chunk["text"].strip(),
                metadata={
                    "bvid": video.bvid,
                    "title": title,
                    "source": video.source.value,
                    "chunk_index": i,
                    "url": self._build_timestamped_url(video.bvid, start_sec),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "timestamp_text": self._format_time(start_sec),
                },
            )
            documents.append(doc)

        try:
            batch_size = 10
            for idx in range(0, len(documents), batch_size):
                self.vectorstore.add_documents(documents[idx : idx + batch_size])
            logger.info(f"[{video.bvid}] indexed {len(documents)} chunks")
        except Exception as e:
            logger.error(f"[{video.bvid}] failed to add documents to vector store: {e}")
            raise

        return len(documents)

    def add_videos_batch(self, videos: List[VideoContent], progress_callback=None) -> dict:
        success = 0
        failed = 0
        total_chunks = 0

        for i, video in enumerate(videos):
            try:
                chunks = self.add_video_content(video)
                total_chunks += chunks
                success += 1
                if progress_callback:
                    progress_callback(i + 1, len(videos), video.title)
            except Exception as e:
                logger.error(f"Failed to add video [{video.bvid}]: {e}")
                failed += 1

        return {"success": success, "failed": failed, "chunks": total_chunks}

    def search(self, query: str, k: int = 5, bvids: Optional[List[str]] = None) -> List[Document]:
        if not query or not query.strip():
            logger.warning("Search query is empty")
            return []

        try:
            if bvids:
                docs = self.vectorstore.similarity_search(query, k=k, filter={"bvid": {"$in": bvids}})
            else:
                docs = self.vectorstore.similarity_search(query, k=k)

            logger.info(f"Search completed: query='{query}', hits={len(docs)}")
            for idx, doc in enumerate(docs):
                meta = doc.metadata or {}
                preview = doc.page_content[:120].replace("\n", " ").strip()
                logger.info(
                    f"Hit[{idx + 1}] {meta.get('bvid', '')} #{meta.get('chunk_index', '')} "
                    f"{meta.get('title', '')} | {preview}"
                )
            return docs
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _fallback_answer(self, question: str, reason: str = "") -> dict:
        try:
            chain = (
                {"question": RunnablePassthrough()}
                | self.fallback_prompt
                | self.llm
                | StrOutputParser()
            )

            answer = await chain.ainvoke(question)
            return {"answer": answer, "sources": []}
        except Exception as e:
            logger.error(f"Fallback answer failed: {e}")
            return {
                "answer": f"抱歉，{reason}。您可以尝试构建更多收藏夹内容，或者换个问法试试。",
                "sources": [],
            }

    async def answer_question(self, question: str, k: int = 5, bvids: Optional[List[str]] = None) -> dict:
        stats = self.get_collection_stats()
        if stats["total_chunks"] == 0:
            return await self._fallback_answer(question, "知识库目前还没有内容")

        try:
            docs = self.search(question, k=k, bvids=bvids if bvids else None)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return await self._fallback_answer(question, "检索时遇到问题")

        if not docs:
            return await self._fallback_answer(question, "没有找到相关内容")

        context_parts = []
        seen_bvids = set()
        sources = []

        for doc in docs:
            bvid = doc.metadata.get("bvid", "")
            title = doc.metadata.get("title", "未知标题")
            content = doc.page_content.strip()

            if content:
                context_parts.append(f"【{title}】\n{content}")

            if bvid and bvid not in seen_bvids:
                seen_bvids.add(bvid)
                sources.append(
                    {
                        "bvid": bvid,
                        "title": title,
                        "url": doc.metadata.get("url", f"https://www.bilibili.com/video/{bvid}"),
                    }
                )

        if not context_parts:
            return {
                "answer": "检索到了相关视频，但没有找到有效的文本内容。可能是视频还未完成内容提取。",
                "sources": sources,
            }

        context = "\n\n---\n\n".join(context_parts)
        if not context.strip():
            return {"answer": "没有找到可用的内容来回答您的问题。", "sources": sources}

        try:
            chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            answer = await chain.ainvoke(question)
            return {"answer": answer, "sources": sources}
        except Exception as e:
            logger.error(f"LLM answer failed: {e}")
            return {"answer": f"AI 回答时发生错误: {str(e)}", "sources": sources}

    async def summarize_content(self, content: str) -> str:
        summary = await self._summarize_chunk(content)
        if summary["summary"]:
            return summary["summary"]
        if summary["key_points"]:
            return "；".join(summary["key_points"][:3])
        return ""

    async def extract_knowledge_graph(self, content: str) -> dict:
        return await self._extract_chunk_knowledge_graph(content)

    async def build_transcript_artifacts(self, title: str, transcript: list[dict]) -> dict:
        chunks = self._chunk_transcript(transcript)
        if not chunks:
            text = "\n".join(
                (item.get("text") or "").strip()
                for item in (transcript or [])
                if (item.get("text") or "").strip()
            ).strip()
            if not text:
                return {"summary": None, "knowledge_graph": {"entities": [], "relations": []}}
            chunks = [{"text": text, "start_sec": None, "end_sec": None}]

        chunk_payloads = [f"视频标题：{title}\n\n片段{i + 1}：\n{chunk['text']}" for i, chunk in enumerate(chunks)]

        summary_candidates, graph_candidates = await asyncio.gather(
            asyncio.gather(*(self._summarize_chunk(payload) for payload in chunk_payloads)),
            asyncio.gather(*(self._extract_chunk_knowledge_graph(payload) for payload in chunk_payloads)),
        )

        final_summary, final_graph = await asyncio.gather(
            self._merge_chunk_summaries(title, list(summary_candidates)),
            self._merge_chunk_graphs(title, list(graph_candidates)),
        )

        normalized_summary = self._normalize_summary_result(final_summary)
        if not normalized_summary["summary"] and not normalized_summary["key_points"]:
            normalized_summary = self._merge_summary_candidates(list(summary_candidates))

        normalized_graph = self._normalize_graph_result(final_graph)
        if not normalized_graph["entities"] and not normalized_graph["relations"]:
            normalized_graph = self._merge_graph_candidates(list(graph_candidates))

        summary_result = None
        if normalized_summary["summary"] or normalized_summary["key_points"]:
            summary_result = {
                "summary": normalized_summary["summary"] or None,
                "key_points": normalized_summary["key_points"][:5],
            }

        return {
            "summary": summary_result,
            "knowledge_graph": {
                "entities": normalized_graph["entities"][:8],
                "relations": normalized_graph["relations"][:8],
            },
        }

    def get_collection_stats(self) -> dict:
        try:
            collection = self.vectorstore._collection
            count = collection.count()

            result = collection.get(include=["metadatas"])
            bvids = set()
            for meta in result.get("metadatas", []):
                if meta and "bvid" in meta:
                    bvids.add(meta["bvid"])

            return {
                "total_chunks": count,
                "total_videos": len(bvids),
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_chunks": 0,
                "total_videos": 0,
                "collection_name": self.collection_name,
            }

    def clear_collection(self):
        try:
            self.vectorstore._collection.delete(where={})
            logger.info(f"Cleared vector collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear vector collection: {e}")
            raise

    def delete_video(self, bvid: str):
        try:
            self.vectorstore._collection.delete(where={"bvid": bvid})
            logger.info(f"Deleted video vectors: {bvid}")
        except Exception as e:
            logger.error(f"Failed to delete video [{bvid}]: {e}")
            raise

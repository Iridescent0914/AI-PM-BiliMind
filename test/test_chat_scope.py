"""问答范围与证据边界的本地回归测试，不调用真实模型。"""

import unittest
from unittest.mock import patch

from fastapi import HTTPException
from langchain.schema import Document
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models import Base, ChatRequest, FavoriteFolder, FavoriteVideo, UserSession, VideoCache
from app.routers import chat


class FakeRag:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def search(self, query, k=5, bvids=None):
        self.calls.append((query, k, bvids))
        return self.documents


class ChatScopeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.db = self.session_factory()
        self.db.add_all([
            UserSession(session_id="user-a", bili_mid=11, is_valid=True),
            UserSession(session_id="user-b", bili_mid=22, is_valid=True),
            UserSession(session_id="expired", bili_mid=33, is_valid=False),
            FavoriteFolder(id=1, session_id="user-a", media_id=101, title="A 收藏夹"),
            FavoriteFolder(id=2, session_id="user-b", media_id=202, title="B 收藏夹"),
            FavoriteVideo(folder_id=1, bvid="BV_A"),
            FavoriteVideo(folder_id=2, bvid="BV_B"),
            VideoCache(
                bvid="BV_A",
                title="A 视频",
                content_source="subtitle",
                transcript_json=[
                    {"text": "这是 A 视频第一段的知识点。", "start": 10, "end": 20},
                    {"text": "这是 A 视频第二段的例子。", "start": 21, "end": 30},
                ],
            ),
            VideoCache(bvid="BV_B", title="B 视频", content_source="basic_info"),
        ])
        await self.db.commit()

    async def asyncTearDown(self):
        await self.db.close()
        await self.engine.dispose()

    async def test_search_filters_even_if_vector_store_returns_other_users_document(self):
        fake = FakeRag([
            Document(page_content="A 的证据", metadata={"bvid": "BV_A", "title": "A 视频", "start_sec": 10}),
            Document(page_content="B 的私有证据", metadata={"bvid": "BV_B", "title": "B 视频", "start_sec": 20}),
        ])
        with patch.object(chat, "get_rag_service", return_value=fake):
            response = await chat.search_videos(
                query="知识点", session_id="user-a", k=5, folder_ids=[101], db=self.db,
            )
        self.assertEqual([item["bvid"] for item in response["results"]], ["BV_A"])
        self.assertEqual(fake.calls[0][2], ["BV_A"])

    async def test_search_rejects_other_folder_and_expired_session(self):
        with self.assertRaises(HTTPException) as forbidden:
            await chat.search_videos(
                query="知识点", session_id="user-a", k=5, folder_ids=[202], db=self.db,
            )
        self.assertEqual(forbidden.exception.status_code, 403)
        with self.assertRaises(HTTPException) as expired:
            await chat.search_videos(
                query="知识点", session_id="expired", k=5, folder_ids=None, db=self.db,
            )
        self.assertEqual(expired.exception.status_code, 401)

    async def test_find_sources_have_reproducible_evidence_ids(self):
        fake = FakeRag([
            Document(page_content="技术趋势片段", metadata={
                "bvid": "BV_A", "title": "A 视频", "start_sec": 10, "end_sec": 20, "source": "subtitle",
            }),
        ])
        with patch.object(chat, "get_rag_service", return_value=fake):
            prepared = await chat._prepare_chat(
                ChatRequest(question="找 AI 趋势视频", session_id="user-a", folder_ids=[101]), self.db,
            )
        self.assertEqual(prepared.sources[0]["evidence_id"], "E1")
        self.assertEqual(prepared.sources[0]["url"], "https://www.bilibili.com/video/BV_A?t=10")
        self.assertIn("[E1]", prepared.messages[0]["content"])

    async def test_note_requires_scoped_transcript_and_reports_coverage(self):
        with self.assertRaises(HTTPException) as forbidden:
            await chat._prepare_chat(
                ChatRequest(question="生成笔记", session_id="user-a", folder_ids=[101], task="note", video_bvids=["BV_B"]),
                self.db,
            )
        self.assertEqual(forbidden.exception.status_code, 403)

        prepared = await chat._prepare_chat(
            ChatRequest(question="生成笔记", session_id="user-a", folder_ids=[101], task="note", video_bvids=["BV_A"]),
            self.db,
        )
        self.assertIsNone(prepared.direct_answer)
        self.assertEqual(prepared.sources[0]["content_source"], "subtitle")
        self.assertIn("[E1]", prepared.messages[0]["content"])
        self.assertIn("1/1", prepared.messages[0]["content"])

    async def test_note_refuses_basic_information_only(self):
        prepared = await chat._prepare_chat(
            ChatRequest(question="生成笔记", session_id="user-b", folder_ids=[202], task="note", video_bvids=["BV_B"]),
            self.db,
        )
        self.assertIn("没有可用的带时间轴文字稿", prepared.direct_answer)
        self.assertEqual(prepared.sources, [])

    async def test_long_note_is_explicitly_partial(self):
        video = (await self.db.execute(select(VideoCache).where(VideoCache.bvid == "BV_A"))).scalar_one()
        video.transcript_json = [
            {"text": f"第 {i} 段。" + "知识" * 800, "start": i * 10, "end": i * 10 + 9}
            for i in range(15)
        ]
        await self.db.commit()
        prepared = await chat._prepare_chat(
            ChatRequest(question="生成笔记", session_id="user-a", task="note", video_bvids=["BV_A"]),
            self.db,
        )
        self.assertEqual(len(prepared.sources), 12)
        self.assertIn("部分内容笔记草稿", prepared.messages[0]["content"])
        self.assertIn("12/15", prepared.messages[0]["content"])
        self.assertIn("12/15", prepared.notice)

    async def test_legacy_vector_route_also_filters_foreign_document(self):
        fake = FakeRag([
            Document(page_content="B 的私有证据", metadata={"bvid": "BV_B", "title": "B 视频"}),
        ])
        with patch.object(chat, "get_rag_service", return_value=fake), patch.object(chat, "_route_with_llm", return_value=("vector", "vector")):
            _, sources, _ = await chat._prepare_messages(
                ChatRequest(question="视频里讲了什么", session_id="user-a", folder_ids=[101]), self.db,
            )
        self.assertTrue(all(source["bvid"] == "BV_A" for source in sources))


if __name__ == "__main__":
    unittest.main()

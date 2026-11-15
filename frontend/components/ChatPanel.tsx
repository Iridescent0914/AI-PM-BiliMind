"use client";

import { useState, useRef, useEffect, useLayoutEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { chatApi, knowledgeApi, KnowledgeStats, ChatSource, API_BASE_URL } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
}

interface Props {
  statsKey?: number;
  sessionId?: string;
  folderIds?: number[];
}

export default function ChatPanel({ statsKey, sessionId, folderIds }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [loadedStorageKey, setLoadedStorageKey] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldFollowRef = useRef(true);
  const marker = "[[SOURCES_JSON]]";
  const sourceLabels: Record<string, string> = {
    subtitle: "字幕",
    asr: "语音转写",
    ai_summary: "AI 摘要（弱证据）",
    basic_info: "标题/简介（弱证据）",
    unknown: "来源未确认",
  };
  const storageKey = sessionId ? `bilimind_chat_messages:${sessionId}` : null;

  useEffect(() => {
    if (!storageKey) {
      setMessages([]);
      setLoadedStorageKey(null);
      return;
    }

    try {
      const saved = window.localStorage.getItem(storageKey);
      const parsed: unknown = saved ? JSON.parse(saved) : [];
      setMessages(
        Array.isArray(parsed) && parsed.every((message) =>
          message && typeof message.id === "string" &&
          (message.role === "user" || message.role === "assistant") &&
          typeof message.content === "string"
        ) ? parsed : []
      );
    } catch {
      setMessages([]);
    }
    setLoadedStorageKey(storageKey);
  }, [storageKey]);

  useEffect(() => {
    if (!storageKey || loadedStorageKey !== storageKey) return;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(messages));
    } catch {
      // 浏览器禁用本地存储或空间不足时，仍允许继续对话。
    }
  }, [messages, storageKey, loadedStorageKey]);

  useEffect(() => {
    knowledgeApi.getStats().then(setStats).catch(() => { });
  }, [statsKey]);

  useLayoutEffect(() => {
    const scrollElement = scrollRef.current;
    if (scrollElement && shouldFollowRef.current) {
      scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  }, [messages]);

  const handleChatScroll = () => {
    const scrollElement = scrollRef.current;
    if (!scrollElement) return;
    const distanceFromBottom = scrollElement.scrollHeight - scrollElement.scrollTop - scrollElement.clientHeight;
    shouldFollowRef.current = distanceFromBottom < 24;
  };

  const clearMessages = () => {
    if (storageKey) {
      try {
        window.localStorage.removeItem(storageKey);
      } catch {
        // 浏览器禁用本地存储时，仍清空当前页面。
      }
    }
    setMessages([]);
  };

  const send = async () => {
    if (!input.trim() || loading) return;
    const q = input.trim();
    setInput("");
    shouldFollowRef.current = true;
    const userId = Date.now().toString();
    const assistantId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: q },
      { id: assistantId, role: "assistant", content: "", sources: [] },
    ]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: q,
          session_id: sessionId,
          folder_ids: folderIds,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error("流式接口不可用");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      let buffer = "";
      let sourcesJson = "";
      let inSources = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunk = decoder.decode(value, { stream: !done });
          if (chunk) {
            if (inSources) {
              sourcesJson += chunk;
            } else {
              buffer += chunk;
              const markerIndex = buffer.indexOf(marker);
              if (markerIndex !== -1) {
                const contentPart = buffer.slice(0, markerIndex);
                sourcesJson = buffer.slice(markerIndex + marker.length);
                buffer = contentPart;
                inSources = true;
              }
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: buffer } : m
                )
              );
            }
          }
        }
      }

      if (sourcesJson) {
        try {
          const parsed = JSON.parse(sourcesJson);
          if (Array.isArray(parsed)) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, sources: parsed } : m
              )
            );
          }
        } catch {
          // 忽略解析错误，避免影响主文本
        }
      }
    } catch {
      try {
        const res = await chatApi.ask(q, sessionId, folderIds);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: res.answer, sources: res.sources } : m
          )
        );
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: `错误: ${err instanceof Error ? err.message : "请求失败"}`,
                }
              : m
          )
        );
      }
    }
    setLoading(false);
  };

  return (
    <div className="panel-inner">
      <div className="panel-header">
        <div>
          <div className="panel-title">对话工作台</div>
          {stats && stats.total_videos > 0 && (
            <div className="panel-subtitle">已收录 {stats.total_videos} 个视频</div>
          )}
        </div>
        {messages.length > 0 && (
          <button onClick={clearMessages} className="btn btn-ghost" title="清空">
            清空对话
          </button>
        )}
      </div>

      <div className="panel-body">
        <div className="chat-scroll" ref={scrollRef} onScroll={handleChatScroll}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <div>
                <div className="status-pill">检索就绪</div>
                <p className="text-sm text-[var(--muted)] mt-3">把收藏夹变成可提问的知识库</p>
              </div>
              <div className="prompt-grid">
                {[
                  "总结收藏夹里最有价值的内容",
                  "有哪些适合快速复习的系列？",
                  "列出与某个主题相关的视频并给出关键点",
                  "按主题整理我的收藏夹内容",
                  "用一句话概括每个视频的重点",
                  "推荐3个最适合入门的学习视频",
                ].map((q, i) => (
                  <button key={i} onClick={() => setInput(q)} className="prompt-chip">
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="chat-window">
              {messages.map((m) => (
                <div key={m.id} className={`message ${m.role}`}>
                  <div className="message-bubble">
                    <ReactMarkdown className="markdown" remarkPlugins={[remarkGfm]}>
                      {m.content}
                    </ReactMarkdown>
                    {m.sources && m.sources.length > 0 && (
                      <div className="source-list">
                        {m.sources.map((s, i) => (
                          <a key={i} href={s.url} target="_blank" rel="noopener noreferrer" className="source-link">
                            {s.evidence_id ? `[${s.evidence_id}] ` : ""}{s.title}{s.timestamp ? ` · ${s.timestamp}` : ""}{s.content_source ? ` · ${sourceLabels[s.content_source] || s.content_source}` : ""}
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message assistant">
                  <div className="message-bubble">
                    <div className="flex gap-1">
                      {[0, 1, 2].map((i) => (
                        <div key={i} className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: `${i * 0.15}s` }} />
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="panel-footer">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
            placeholder="输入问题..."
            className="input"
          />
          <button onClick={send} disabled={!input.trim() || loading} className="btn btn-primary">
            发送
          </button>
        </div>
      </div>
    </div>
  );
}

# 固定题集与人工复核

题集为 UTF-8 JSON。每条至少有唯一 `id` 和真实的 `question`。`task` 可为 `auto`、`find`、`note`；`folder_ids` 使用收藏夹 `media_id`，`video_bvids` 用于明确指定笔记目标。`expected_bvids` 与 `must_cover` 由人工对原视频标注；没有核对前留空，不把模型回答当作标准答案。

格式示例（仅说明字段，不是已验证的题集）：

```json
{
  "cases": [
    {
      "id": "find-example-01",
      "question": "我收藏的哪些视频讨论近期 AI 发展趋势？",
      "task": "find",
      "folder_ids": [],
      "expected_bvids": [],
      "must_cover": []
    }
  ]
}
```

运行器：`python skills/bilibili-rag-local/scripts/evaluate.py --cases <题集路径>` 先校验题集，不调用模型。用户授权回放后，在本机设置 `BILIMIND_SESSION_ID`，添加 `--run`；报告默认写入 Git 忽略的 `data/eval_runs/`。报告保留原始答案、来源片段、耗时、缺失的预期 BV 号和未知引用编号，并预留 `human_review`。

人工逐条检查：

1. **找回**：视频是否相关、片段是否真的讨论问题、时间链接是否能跳转、只有标题/简介时是否标弱证据、是否混入未选收藏夹。
2. **笔记**：关键内容覆盖、每条论点的 `[E#]` 是否支持、跨视频观点归属、部分文字稿时是否明确标成草稿、是否凭空补充。
3. 在 `human_review` 填 `task_completed`、`evidence_supported`、`failure_stage` 与具体 `notes`。失败阶段用产品计划中的七类：需求/范围理解、内容获取或转写、索引与检索、路由、证据使用、生成、交互呈现。自动检查只提供线索，不替代人工判定。

对照前后版本时固定题集、收藏夹范围、数据快照与模型；另记实际模型调用次数、费用或可得的 Token 数据，未知时留空。

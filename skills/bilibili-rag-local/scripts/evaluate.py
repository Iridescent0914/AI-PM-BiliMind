"""Run fixed BiliMind questions and save evidence for human badcase review."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ALLOWED_TASKS = {"auto", "find", "note"}
FAILURE_STAGES = [
    "需求/范围理解", "内容获取或转写", "索引与检索", "路由", "证据使用", "生成", "交互呈现",
]


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("题集需包含非空 cases 数组")
    seen_ids = set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError(f"第 {index} 条题目不是对象")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError(f"第 {index} 条题目缺少唯一 id")
        seen_ids.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{case_id}: question 不能为空")
        if case.get("task", "auto") not in ALLOWED_TASKS:
            raise ValueError(f"{case_id}: task 只能是 auto、find 或 note")
        for field, kind in (("folder_ids", int), ("video_bvids", str), ("expected_bvids", str), ("must_cover", str)):
            value = case.get(field, [])
            if not isinstance(value, list) or any(type(item) is not kind for item in value):
                raise ValueError(f"{case_id}: {field} 必须是 {kind.__name__} 数组")
        if "session_id" in case:
            raise ValueError(f"{case_id}: 不要将 session_id 写入题集，改用 BILIMIND_SESSION_ID 环境变量")
    return cases


def request_json(url: str, method: str = "GET", body: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_case(case: dict, base_url: str, session_id: str, timeout: int) -> dict:
    request_body = {
        "question": case["question"],
        "session_id": session_id,
        "folder_ids": case.get("folder_ids"),
        "task": case.get("task", "auto"),
        "video_bvids": case.get("video_bvids"),
    }
    started = time.monotonic()
    answer = ""
    sources = []
    error = None
    try:
        response = request_json(f"{base_url}/chat/ask", "POST", request_body, timeout)
        answer = response.get("answer", "")
        sources = response.get("sources", [])
    except HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read(500).decode('utf-8', errors='replace')}"
    except (URLError, TimeoutError, ValueError) as exc:
        error = str(exc)
    elapsed_ms = round((time.monotonic() - started) * 1000)

    source_ids = {item.get("evidence_id") for item in sources if isinstance(item, dict)}
    cited_ids = set(re.findall(r"\[(E\d+)\]", answer))
    actual_bvids = {item.get("bvid") for item in sources if isinstance(item, dict)}
    expected_bvids = set(case.get("expected_bvids", []))
    return {
        "id": case["id"],
        "question": case["question"],
        "task": case.get("task", "auto"),
        "folder_ids": case.get("folder_ids"),
        "video_bvids": case.get("video_bvids"),
        "expected_bvids": case.get("expected_bvids", []),
        "must_cover": case.get("must_cover", []),
        "latency_ms": elapsed_ms,
        "answer": answer,
        "sources": sources,
        "error": error,
        "automatic_checks": {
            "missing_expected_bvids": sorted(expected_bvids - actual_bvids),
            "unknown_citations": sorted(cited_ids - source_ids),
        },
        "human_review": {
            "task_completed": None,
            "evidence_supported": None,
            "failure_stage": None,
            "notes": "",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="固定问题回放与来源/badcase 记录；默认只验证题集。")
    parser.add_argument("--cases", type=Path, required=True, help="JSON 题集路径")
    parser.add_argument("--output", type=Path, help="报告路径；默认在 data/eval_runs 下")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--run", action="store_true", help="实际调用本地模型接口；会产生模型费用")
    args = parser.parse_args()

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"题集有效：{len(cases)} 条。可人工标注的失败阶段：{'、'.join(FAILURE_STAGES)}")
    if not args.run:
        print("未调用模型；添加 --run 才会执行。")
        return 0

    parsed_url = urlparse(args.base_url)
    if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("只允许访问本机 HTTP 服务")
    session_id = os.environ.get("BILIMIND_SESSION_ID", "").strip()
    if not session_id:
        parser.error("实际执行前需设置 BILIMIND_SESSION_ID 环境变量")
    base_url = args.base_url.rstrip("/")
    try:
        stats = request_json(f"{base_url}/knowledge/stats", timeout=min(args.timeout, 10))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        parser.error(f"本地后端不可用：{exc}")

    results = []
    for case in cases:
        result = run_case(case, base_url, session_id, args.timeout)
        results.append(result)
        print(f"{case['id']}: {result['latency_ms']} ms，来源 {len(result['sources'])} 条" + (f"，错误 {result['error']}" if result["error"] else ""))

    output = args.output or Path("data/eval_runs") / f"run-{datetime.now():%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_file_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "base_url": base_url,
        "knowledge_stats": stats,
        "results": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入：{output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

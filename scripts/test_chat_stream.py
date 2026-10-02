# -*- coding: utf-8 -*-
"""SSE 流式对话接口测试：事件序列、工单流、追问流、错误流。"""
import json
import os
from unittest.mock import patch

os.environ["AGENT_CONVERSATION_DB_PATH"] = ":memory:"
os.environ["AGENT_SUPPORT_DB_PATH"] = ":memory:"

from fastapi.testclient import TestClient

from app.agent_policy import AgentDecision
from app.main import app, conversation_store

client = TestClient(app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 文本解析为 (event, data) 列表。"""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if event_name:
            events.append((event_name, data))
    return events


def make_service_result(answer: str, references: int = 1) -> dict:
    records = [
        {
            "chunk_id": f"stream_test_{i}",
            "product": "健身追踪器",
            "text": "表带更换说明。",
            "image_ids": [],
            "source": "stream_test.jsonl",
            "score": 1.0,
            "retrieval_type": "semantic",
        }
        for i in range(references)
    ]
    return {
        "answer": answer,
        "references": records,
        "image_ids": [],
        "metrics": {
            "llm_called": True,
            "provider": "mock",
            "model": "mock",
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "llm_latency_ms": 5,
            "vision_analysis_called": False,
            "vision_analysis_latency_ms": 0,
            "vision_analysis_prompt_tokens": None,
            "vision_analysis_completion_tokens": None,
            "vision_analysis_total_tokens": None,
            "vision_analysis_succeeded": False,
            "vision_fallback_used": False,
            "vision_analysis_error": None,
            "retrieval_latency_ms": 2,
            "total_latency_ms": 10,
            "retrieved_record_count": references,
            "context_record_count": references,
            "context_length": 30,
            "finish_reason": "stop",
            "selected_image_count": 0,
            "input_image_count": 0,
            "vision_used": False,
        },
    }


def check(name: str, passed: bool) -> bool:
    print(f"{'通过' if passed else '失败'}|{name}")
    return passed


def main() -> None:
    checks: dict[str, bool] = {}

    # 1. 知识类回答：routing -> references -> delta* -> done
    conversation_store.clear("stream-kb-session")
    with patch(
        "app.main.answer_question",
        return_value=make_service_result(
            "第一步松开卡扣。第二步取下旧表带。第三步装上新表带。"
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "question": "健身追踪器的表带怎么更换？",
                "session_id": "stream-kb-session",
            },
        ) as response:
            body = response.read().decode("utf-8")
            status = response.status_code
    events = parse_sse(body)
    names = [name for name, _ in events]
    checks["流式响应200且SSE媒体类型"] = (
        status == 200 and "text/event-stream" in response.headers["content-type"]
    )
    checks["事件序列完整"] = (
        names[0] == "routing"
        and "references" in names
        and names.count("delta") >= 2
        and names[-1] == "done"
    )
    done_payload = events[-1][1]
    checks["done携带完整答案与指标"] = (
        "表带" in done_payload["answer"]
        and done_payload["metrics"]["llm_called"] is True
    )
    deltas = "".join(data["text"] for name, data in events if name == "delta")
    checks["delta拼接等于完整答案"] = deltas == done_payload["answer"]
    checks["routing事件带routed_by"] = events[0][1].get("routed_by") == "rules"

    # 2. 安全转人工：routing -> ticket -> delta -> done
    conversation_store.clear("stream-safety-session")
    with client.stream(
        "POST",
        "/v1/chat/stream",
        json={
            "question": "设备突然冒烟并有烧焦味，我还能继续使用吗？",
            "session_id": "stream-safety-session",
        },
    ) as response:
        body = response.read().decode("utf-8")
    events = parse_sse(body)
    names = [name for name, _ in events]
    ticket_payload = dict(events).get("ticket", {})
    checks["安全流创建工单"] = (
        "ticket" in names
        and ticket_payload.get("priority") == "urgent"
        and ticket_payload.get("ticket_id")
    )
    checks["安全流不推送references"] = "references" not in names
    checks["安全流done状态正常"] = (
        events[-1][0] == "done" and events[-1][1]["status"] == "ok"
    )

    # 3. 追问流：routing -> message -> delta -> done
    conversation_store.clear("stream-clarify-session")
    clarification = AgentDecision(
        intent="clarification",
        action="ask_clarification",
        confidence=0.7,
        handoff_required=False,
        response_message="请补充设备型号。",
    )
    with patch(
        "app.main._route_request",
        return_value=(clarification, "llm"),
    ):
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "question": "这个怎么弄",
                "session_id": "stream-clarify-session",
            },
        ) as response:
            body = response.read().decode("utf-8")
    events = parse_sse(body)
    names = [name for name, _ in events]
    checks["追问流推送message事件"] = (
        "message" in names
        and dict(events)["message"]["text"] == "请补充设备型号。"
    )
    checks["追问流不触发检索与工单"] = (
        "references" not in names and "ticket" not in names
    )

    # 4. 模型错误流：routing -> done(error)
    conversation_store.clear("stream-error-session")
    from app.llm_client import LLMClientError

    with patch(
        "app.main.answer_question",
        side_effect=LLMClientError(
            "大模型接口触发限流，请稍后重试。",
            category="rate_limit",
            retryable=True,
        ),
    ):
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "question": "健身追踪器防水吗？",
                "session_id": "stream-error-session",
            },
        ) as response:
            body = response.read().decode("utf-8")
    events = parse_sse(body)
    checks["模型错误流done携带error"] = (
        events[-1][0] == "done"
        and events[-1][1]["status"] == "error"
        and events[-1][1]["retryable"] is True
    )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"SSE流式对话测试失败：{failed}")
    print("\nSSE流式对话测试：全部通过")


if __name__ == "__main__":
    main()

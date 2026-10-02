# -*- coding: utf-8 -*-
"""LLM 路由模块测试：工具调用映射、护栏判断、配置默认值、对话链路集成。"""
import os
from unittest.mock import patch

os.environ["AGENT_CONVERSATION_DB_PATH"] = ":memory:"
os.environ["AGENT_SUPPORT_DB_PATH"] = ":memory:"

from fastapi.testclient import TestClient

from app.agent_policy import AgentDecision, classify_request
from app.config import Settings, get_settings
from app.llm_router import (
    GUARDRAIL_INTENTS,
    decision_from_tool_call,
    needs_llm_routing,
)
from app.main import app, conversation_store

client = TestClient(app)


def check(name: str, passed: bool) -> bool:
    print(f"{'通过' if passed else '失败'}|{name}")
    return passed


def main() -> None:
    checks: dict[str, bool] = {}

    # 1. 工具调用 -> AgentDecision 映射
    knowledge = decision_from_tool_call(
        "search_knowledge",
        {"question": "怎么更换表带"},
    )
    checks["检索工具映射为知识意图"] = (
        knowledge.intent == "knowledge_question"
        and knowledge.action == "search_knowledge"
        and not knowledge.handoff_required
    )

    ticket_safety = decision_from_tool_call(
        "create_support_ticket",
        {"category": "safety_risk", "issue_summary": "冒烟"},
    )
    checks["安全工单映射正确"] = (
        ticket_safety.intent == "safety_risk"
        and ticket_safety.handoff_required
        and ticket_safety.handoff_reason == "potential_safety_risk"
    )

    ticket_biz = decision_from_tool_call(
        "create_support_ticket",
        {"category": "business_action", "issue_summary": "退款"},
    )
    checks["业务工单映射正确"] = (
        ticket_biz.intent == "business_action"
        and ticket_biz.handoff_reason == "business_system_action_required"
    )

    clarification = decision_from_tool_call(
        "ask_clarification",
        {"missing_information": "设备型号"},
    )
    checks["追问工具映射正确"] = (
        clarification.intent == "clarification"
        and clarification.action == "ask_clarification"
        and not clarification.handoff_required
        and clarification.response_message is not None
    )

    unknown_tool = decision_from_tool_call("bad_tool", {})
    checks["未知工具兜底为知识检索"] = (
        unknown_tool.action == "search_knowledge"
    )

    # 2. hybrid 护栏判断：安全/业务/转人工走硬规则，其余交大模型
    checks["安全意图不需要LLM路由"] = not needs_llm_routing(
        classify_request("设备冒烟了怎么办")
    )
    checks["转人工意图不需要LLM路由"] = not needs_llm_routing(
        classify_request("我要投诉")
    )
    checks["知识问题需要LLM路由"] = needs_llm_routing(
        classify_request("滤网怎么拆下来清洗")
    )
    checks["护栏意图集合完整"] = GUARDRAIL_INTENTS == frozenset(
        {"safety_risk", "human_handoff", "business_action"}
    )

    # 3. 配置默认值：routing_mode 默认 rules
    settings = Settings(
        _env_file=None,
        environment="test",
        preload_embedding_model=False,
    )
    checks["路由模式默认rules"] = settings.routing_mode == "rules"

    # 4. 对话链路集成：llm 模式下路由器返回追问时，接口直接返回追问文案
    conversation_store.clear("router-test-session")
    fake_decision = AgentDecision(
        intent="clarification",
        action="ask_clarification",
        confidence=0.7,
        handoff_required=False,
        response_message="请补充设备型号。",
    )
    with patch("app.main.settings", get_settings().model_copy(update={"routing_mode": "llm"})):
        with patch(
            "app.main.route_with_llm",
            return_value={
                "decision": fake_decision,
                "tool_name": "ask_clarification",
                "usage": {"total_tokens": 100},
                "latency_ms": 12,
            },
        ):
            response = client.post(
                "/chat",
                json={
                    "question": "这个怎么弄",
                    "session_id": "router-test-session",
                },
            )
    checks["追问分支直接返回追问文案"] = (
        response.status_code == 200
        and "请补充设备型号" in response.json()["data"]["answer"]
    )
    checks["追问分支不触发检索和生成"] = (
        response.json()["data"]["metrics"]["llm_called"] is False
        and response.json()["data"]["metrics"]["retrieved_record_count"] == 0
    )

    # 5. 对话链路集成：路由器异常时回退规则路由
    conversation_store.clear("router-fallback-session")
    with patch("app.main.settings", get_settings().model_copy(update={"routing_mode": "llm"})):
        with patch(
            "app.main.route_with_llm",
            side_effect=RuntimeError("路由模型不可用"),
        ):
            response = client.post(
                "/chat",
                json={
                    "question": "健身追踪器的表带怎么更换？",
                    "session_id": "router-fallback-session",
                },
            )
    checks["路由失败回退规则路由仍正常回答"] = (
        response.status_code == 200
        and len(response.json()["data"]["references"]) > 0
    )

    failed = [name for name, passed in checks.items() if not passed]
    for name in failed:
        print(f"失败|{name}")
    if failed:
        raise SystemExit(f"LLM路由测试失败：{failed}")
    print("\nLLM路由测试：全部通过")


if __name__ == "__main__":
    main()

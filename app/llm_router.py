# -*- coding: utf-8 -*-
"""大模型 function-calling 路由：让模型在知识检索/创建工单/追问澄清之间决策。

与 agent_policy.py 的确定性规则构成三档路由模式（见 config.routing_mode）：
- rules  ：只用 agent_policy.classify_request
- llm    ：只用本模块 route_with_llm
- hybrid ：classify_request 判定为安全/业务/转人工时直接短路（硬护栏），
           其余问题交给 route_with_llm

设计约束：
- 工具只有三个，对应客服场景的全部合法动作；证据复核仍由
  agent_policy.review_evidence 在检索之后执行，两层护栏结构不变。
- 任何路由模型调用失败都不阻断服务：调用方捕获异常后回退到
  确定性规则（记 metrics routing_llm_fallback_total）。
"""
import json
from functools import lru_cache
from time import perf_counter
from typing import Any

from openai import OpenAI

from app.agent_policy import AgentDecision
from app.config import Settings

ROUTER_SYSTEM_PROMPT = """你是产品客服系统的请求路由器。你的唯一职责是把用户请求
分派给一个工具，不要回答用户问题本身。

分派规则：
- 用户描述起火、冒烟、冒火星、漏电、触电、被电、烧焦、烧糊、焦味、
  电池鼓包、进水、电线破皮、受伤、烫伤、头晕、恶心等任何安全风险或
  身体伤害迹象时，调用 create_support_ticket，category 选 safety_risk。
  安全判断宁可保守：拿不准时一律按安全风险处理。
- 用户要求退款、退货、换货、换新、差价、发货、到货、发票、收据、
  保修、维修、取消订单等需要业务系统操作的售后，或要求转人工、
  表达投诉不满时，调用 create_support_ticket（对应类别）。
- 用户询问产品使用方法、故障排查、规格参数等知识类问题时，
  调用 search_knowledge，并标注 question_type：
  - 故障、报错、不亮、发烫、卡住、异响、漏水等异常现象 → troubleshooting
  - 怎么安装/更换/拆卸/清洁/调/设置等操作方法 → product_guidance
  - 其余知识咨询 → knowledge_question
- 只有当问题缺少关键信息、直接检索必然无法命中时（例如完全不知道用户
  在问哪个设备、问什么问题），才调用 ask_clarification。
- 宁可交给 search_knowledge，也不要过度追问。

只调用一个工具。"""

ROUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "检索产品手册知识库并回答使用方法、故障排查、规格参数类问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "整理后的检索问题",
                    },
                    "question_type": {
                        "type": "string",
                        "enum": [
                            "troubleshooting",
                            "product_guidance",
                            "knowledge_question",
                        ],
                        "description": "问题类型：故障排查/使用指导/一般知识询问",
                    }
                },
                "required": ["question", "question_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "创建人工工单：安全风险、需要业务系统操作的售后请求、用户要求人工或投诉。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["safety_risk", "business_action", "human_handoff"],
                        "description": "工单类别",
                    },
                    "issue_summary": {
                        "type": "string",
                        "description": "一句话概括用户诉求",
                    }
                },
                "required": ["category", "issue_summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "问题信息严重不足、无法检索也无法分派时，向用户追问缺失信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "missing_information": {
                        "type": "string",
                        "description": "需要用户补充的具体信息",
                    }
                },
                "required": ["missing_information"],
            },
        },
    },
]

HANDOFF_REASON_BY_CATEGORY = {
    "safety_risk": "potential_safety_risk",
    "business_action": "business_system_action_required",
    "human_handoff": "user_requested_human",
}

# hybrid 模式下由硬规则直接短路的意图（安全/转人工/业务操作）
GUARDRAIL_INTENTS = frozenset(
    {"safety_risk", "human_handoff", "business_action"}
)


def decision_from_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
) -> AgentDecision:
    """把模型的工具调用映射为与规则路由同构的 AgentDecision。"""
    if tool_name == "search_knowledge":
        # 细粒度意图由模型随工具参数给出，与规则路由的意图体系对齐
        question_type = str(arguments.get("question_type", "knowledge_question"))
        intent = (
            question_type
            if question_type in {
                "troubleshooting",
                "product_guidance",
                "knowledge_question",
            }
            else "knowledge_question"
        )
        return AgentDecision(
            intent=intent,
            action="search_knowledge",
            confidence=0.75,
            handoff_required=False,
        )
    if tool_name == "create_support_ticket":
        category = str(arguments.get("category", "human_handoff"))
        reason = HANDOFF_REASON_BY_CATEGORY.get(
            category, "user_requested_human"
        )
        return AgentDecision(
            intent=category if category in HANDOFF_REASON_BY_CATEGORY else "human_handoff",
            action="handoff_to_human",
            confidence=0.9,
            handoff_required=True,
            handoff_reason=reason,
            response_message=(
                "已记录您的诉求，将为您转接人工客服处理。"
            ),
        )
    if tool_name == "ask_clarification":
        missing = str(arguments.get("missing_information", "更多问题描述"))
        return AgentDecision(
            intent="clarification",
            action="ask_clarification",
            confidence=0.7,
            handoff_required=False,
            response_message=(
                f"为了更准确地帮您解决问题，请补充以下信息：{missing}"
            ),
        )
    # 模型返回了未定义工具：按知识问题兜底，走检索与证据复核
    return AgentDecision(
        intent="knowledge_question",
        action="search_knowledge",
        confidence=0.5,
        handoff_required=False,
    )


def needs_llm_routing(decision: AgentDecision) -> bool:
    """hybrid 模式判断：护栏意图由硬规则直接短路，其余交给大模型。"""
    return decision.intent not in GUARDRAIL_INTENTS


def route_with_llm(
    question: str,
    settings: Settings,
) -> dict[str, Any]:
    """调用大模型做 function-calling 路由。

    返回 {"decision": AgentDecision, "usage": {...}, "latency_ms": int}。
    模型不可用、返回格式异常等情况抛出异常，由调用方回退规则路由。
    """
    if settings.llm_provider == "mock":
        raise RuntimeError("mock provider 不支持 function-calling 路由。")

    client = _client_for(settings)
    started = perf_counter()

    request_options: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "tools": ROUTER_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0,
    }
    if settings.llm_thinking is not None:
        # 智谱 GLM 需要显式关闭思考模式，否则思考耗尽输出额度
        request_options["extra_body"] = {
            "thinking": {"type": settings.llm_thinking}
        }

    response = client.chat.completions.create(**request_options)
    latency_ms = round((perf_counter() - started) * 1000)

    message = response.choices[0].message
    tool_calls = list(message.tool_calls or [])
    if not tool_calls:
        raise ValueError("路由模型未返回工具调用。")
    tool_call = tool_calls[0]
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("路由模型工具参数不是合法 JSON。") from error
    if not isinstance(arguments, dict):
        raise ValueError("路由模型工具参数必须是对象。")

    decision = decision_from_tool_call(tool_call.function.name, arguments)
    usage = response.usage
    return {
        "decision": decision,
        "tool_name": tool_call.function.name,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
        "latency_ms": latency_ms,
    }


@lru_cache(maxsize=4)
def _get_client(
    api_key: str,
    base_url: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> OpenAI:
    """缓存 OpenAI 兼容客户端，避免每个请求重建连接。"""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _client_for(settings: Settings) -> OpenAI:
    return _get_client(
        api_key=(
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else ""
        ),
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

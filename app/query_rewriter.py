# -*- coding: utf-8 -*-
"""多轮对话检索改写：把依赖上下文的追问改写成可独立检索的完整问题。

动机：混合检索用"对话历史原文 + 当前问题"拼接串作为向量检索查询，
历史内容会稀释当前问题的语义信号，且代词/省略（"它怎么换"）依赖
拼接碰运气。改写器让大模型把追问还原成独立完整的问题，例如
历史在聊健身追踪器时，"充满要多久" 改写为 "健身追踪器充满电需要多长时间"。

失败语义：改写失败（模型不可用/输出为空）由调用方回退到原拼接行为，
不阻断对话。
"""
from time import perf_counter
from typing import Any

from app.config import Settings
from app.conversation_store import ConversationTurn, format_conversation_history
from app.llm_router import client_for

REWRITE_SYSTEM_PROMPT = """你把用户在多轮客服对话中的最新追问，改写成一句
可以独立用于知识库检索的完整问题。

规则：
- 必须把代词、省略还原成明确对象（如"它/这个设备/充满要多久"）。
- 改写结果必须包含具体产品名称或型号（从对话历史中获取），
  不得丢失产品指代，例如历史在聊发电机时，"那个开关怎么设置"
  应改写为"发电机的经济控制开关怎么设置"。
- 只结合对话历史补充指代对象，不要引入历史和当前问题之外的新信息。
- 不要回答问题，只输出改写后的问题本身，一行，不要任何解释或引号。"""


def build_rewrite_messages(
    question: str,
    history: list[ConversationTurn],
) -> list[dict[str, str]]:
    history_text = format_conversation_history(history)
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"对话历史：\n{history_text}\n\n当前问题：{question}\n\n改写后的问题：",
        },
    ]


def rewrite_query_with_history(
    question: str,
    history: list[ConversationTurn],
    settings: Settings,
) -> dict[str, Any]:
    """调用大模型改写检索查询。

    返回 {"query": str, "latency_ms": int, "total_tokens": int|None}。
    模型不可用或输出为空时抛出异常，由调用方回退。
    """
    if settings.llm_provider == "mock":
        raise RuntimeError("mock provider 不支持查询改写。")

    client = client_for(settings)
    started = perf_counter()

    request_options: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": build_rewrite_messages(question, history),
        "max_tokens": 200,
        "temperature": 0,
    }
    if settings.llm_thinking is not None:
        # GLM 系需要显式关闭思考模式
        request_options["extra_body"] = {
            "thinking": {"type": settings.llm_thinking}
        }

    response = client.chat.completions.create(**request_options)
    latency_ms = round((perf_counter() - started) * 1000)

    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("改写模型返回空内容。")
    # 防御：模型偶尔带引号或前缀，截断多行取首行并去引号
    first_line = content.splitlines()[0].strip().strip("\"“”'")
    if not first_line:
        raise ValueError("改写模型返回内容无法解析。")

    usage = response.usage
    return {
        "query": first_line,
        "latency_ms": latency_ms,
        "total_tokens": usage.total_tokens if usage else None,
    }

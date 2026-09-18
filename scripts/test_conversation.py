from unittest.mock import patch

from app.conversation_store import (
    ConversationStore,
    format_conversation_history,
)
from app.llm_client import LLMResult
from app.services.chat_service import answer_question

"""多轮会话记忆模块的端到端单元测试脚本"""

QUESTION = "那拆卸时先做什么？"
OLD_QUESTION = "我有一个健身追踪器，想更换表带。"
OLD_ANSWER = "请先将设备翻面，找到表带卡扣。"

FAKE_RECORD = {
    "chunk_id": "conversation_test_001",
    "product": "健身追踪器",
    "text": "更换表带前，请先找到并松开表带卡扣。",
    "image_ids": [],
    "source": "conversation_test.jsonl",
    "score": 1.0,
    "retrieval_type": "semantic",
}

FAKE_LLM_RESULT = LLMResult(
    answer="请先翻面并找到表带卡扣。",
    image_ids=[],
    provider="openai_compatible",
    model="fake-model",
    prompt_tokens=10,
    completion_tokens=8,
    total_tokens=18,
    latency_ms=5,
    finish_reason="stop",
)


def main() -> None:
    checks: dict[str, bool] = {}

    # 测试 ConversationStore 会话存储逻辑
    store = ConversationStore(
        max_turns=2,
        max_sessions=2,
    )

    store.append_turn(
        "session-001",
        "第一轮问题",
        "第一轮回答",
    )
    store.append_turn(
        "session-001",
        "第二轮问题",
        "第二轮回答",
    )
    store.append_turn(
        "session-001",
        "第三轮问题",
        "第三轮回答",
    )

    stored_history = store.get_history("session-001")
    formatted_history = format_conversation_history(stored_history)
    # 超过最大轮数，自动截断旧对话，只保留最新 N 轮
    checks["会话保存最近两轮"] = (
        len(stored_history) == 2
        and stored_history[0]["question"] == "第二轮问题"
        and stored_history[1]["question"] == "第三轮问题"
    )

    checks["历史格式包含用户和客服角色"] = (
        "用户：第二轮问题" in formatted_history
        and "客服：第二轮回答" in formatted_history
        and "用户：第三轮问题" in formatted_history
    )

    checks["清空会话成功"] = (
        store.clear("session-001") is True and store.get_history("session-001") == []
    )

    # Mock 整条业务链路，测试历史能否透传进检索 + Prompt
    history = [
        {
            "question": OLD_QUESTION,
            "answer": OLD_ANSWER,
        }
    ]

    with (
        patch(
            "app.services.chat_service.hybrid_search",
            return_value=[FAKE_RECORD],
        ) as search_mock,
        patch(
            "app.services.chat_service.generate_answer",
            return_value=FAKE_LLM_RESULT,
        ) as generate_mock,
    ):
        result = answer_question(
            question=QUESTION,
            conversation_history=history,
        )

    search_query = search_mock.call_args.kwargs["query"]
    prompt_package = generate_mock.call_args.kwargs["prompt_package"]
    user_prompt = prompt_package["user_prompt"]

    checks["历史进入检索问题"] = (
        OLD_QUESTION in search_query
        and OLD_ANSWER in search_query
        and QUESTION in search_query
    )

    checks["历史进入最终Prompt"] = (
        OLD_QUESTION in user_prompt
        and OLD_ANSWER in user_prompt
        and QUESTION in user_prompt
    )

    checks["历史只用于理解上下文的提示存在"] = (
        "对话历史只用于理解用户指代和上下文" in user_prompt
    )

    checks["带历史时仍然成功生成回答"] = result["answer"] == FAKE_LLM_RESULT.answer

    print("====多轮会话测试====")

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n多轮会话测试：全部通过")


if __name__ == "__main__":
    main()

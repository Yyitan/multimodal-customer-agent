from unittest.mock import patch

from app.llm_client import LLMResult
from app.services.chat_service import answer_question


QUESTION = "请看图片，这个设备应该怎么更换配件？"
PRODUCT = "健身追踪器"
PRODUCT_HINT_LINE = f"图片识别到的产品线索：{PRODUCT}"

FAKE_RECORD = {
    "chunk_id": "test_fitness_001",
    "product": PRODUCT,
    "text": "更换表带时，请先松开卡扣，再安装新表带。",
    "image_ids": [],
    "source": "test_fitness_manual.jsonl",
    "score": 1.0,
    "retrieval_type": "semantic",
}

FAKE_LLM_RESULT = LLMResult(
    answer="请先松开卡扣，然后安装新表带。",
    image_ids=[],
    provider="openai_compatible",
    model="fake-model",
    prompt_tokens=10,
    completion_tokens=8,
    total_tokens=18,
    latency_ms=5,
    finish_reason="stop",
)

# 测试辅助函数：用模拟结果替换三个真实函数。
def run_scenario(
    confidence: float,
) -> tuple[dict, dict]:
    """模拟一次指定产品置信度的完整客服业务链。"""
    fake_vision_analysis = {
        "product_hint": PRODUCT,
        "product_confidence": confidence,
        "visual_keywords": [
            "智能手表",
            "硅胶表带",
        ],
        "visual_description": (
            "图片中是一块带有硅胶表带的智能手表。"
        ),
        "latency_ms": 5,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    with (
        patch(
            "app.services.chat_service."
            "analyze_images_for_retrieval",
            return_value=fake_vision_analysis,
        ),
        patch(
            "app.services.chat_service.hybrid_search",
            return_value=[FAKE_RECORD],
        ) as search_mock,
        patch(
            "app.services.chat_service.generate_answer",
            return_value=FAKE_LLM_RESULT,
        ),
    ):
        result = answer_question(
            question=QUESTION,
            images=["data:image/jpeg;base64,TEST"],
        )

    search_arguments = search_mock.call_args.kwargs
    return result, search_arguments


def main() -> None:
    hard_result, hard_search = run_scenario(0.95)
    soft_result, soft_search = run_scenario(0.60)
    ignored_result, ignored_search = run_scenario(0.20)

    hard_metrics = hard_result["metrics"]
    soft_metrics = soft_result["metrics"]
    ignored_metrics = ignored_result["metrics"]

    checks = {
        "高置信度记录硬过滤策略": (
            hard_metrics["vision_product_strategy"]
            == "hard_filter"
        ),
        "高置信度把产品写入检索问题": (
            PRODUCT_HINT_LINE
            in hard_search["query"]
        ),
        "高置信度不使用软加权参数": (
            hard_search["preferred_product"] is None
        ),
        "中置信度记录软加权策略": (
            soft_metrics["vision_product_strategy"]
            == "soft_boost"
        ),
        "中置信度不触发产品硬过滤": (
            PRODUCT_HINT_LINE
            not in soft_search["query"]
        ),
        "中置信度传递偏好产品": (
            soft_search["preferred_product"]
            == PRODUCT
        ),
        "低置信度记录忽略策略": (
            ignored_metrics["vision_product_strategy"]
            == "ignored"
        ),
        "低置信度不触发产品硬过滤": (
            PRODUCT_HINT_LINE
            not in ignored_search["query"]
        ),
        "低置信度不使用产品软加权": (
            ignored_search["preferred_product"] is None
        ),
        "三个场景都保留原始问题": all(
            QUESTION in search_arguments["query"]
            for search_arguments in (
                hard_search,
                soft_search,
                ignored_search,
            )
        ),
        "三个场景都记录产品名称": all(
            result["metrics"]["vision_product_hint"]
            == PRODUCT
            for result in (
                hard_result,
                soft_result,
                ignored_result,
            )
        ),
        "三个场景都记录正确置信度": (
            hard_metrics["vision_product_confidence"]
            == 0.95
            and soft_metrics["vision_product_confidence"]
            == 0.60
            and ignored_metrics["vision_product_confidence"]
            == 0.20
        ),
        "三个场景都成功生成回答": all(
            result["answer"]
            == FAKE_LLM_RESULT.answer
            for result in (
                hard_result,
                soft_result,
                ignored_result,
            )
        ),
    }

    print("====产品置信度业务链测试====")

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    print("\n====实际传给检索模块的参数====")
    print("高置信度：", hard_search)
    print("中置信度：", soft_search)
    print("低置信度：", ignored_search)

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n产品置信度业务链测试：全部通过")


if __name__ == "__main__":
    main()

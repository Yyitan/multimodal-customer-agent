# Python 内置 mock 工具。作用：拦截替换项目里任意一个函数。可以让被拦截的函数：返回你指定的假数据（return_value）；直接抛出异常（side_effect=异常）
from unittest.mock import patch
from app.llm_client import LLMClientError,LLMResult
from app.services.chat_service import answer_question

# 视觉预分析接口：验证能不能触发降级逻辑
def main()->None:
    question="这个设备应该怎么操作？"
    
    # 伪造一条检索出来的知识库片段
    fake_record={
        "chunk_id":"test_chunk_001",
        "product":"测试产品",
        "text":"操作前请断开电源，然后按照说明完成操作。",
        "image_ids":[],
        "source":"test_manual.jsonl",
        "score":1.0,
        "retrieval_type":"semantic",
    }
    # 伪造大模型最终输出结果
    fake_result=LLMResult(
        answer="已根据文字检索结果生成回答。",
        image_ids=[],
        provider="openai_compatible",
        model="fake-vision-model",
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18,
        latency_ms=5,
        finish_reason="stop",
    )
    
    # with上下文管理器：只在这个with内部，这三个函数才会被替换；离开 with，恢复原来真实函数。
    with (
        # side_effect：当这个函数被调用时，直接抛出这个异常
        patch(
            "app.services.chat_service."
            "analyze_images_for_retrieval",
            side_effect=LLMClientError(
                "模拟视觉预分析失败"
            ),
        ),
        patch(
            "app.services.chat_service.hybrid_search",
            # 调用混合检索时，直接返回我们写好的假知识库记录，不走真实向量检索。
            return_value=[fake_record],
        ) as search_mock,
        # 劫持大模型生成函数，直接返回 fake_result，不请求真实 API。
        patch(
            "app.services.chat_service.generate_answer",
            return_value=fake_result,
        ),
    ):
        result=answer_question(
            question=question,
            images=["data:image/jpeg;base64,TEST"],
        )
    
    # search_mock.call_args:记录 mock 函数上一次是怎么被调用的;.kwargs：关键字参数,拿到传给 hybrid_search 的 query 参数
    retrieval_query=search_mock.call_args.kwargs["query"]
    metrics=result["metrics"]
    
    checks={
        "降级后使用原始问题检索":(
            retrieval_query==question
        ),
        "记录了视觉预分析调用":(
            metrics["vision_analysis_called"] is True
        ),
        "视觉预分析标记为失败":(
            metrics["vision_analysis_succeeded"] is False
        ),
        "触发了文字检索降级": (
            metrics["vision_fallback_used"] is True
        ),
        "记录了失败原因": (
            metrics["vision_analysis_error"]
            == "模拟视觉预分析失败"
        ),
        "降级后仍然生成回答":(
            result["answer"]=="已根据文字检索结果生成回答。"
        ),
    }
    
    for name,passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")
        
    if not all(checks.values()):
        raise SystemExit(1)
    print("\n视觉预分析降级测试：全部通过")
    
if __name__=="__main__":
    main()
    
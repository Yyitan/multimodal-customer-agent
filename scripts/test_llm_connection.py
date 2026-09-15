from app.config import get_settings
from app.llm_client import LLMClientError, generate_answer


def main() -> None:
    settings = get_settings()

    print("当前模型配置：")
    print("provider:", settings.llm_provider)
    print("model:", settings.llm_model)
    print("base_url:", settings.llm_base_url)
    print(
        "api_key:",
        "已配置" if settings.llm_api_key else "未配置",
    )

    prompt_package = {
        "system_prompt": (
            "你是一个产品客服助手。"
            "只返回合法JSON对象，不要使用Markdown代码块。"
        ),
        "user_prompt": """
        请简洁回答：电钻使用前需要注意什么？
        
        只返回以下JSON格式：
        {
            "answer":"内容回答",
            "image_ids":[]
        }
        """,
        
        "image_ids": [],
        "max_answer_images":3,
        "record_count": 0,
        "context_length": 0,
    }

    try:
        result = generate_answer(
            prompt_package=prompt_package,
            settings=settings,
        )
    except LLMClientError as error:
        print("\n模型调用失败：")
        print(error)
        raise SystemExit(1) from error

    print("\n模型调用成功：")
    print(result.answer)
    
    print("\n调用指标：")
    print("provider:", result.provider)
    print("model:", result.model)
    print("prompt_tokens:", result.prompt_tokens)
    print("completion_tokens:", result.completion_tokens)
    print("total_tokens:", result.total_tokens)
    print("latency_ms:", result.latency_ms)
    print("image_ids:", result.image_ids)
    print("finish_reason:", result.finish_reason)

if __name__ == "__main__":
    main()
import json
from typing import Any

from app.config import Settings
from app.llm_client import LLMClientError, generate_answer
from app.retriever import KNOWLEDGE_RECORDS

KNOWN_PRODUCTS=sorted(
    {
        record["product"]
        for record in KNOWLEDGE_RECORDS
        if record.get("product")
    }
)


VISION_ANALYSIS_SYSTEM_PROMPT = """你是一个用于知识库检索的图片分析器。

你的任务不是回答用户的最终问题，而是从用户上传的图片中提取检索线索。

请遵守以下规则：

1. 只能描述图片中能够直接观察到的内容。
2. 不要猜测品牌、具体型号、价格或保修信息。
3. 如果无法确定产品名称，将 product_hint 返回为空字符串。
4. product_hint 必须从用户提示中的知识库产品名称中选择。
5. 不要补充图片中看不到的功能。
6. 外层 JSON 的 answer 字段必须是字符串。
7. answer 字符串内部必须包含一个合法的 JSON 对象。
8. image_ids 必须返回空数组。
9. 不要使用 Markdown 代码块。
"""

VISION_ANALYSIS_USER_PROMPT = """请分析用户上传的图片，为后续知识库检索提取线索。

用户原始问题：
{question}

知识库已有产品名称：
{known_products}

请严格按照以下规则：

1. 如果能判断图片对应的产品，请从知识库已有产品名称中选择一个最匹配的产品。
2. 如果无法确定对应产品，product_hint 返回空字符串。
3. 不要自己创造产品名称。
4. 只能描述图片中能够直接观察到的内容。
5. 不要猜测品牌、具体型号、价格或保修信息。
6. visual_keywords 必须是有助于知识库检索的关键词。
7. 最终必须返回外层 JSON。
8. 外层 JSON 的 answer 必须是字符串。
9. answer 字符串里面再放一个 JSON 对象。
10. image_ids 必须是空数组。

外层 JSON 必须严格使用下面格式：

{{
  "answer": "{{\\\"product_hint\\\":\\\"健身追踪器\\\",\\\"visual_keywords\\\":[\\\"智能手表\\\",\\\"硅胶表带\\\"],\\\"visual_description\\\":\\\"图片中是一块带有硅胶表带的智能手表。\\\"}}",
  "image_ids": []
}}

其中，answer 字符串里的内容应该根据真实图片填写。
只返回合法 JSON，不要添加其他文字。
"""


def analyze_images_for_retrieval(
    question: str,
    images: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """使用视觉模型提取产品名称和检索关键词。"""

    if not images:
        return {
            "product_hint": "",
            "visual_keywords": [],
            "visual_description": "",
        }

    prompt_package = {
        "system_prompt": VISION_ANALYSIS_SYSTEM_PROMPT,
        # `.format()` 是字符串方法：把`{question}`占位替换成真实 question 文本
        "user_prompt": VISION_ANALYSIS_USER_PROMPT.format(question=question.strip(),known_products="、".join(KNOWN_PRODUCTS),),
        "user_images": images,
        "image_ids": [],
        "max_answer_images": 1,
        "record_count": 0,
        "context_length": 0,
    }
    
    result=generate_answer(
        prompt_package=prompt_package,
        settings=settings,
    )
    
    try:
        analysis=json.loads(result.answer)
    except json.JSONDecodeError as error:
        raise LLMClientError(
            "视觉预分析没有返回合法JOSN。"
        ) from error
    
    if not isinstance(analysis,dict):
        raise LLMClientError(
            "视觉预分析结果必须是JSON对象。"
        )
    # `dict.get(key, 默认值)`：字典安全取值方法
    # 如果字典里存在这个 key，就拿它的值；**如果 key 不存在，返回第二个参数的默认值**
    product_hint=analysis.get(
        "product_hint",
        "",
    )
    
    visual_keywords=analysis.get(
        "visual_keywords",
        [],
    )
    
    visual_description=analysis.get(
        "visual_description",
        "",
    )

    if not isinstance(product_hint,str):
        product_hint=""
    
    if not isinstance(visual_keywords,list):
        visual_keywords=[]
    
    visual_keywords=[
        # 遍历`visual_keywords`里面每一个 keyword，必须是字符串
        keyword.strip()
        for keyword in visual_keywords
        # 必须是字符串且去掉首尾空格不为空 放进新列表
        if isinstance(keyword,str)
        and keyword.strip()
    ]
    
    if not isinstance(visual_description,str):
        visual_description=""
    
    return {
        "product_hint":product_hint,
        "visual_keywords":visual_keywords,
        "visual_description":visual_description.strip(),
        "latency_ms":result.latency_ms,
        "prompt_tokens":result.prompt_tokens,
        "completion_tokens":result.completion_tokens,
        "total_tokens":result.total_tokens,
    }
    

def build_multimodal_retrieval_query(
    question:str,
    analysis:dict[str,Any],
)->str:
    """把视觉分析结果合并到原始问题中，形成增强检索问题。"""
    
    parts=[
        f"用户问题：{question.strip()}",
    ]
    
    product_hint=analysis.get(
        "product_hint",
        "",
    )
    
    if product_hint:
        parts.append(
            f"图片识别到的产品线索：{product_hint}"
        )
    
    visual_keywords=analysis.get(
        "visual_keywords",
        [],
    )
    
    if visual_keywords:
        parts.append(
            "图片关键词："
            +"、".join(visual_keywords)
        )
    visual_description=analysis.get(
        "visual_description",
        "",
    )
    
    if visual_description:
        parts.append(
            f"图片可见内容：{visual_description}"
        )
        
    return "\n".join(parts)


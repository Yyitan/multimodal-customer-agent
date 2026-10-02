from typing import Any
from time import perf_counter

from app.config import Settings, get_settings
from app.context_builder import build_context_package
from app.hybrid_retriever import hybrid_search
from app.llm_client import LLMClientError, LLMResult, generate_answer
from app.prompt_builder import build_prompt_package
from app.query_rewriter import rewrite_query_with_history
from app.vision_analyzer import (
    analyze_images_for_retrieval,
    build_multimodal_retrieval_query,
    choose_product_retrieval_strategy,
)
from app.conversation_store import (
    ConversationTurn,
    format_conversation_history,
)
import logging

logger = logging.getLogger(__name__)


# 指标打包：一次性组装成固定结构的metric字典。
def build_metrics(
    settings: Settings,
    retrieval_latency_ms: int,
    total_latency_ms: int,
    retrieved_record_count: int,
    context_record_count: int,
    context_length: int,
    llm_result: LLMResult | None,
    vision_analysis: dict[str, Any] | None = None,
    input_image_count: int = 0,
) -> dict[str, Any]:
    """构造一次问答流水线的运行指标。"""
    # 没有图片时，不进行视觉预分析，耗时0.Token为None;产品提示、置信度都为空
    if vision_analysis is None:
        vision_analysis_latency_ms = 0
        vision_analysis_prompt_tokens = None
        vision_analysis_completion_tokens = None
        vision_analysis_total_tokens = None
        vision_analysis_succeeded = False
        vision_fallback_used = False
        vision_analysis_error = None
        vision_product_hint=None
        vision_product_confidence = None
        vision_product_strategy = "none"
    # 有图片时，从vision_analysis字典中读取第一次模型调用的数据
    else:
        vision_analysis_latency_ms = vision_analysis.get(
            "latency_ms",
            0,
        )
        vision_analysis_prompt_tokens = vision_analysis.get("prompt_tokens")
        vision_analysis_completion_tokens = vision_analysis.get("completion_tokens")
        vision_analysis_total_tokens = vision_analysis.get("total_tokens")
        # 图片预分析成功否
        vision_analysis_succeeded = bool(vision_analysis.get("succeeded", True))
        # 是否退回纯文字检索
        vision_fallback_used = not vision_analysis_succeeded
        # 失败原因
        vision_analysis_error = vision_analysis.get("error")
        vision_product_hint = (
            vision_analysis.get("product_hint")
            or None
        )
        vision_product_confidence = (
            vision_analysis.get(
            "product_confidence",
                0.0,
            )
        )
        vision_product_strategy = (
            vision_analysis.get(
            "retrieval_strategy",
            "none",
            )
        )

    return {
        "llm_called": llm_result is not None,
        "provider": (
            llm_result.provider if llm_result is not None else settings.llm_provider
        ),
        "model": (llm_result.model if llm_result is not None else settings.llm_model),
        "prompt_tokens": (llm_result.prompt_tokens if llm_result is not None else None),
        "completion_tokens": (
            llm_result.completion_tokens if llm_result is not None else None
        ),
        "total_tokens": (llm_result.total_tokens if llm_result is not None else None),
        "llm_latency_ms": (llm_result.latency_ms if llm_result is not None else 0),
        "retrieval_latency_ms": retrieval_latency_ms,
        "total_latency_ms": total_latency_ms,
        "retrieved_record_count": retrieved_record_count,
        "context_record_count": context_record_count,
        "context_length": context_length,
        "finish_reason": (llm_result.finish_reason if llm_result is not None else None),
        "llm_attempt_count": (
            llm_result.attempt_count if llm_result is not None else 0
        ),
        "llm_retry_count": (
            max(llm_result.attempt_count - 1, 0) if llm_result is not None else 0
        ),
        "llm_circuit_state": (
            llm_result.circuit_state if llm_result is not None else "not_called"
        ),
        "selected_image_count": (
            len(llm_result.image_ids) if llm_result is not None else 0
        ),
        "input_image_count": input_image_count,
        "vision_used": (input_image_count > 0 and llm_result is not None),
        "vision_analysis_called": vision_analysis is not None,
        "vision_analysis_latency_ms": vision_analysis_latency_ms,
        "vision_analysis_prompt_tokens": vision_analysis_prompt_tokens,
        "vision_analysis_completion_tokens": vision_analysis_completion_tokens,
        "vision_analysis_total_tokens": vision_analysis_total_tokens,
        "vision_analysis_succeeded": vision_analysis_succeeded,
        "vision_fallback_used": vision_fallback_used,
        "vision_analysis_error": vision_analysis_error,
        "vision_product_hint": vision_product_hint,
        "vision_product_confidence": vision_product_confidence,
        "vision_product_strategy": vision_product_strategy,
    }

def build_history_aware_retrieval_query(
    question:str,
    history:list[ConversationTurn],
)->str:
    # 构造感知历史的检索 query
    history_text=format_conversation_history(history)
    # 如果没有历史 → 直接返回原始 question，不做改动
    if not history_text:
        return question
    # 如果存在历史 → 拼接：历史 + 当前问题，交给向量检索
    return (
        f"最近对话历史：\n{history_text}\n"
        f"当前问题：{question}"
    )

# 整个项目对外的主入口函数
# `question`：用户输入的原始问题字符串；`settings`：可选，可以手动传入自定义配置对象；不传就自动调用`get_settings()`读取全局配置
# 返回：字典，包含回答文本、用到的参考切片、图片 ID。
def answer_question(
    question: str,
    images: list[str] | None = None,
    settings: Settings | None = None,
    conversation_history:list[ConversationTurn]|None=None
) -> dict[str, Any]:
    """Run the complete customer-service pipeline."""
    # - 如果调用方手动传入 settings 对象，就用传入的；
    # - 如果不传，就拿全局缓存的配置实例`get_settings()`。
    current_settings = settings if settings is not None else get_settings()

    # 如果调用方没有传图片，统一转成空列表。
    current_images = images or []

    # 整条业务链从这里开始计时。
    # 视觉预分析、检索、上下文构造、大模型回答都会计入总耗时。
    pipeline_started_at = perf_counter()

    current_history=conversation_history or []

    # 检索查询构造：开启改写且有历史时，让大模型把追问还原成
    # 可独立检索的完整问题；失败或未开启时回退历史原文拼接。
    # 改写器自身在 mock provider 下抛错，走统一的失败回退路径。
    rewrite_meta: dict[str, Any] | None = None
    if current_settings.query_rewrite_enabled and current_history:
        try:
            rewrite_meta = rewrite_query_with_history(
                question,
                current_history,
                current_settings,
            )
            retrieval_query = rewrite_meta["query"]
        except Exception as error:
            logger.warning(
                "query_rewrite_failed fallback_to_concat error=%s",
                error,
            )
            rewrite_meta = None

    if rewrite_meta is None:
        retrieval_query = build_history_aware_retrieval_query(
            question,
            current_history,
        )
    # 保存视觉预分析结果，便于后面记录指标或调试。
    vision_analysis = None

    preferred_product = None
    # 如果用户上传了图片，先调用视觉模型提取检索线索
    if current_images:
        vision_analysis_started_at = perf_counter()
        try:
            # 尝试视觉预分析
            vision_analysis = analyze_images_for_retrieval(
                question=question,
                images=current_images,
                settings=current_settings,
            )

        # 只捕获模型调用类错误
        # 失败后不构成增强问题，保持原问题
        except LLMClientError as error:
            vision_analysis = {
                "product_hint": "",
                "product_confidence": 0.0,
                "retrieval_strategy": "fallback",  # 新增策略标记
                "visual_keywords": [],
                "visual_description": "",
                "latency_ms": round(
                    (perf_counter() - vision_analysis_started_at) * 1000
                ),
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "succeeded": False,
                "error": str(error),
            }

            logger.warning(
                "vision_analysis_failed " "fallback_to_text_retrieval error=%s",
                error,
            )
        # 没有异常时才生成增强检索问题
        else:
            vision_analysis["succeeded"] = True
            vision_analysis["error"] = None
            # 用视觉分析结果，选择产品检索策略
            product_strategy = choose_product_retrieval_strategy(vision_analysis)
            # 把本次策略存入vision_analysis字典，方便日志/评测查看
            vision_analysis["retrieval_strategy"] = product_strategy["strategy"]
            # 先用图片分析结果，生成multimodal_query（带产品提示的检索文本）
            multimodal_query=build_multimodal_retrieval_query(
                question=question,
                analysis=vision_analysis,
                include_product_hint=(
                    product_strategy["include_product_hint"]
                ),
            )
            # 再构造最终用于向量检索的retrieval_query：
            # 开启改写时对多模态查询同样先做指代消解，否则回退拼接
            if current_settings.query_rewrite_enabled and current_history:
                try:
                    rewrite_meta = rewrite_query_with_history(
                        multimodal_query,
                        current_history,
                        current_settings,
                    )
                    retrieval_query = rewrite_meta["query"]
                except Exception as error:
                    logger.warning(
                        "query_rewrite_failed_multimodal fallback_to_concat error=%s",
                        error,
                    )
                    rewrite_meta = None
                    retrieval_query = build_history_aware_retrieval_query(
                        multimodal_query,
                        current_history,
                    )
            else:
                retrieval_query = build_history_aware_retrieval_query(
                    multimodal_query,
                    current_history,
                )

            # 取出偏好产品，传给hybrid_search做软加分（soft_boost场景生效）
            preferred_product = product_strategy["preferred_product"]

    # 检索模块单独的计时起点专门用来只测 hybrid_search这一步花多久。
    retrieval_started_at = perf_counter()

    # 第一步：执行混合检索。
    # 所有检索参数不再写死硬编码，全部从 settings 配置读取。
    retrieved_records = hybrid_search(
        query=retrieval_query,
        semantic_top_k=current_settings.semantic_top_k,
        seed_top_k=current_settings.seed_top_k,
        window_size=current_settings.neighbor_window_size,
        max_results=current_settings.max_retrieval_results,
        preferred_product=preferred_product,
    )
    # 输出：排序完成的知识库切片列表`retrieved_records`。

    # 关键词检索 + 语义检索 + 排序 + 邻居扩展 :一共消耗了多少毫秒。
    retrieval_latency_ms = round(
        # `perf_counter()` 是检索跑完**当前的时间**
        (perf_counter() - retrieval_started_at)
        * 1000
    )

    # 分支判断：**检索结果为空**，没有召回任何切片。
    if not retrieved_records:

        total_latency_ms = round((perf_counter() - pipeline_started_at) * 1000)
        # 直接提前返回固定回答，不走后面上下文、prompt 逻辑。
        return {
            "answer": (
                "知识库中没有找到与该问题相关的资料。"
                "请补充产品名称、型号或具体故障现象。"
            ),
            "references": [],
            "image_ids": [],
            "metrics": build_metrics(
                settings=current_settings,
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=total_latency_ms,
                retrieved_record_count=0,
                context_record_count=0,
                context_length=0,
                # 这里 llm_result=None 表示根本没有调用模型。
                llm_result=None,
                vision_analysis=vision_analysis,
                input_image_count=len(current_images),
            ),
        }

    # 第二步：上下文组装。
    # 把检索出来的切片列表送入`build_context_package`；
    context_package = build_context_package(
        records=retrieved_records,
        max_context_chars=current_settings.max_context_chars,
        max_text_chars=current_settings.max_text_chars,
        query=retrieval_query,
    )

    # 分支判断：检索有结果，但是全部被上下文字符上限过滤丢弃。
    if not context_package["records"]:

        total_latency_ms = round((perf_counter() - pipeline_started_at) * 1000)
        # 最终`selected_records`为空
        # 此时直接返回提示，不再构建 prompt、调用大模型。
        return {
            "answer": ("找到了一些相关资料，但资料整理后" "没有足够内容用于生成答案。"),
            "references": [],
            "image_ids": [],
            "metrics": build_metrics(
                settings=current_settings,
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=total_latency_ms,
                retrieved_record_count=len(retrieved_records),
                context_record_count=0,
                context_length=0,
                llm_result=None,
                vision_analysis=vision_analysis,
                input_image_count=len(current_images),
            ),
        }

    # 第三步：构建完整 prompt 数据包。
    # 传入用户问题和上下文包，输出包含 system_prompt、user_prompt、image_ids、统计元数据。内部会做空校验，问题为空 / 上下文为空会抛异常。
    prompt_package = build_prompt_package(
        question=question,
        context_package=context_package,
        user_images=current_images,
        max_answer_images=current_settings.max_answer_images,
        conversation_history=current_history,
    )

    # 第四步：根据配置选择模型后端。
    llm_result = generate_answer(
        prompt_package=prompt_package,
        settings=current_settings,
    )

    # 计算整条流水线总耗时
    total_latency_ms = round((perf_counter() - pipeline_started_at) * 1000)

    result_metrics = build_metrics(
        settings=current_settings,
        retrieval_latency_ms=retrieval_latency_ms,
        total_latency_ms=total_latency_ms,
        retrieved_record_count=len(retrieved_records),
        context_record_count=context_package["record_count"],
        context_length=context_package["context_length"],
        llm_result=llm_result,
        vision_analysis=vision_analysis,
        input_image_count=len(current_images),
    )
    # 检索查询改写观测字段：是否启用改写、改写消耗
    result_metrics["query_rewrite_used"] = rewrite_meta is not None
    if rewrite_meta is not None:
        result_metrics["query_rewrite_tokens"] = rewrite_meta.get("total_tokens")
        result_metrics["query_rewrite_latency_ms"] = rewrite_meta.get(
            "latency_ms"
        )

    # 函数最终返回结构化字典。
    return {
        "answer": llm_result.answer,
        "references": context_package["records"],
        # 模型结合问题和资料选择的最终图片，最多3张
        "image_ids": llm_result.image_ids,
        "metrics": result_metrics,
    }


if __name__ == "__main__":
    result = answer_question("电钻电池温度太高时为什么不能马上充电？")

    print("回答：")
    print(result["answer"])

    print("\n使用的参考资料数：")
    print(len(result["references"]))

    print("\n相关图片：")
    print(result["image_ids"])

    print("\n运行指标：")
    for name, value in result["metrics"].items():
        print(f"{name}: {value}")

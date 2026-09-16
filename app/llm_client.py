# lru_cache：缓存装饰器，缓存OpenAI 客户端实例，避免反复新建连接
from functools import lru_cache
from typing import Any
import json
import re

# OpenAI SDK 内置异常类，用来捕获不同类型的接口报错
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from app.config import Settings
from app.prompt_builder import mock_generate_answer

# 数据类装饰器，用来专门存放数据的类
from dataclasses import dataclass

# 高精度计时器，专门用来测代码执行耗时
from time import perf_counter


class LLMClientError(RuntimeError):
    """调用大模型失败时抛出的统一业务异常。"""

    # **自定义异常类**：继承 Python 内置`RuntimeError`。作用：统一包装所有大模型调用相关错误。


# 自动生成`__init__`、`__repr__`，实例化的时候直接传参数，不用手写构造函数。
# `frozen=True`：**冻结只读**。一旦创建完`LLMResult`对象，所有字段**不能修改**，防止代码里不小心篡改返回结果，保证结果数据安全。
@dataclass(frozen=True)
class LLMResult:
    """一次大模型调用的结构化结果"""

    answer: str
    # 保存模型最终选择的图片
    image_ids: list[str]
    provider: str
    model: str
    # 输入prompt消耗token数量
    prompt_tokens: int | None
    # 模型生成回答消耗的token数量
    completion_tokens: int | None
    # 本次请求总共消耗token = prompt + completion
    total_tokens: int | None
    # 耗时，单位：毫秒 ms
    latency_ms: int
    # 记录模型为什么停止生成。（stop：正常完成；length：达到max_tokens，被强制停止）
    finish_reason: str | None


# `@lru_cache(maxsize=4)`：缓存客户端实例（最多只缓存4个客户端实例）。参数：api 密钥、接口地址、超时时间。
@lru_cache(maxsize=4)
# `_create_client`：私有函数（下划线开头，约定只能在本文件内部调用）
def _create_client(
    api_key: str,
    base_url: str | None,
    timeout_seconds: float,
) -> OpenAI:
    """创建并缓存OpenAI SDK客户端。"""
    client_options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        # `ax_retries=2`：SDK 内部自动重试 2 次，网络抖动自动重试
        "max_retries": 2,
    }
    # 如果`base_url`不为 None（对接第三方兼容接口，如本地 ollama、阿里云），加入配置；原生 OpenAI 不需要 base_url。
    if base_url:
        client_options["base_url"] = base_url
    # `OpenAI(**client_options)`字典解包，创建 SDK 客户端对象
    return OpenAI(**client_options)


# 私有函数：配置校验+获取客户端
def _get_client(
    settings: Settings,
    use_vision:bool=False,
    ) -> OpenAI:
    """检查模型配置并取得客户端。"""

    # 判断哪一套配置
    if use_vision:
        api_key_setting=settings.llm_vision_api_key
        base_url=settings.llm_vision_base_url
    else:
        api_key_setting=settings.llm_api_key
        base_url=settings.llm_base_url

    # 判断密钥有没有配置
    if api_key_setting is None:
        if use_vision:
            raise LLMClientError(
                "未配置视觉模型API Key,请检查 .env文件."
            )
        raise LLMClientError(
            "未配置大模型API Key,请检查 .env文件"
        )

    # 取出真实密钥字符串,去除首尾空格
    api_key=api_key_setting.get_secret_value().strip()
    # 判断key是不是空字符
    if not api_key:
        if use_vision:
            raise LLMClientError(
                "视觉模型 API Key 不能为空."
            )

        raise LLMClientError(
            "大模型API Key不能为空。"
        )

    # 创建OpenAI客户端
    return _create_client(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=settings.llm_timeout_seconds,
    )

def sanitize_answer(
    answer:str,
    candidate_image_ids:list[str],
)->str:
    """防止内部知识库图片ID出现在用户可见回答中。"""
    # 所以就是把大模型暴露出的内部图片ID，同义替换为中文“相关示意图”
    clean_answer=answer
    # 先处理较长的ID，避免一个短ID恰好是长ID的一部分。(image_01,image010:如果先替换短的 那么长的就变成相关示意图0)
    # 按字符串长度从长到短排序
    unqiue_image_ids=sorted(
        # `set(candidate_image_ids)`：去重。候选列表可能重复出现同一个图片 ID，转集合删掉重复项。
        set(candidate_image_ids),
        key=len,
        reverse=True,
    )

    # 只要文本里面出现这个图片 ID，就替换成中文文字`相关示意图`
    for image_id in unqiue_image_ids:
        clean_answer=clean_answer.replace(
            image_id,
            "相关示意图",
        )

    reference_sequence_pattern = (
        r"(?:参考)?资料\s*\d+"
        r"(?:\s*[、,，]\s*"
        r"(?:(?:参考)?资料\s*)?\d+)*"
    )

    parenthesized_reference_pattern = (
        rf"[（(]\s*"
        rf"{reference_sequence_pattern}"
        rf"\s*[）)]"
    )

    # 删除只用于标注内部资料编号的括号。
    clean_answer = re.sub(
        parenthesized_reference_pattern,
        "",
        clean_answer,
    )

    # 没有括号的内部编号替换为用户可以理解的普通称呼。
    clean_answer = re.sub(
        reference_sequence_pattern,
        "知识库资料",
        clean_answer,
    )

    return clean_answer.strip()

# raw_output:大模型原始返回的字符串;candidate_image_ids:候选图片 ID 列表;
def parse_structured_output(
    raw_output: str,
    candidate_image_ids: list[str],
    max_images: int,
) -> tuple[str, list[str]]:
    """解析并校验模型返回的JSON结果。"""

    try:
        payload = json.loads(raw_output)
    # 如果模型输出不是合法 JSON（比如多了文字、markdown ```json 标记、JSON 少逗号少括号），就会触发 `JSONDecodeError`
    except json.JSONDecodeError as error:
        # 捕获这个异常，包装成我们自定义的业务异常
        raise LLMClientError("大模型没有返回合法的JSON结果。") from error

    # 解析出来的 JSON 必须是对象（dict）
    if not isinstance(payload, dict):
        raise LLMClientError("大模型返回结果必须是JSON对象。")

    # 读取 json 里的 answer 字段
    answer = payload.get("answer")

    # answer 必须是字符串,并且去掉空格之后不能为空
    if not isinstance(answer, str) or not answer.strip():
        raise LLMClientError("大模型返回结果中缺少有效answer字段。")

    # 读取模型选出来的图片数组
    # 如果 json 里面没有 image_ids 这个 key，默认给空列表`[]`
    proposed_image_ids = payload.get(
        "image_ids",
        [],
    )
    # 如果模型返回的 image_ids 不是数组,强制改成空列表
    if not isinstance(proposed_image_ids, list):
        proposed_image_ids = []

    # 把候选图片列表转为集合
    allowed_image_ids = set(candidate_image_ids)
    # 用来存放清洗后、真正可用的图片 ID
    selected_image_ids: list[str] = []

    # 循环遍历模型提出的每一张图片 ID，层层过滤
    for image_id in proposed_image_ids:
        # 如果图片 id 不是字符串（比如数字、null）直接跳过
        if not isinstance(image_id, str):
            continue

        # 去掉前后空格，防止模型输出 " img_001 " 带空格导致匹配失败
        clean_image_id = image_id.strip()

        # 核心防幻觉。模型编造的、不在候选列表的图片 ID 直接丢弃
        if clean_image_id not in allowed_image_ids:
            continue

        # 去重。防止模型写两遍同一个图片 id
        if clean_image_id in selected_image_ids:
            continue

        # 通过全部校验 → 加入最终选中列表
        selected_image_ids.append(clean_image_id)

        # 达到图片上限，立刻停止循环
        if len(selected_image_ids) >= max_images:
            break

    clean_answer=sanitize_answer(
        answer=answer,
        candidate_image_ids=candidate_image_ids,
    )

    return clean_answer, selected_image_ids


# 对外公开主函数：pipline文件里要调用的入口。
def generate_answer(
    prompt_package: dict[str, Any],
    settings: Settings,
) -> LLMResult:
    """根据配置选择模型后端并返回回答及调用指标。"""

    # 在进入函数的第一时间记录当前时间戳
    started_at = perf_counter()

    # 第一分支判断：mock：直接调用模拟函数，不走任何网络请求。本地调试模式。
    if settings.llm_provider == "mock":
        answer = mock_generate_answer(prompt_package)

        # （结束时间戳 - 开始时间戳）`* 1000`：把秒转成毫秒
        latency_ms = round((perf_counter() - started_at) * 1000)
        return LLMResult(
            answer=answer,
            # 直接取候选图片前几张，模拟模型候选。
            image_ids=prompt_package["image_ids"][
                : prompt_package.get("max_answer_images", 3)
            ],
            provider=settings.llm_provider,
            model=settings.llm_model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
            finish_reason="mock",
        )

    # 从Prompt数据包中读取用户上传的图片
    user_images=prompt_package.get(
        "user_images",
        [],
    )

    # 只要用户上传了图片,就使用视觉模型.(把列表转布尔值:空列表,False,没图片,用文本模型;列表有图片字符串,True,视觉模型)
    use_vision=bool(user_images)

    if use_vision:
        request_provider=settings.llm_vision_provider
        request_model=settings.llm_vision_model

        # 如果开启视觉模式，但是视觉模型名字为空，直接抛出异常
        if not request_model:
            raise LLMClientError(
                "请求包含图片,但没有配置视觉模型名称."
            )
    else:
        request_provider=settings.llm_provider
        request_model=settings.llm_model

    # 根据use_vision选择对应的API Key 和Base URL.
    client=_get_client(
        settings,
        use_vision=use_vision,
    )

    # 真实分支前，初始化 token 变量。
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # 提前定义变量，存模型返回的停止原因。
    finish_reason: str | None = None

    # 拿到客户端，进入try捕获异常块
    try:
        # OpenAI 最新`responses`接口
        if request_provider == "openai":
            response = client.responses.create(
                model=request_model,
                instructions=prompt_package["system_prompt"],
                input=prompt_package["user_prompt"],
                max_output_tokens=settings.llm_max_output_tokens,
                # 不在OpenAI平台存储对话记录（隐私控制）
                store=False,
            )
            # 直接拿到模型输出文本。
            raw_output = response.output_text
            finish_reason = response.status

            # 取出本次调用的 token 用量对象。
            usage = response.usage

            if usage is not None:
                prompt_tokens = usage.input_tokens
                completion_tokens = usage.output_tokens
                total_tokens = usage.total_tokens

        # `openai_compatible` 兼容 OpenAI 旧版 chat 接口（绝大多数国产大模型、ollama 都用这套）
        elif request_provider == "openai_compatible":
            user_images = prompt_package.get(
                "user_images",
                [],
            )
            # 有用户图片时，必须配置视觉模型。
            if user_images and not settings.llm_vision_model:
                raise LLMClientError("当前请求包括图片，但项目尚未配置视觉模型。")

            # 没有图片使用普通文本模型：
            # 有图片使用视觉模型。
            request_model = (
                settings.llm_vision_model if user_images else settings.llm_model
            )

            # 没有图片时，content是普通字符串。
            user_content: Any = prompt_package["user_prompt"]

            # 有图片时，content改成多模态内容数组。
            if user_images:
                user_content = [{"type": "text", "text": prompt_package["user_prompt"]}]

                for image in user_images:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                            },
                        }
                    )

            request_options: dict[str, Any] = {
                "model": request_model,
                # 消息结构：`messages`数组，区分 system 角色和 user 角色。
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_package["system_prompt"]
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "max_tokens": settings.llm_max_output_tokens,
                # 用于保证模型返回合法 JSON
                "response_format": {
                    "type": "json_object",
                },
            }

            if use_vision:
                # Qwen3-VL 使用 enable_thinking 参数
                request_options["extra_body"] = {
                    "enable_thinking": settings.llm_vision_thinking,
                }

            elif settings.llm_thinking is not None:
                # 智谱 GLM 使用 thinking.type 参数
                request_options["extra_body"] = {
                    "thinking": {
                        "type": settings.llm_thinking,
                    }
                }

            # `**request_options`字典解包，把上面所有参数一次性传给接口。
            response = client.chat.completions.create(**request_options)

            choice = response.choices[0]
            # 模型返回原始字符串
            raw_output = choice.message.content or ""
            # 获取停止原因
            finish_reason = choice.finish_reason

            usage = response.usage

            if usage is not None:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens

        else:
            raise LLMClientError(f"不支持的大模型类型：{request_provider}")
    # 异常捕获块：捕获 OpenAI SDK 抛出的各类原生异常，**翻译成中文友好提示，重新包装为 LLMClientError**。
    # `raise ... from error`：保留原始异常链路，调试时可以看到底层原始报错，同时对外抛出我们自定义业务异常。
    except APITimeoutError as error:
        raise LLMClientError("大模型请求超时，请稍后重试。") from error

    except RateLimitError as error:
        raise LLMClientError("大模型接口触发限流，请稍后重试。") from error

    except APIConnectionError as error:
        raise LLMClientError("无法连接大模型服务，请检查网络和接口地址。") from error

    except APIStatusError as error:
        raise LLMClientError(
            f"大模型服务返回错误，状态码：{error.status_code}"
        ) from error

    # 结果校验：拿到模型返回内容之后做检查：1、必须是字符串类型。2、去掉首尾空白不能是空，空就抛异常。
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise LLMClientError("大模型返回了空回答。")

    answer, selected_image_ids = parse_structured_output(
        raw_output=raw_output,
        candidate_image_ids=prompt_package["image_ids"],
        max_images=prompt_package.get(
            "max_answer_images",
            3,
        ),
    )

    latency_ms = round((perf_counter() - started_at) * 1000)

    return LLMResult(
        answer=answer.strip(),
        image_ids=selected_image_ids,
        provider=request_provider,
        model=request_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )

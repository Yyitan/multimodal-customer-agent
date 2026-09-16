"""FastAPI entry point for the knowledge-based customer-service agent."""

# 生成返回时间戳
from datetime import datetime, timezone

# 生成随机且几乎不会重复得ID，后面用来生成 session_id
from uuid import uuid4

# 用来创建Web接口服务
from fastapi import FastAPI, HTTPException

# 用来规定接口JSON的格式，并自动检查用户传的数据是否合格
# `field_validator`：Pydantic 的字段校验装饰器，
from pydantic import BaseModel, Field,field_validator

from app.config import get_settings
from app.services.chat_service import answer_question
from app.llm_client import LLMClientError

# 增加结构化日志（区分日志等级、写入文件、记录时间、记录异常堆栈，适合后端服务。）
import logging


# 请求数据说明书:告诉 FastAPI：用户调用 /chat 时，允许传哪些字段、字段类型是什么、哪些必填。
class ChatRequest(BaseModel):
    """Request body accepted by POST /chat."""

    # 字段名:必须字符串 至少一个字符串,不能空传字符串
    question: str = Field(..., min_length=1, description="User question")
    # images:字符串列表 ,最多3张,后面再要求列表中每个字符串是Base64图片
    # default_factory=list:每个请求都新建一个空列表.这样不同的用户图片不会意外共用.
    images: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Optional Base64 data URLs. "
            "Up to 3 images are allowed."
        )
    )
    
    # 绑定这个校验函数，专门校验`images`这个字段。
    @field_validator("images")
    # 固定写法，Pydantic v2 要求。
    @classmethod
    # 触发时机：前端 POST 请求到达 FastAPI，在代码进入`/chat`接口函数之前自动执行。只要校验抛`ValueError`，FastAPI 直接返回`422 Unprocessable Entity`，后面所有业务代码完全不会跑。
    def validate(cls, values:list[str])->list[str]:
        """检查用户上传图片是否是允许的Base64 Data URL。"""
        # Base64 DataURL 格式：图片二进制直接放在 JSON 请求体里面，不需要服务器保存图片文件，接口直接拿到图片内容传给多模态大模型。
        allowed_prefixes=(
            "data:image/jpeg;base64",
            "data:image/jpg;base64",
            "data:image/png;base64",
            "data:image/webp;base64",
        )
        
        # `values` 就是用户传进来的图片字符串列表，循环遍历每一张图片。
        for image in values:
            if not isinstance(image,str):
                raise ValueError("每张图片必须是字符串。")
            
            # 判断字符串是否以其中任意一个前缀开头
            if not image.startswith(allowed_prefixes):
                raise ValueError(
                    "图片必须是JPEG、PNG或WEBP的Base64 Data URL。"
                )
            # 防止单张图片过大，避免请求体无限膨胀。
            if len(image)>8_000_000:
                raise ValueError(
                    "单张图片过大，Base64字符串不能超过8MB。"
                )
                
        return values
    
    session_id: str | None = Field(default=None, description="Conversation ID")
    # 是否流式返回,接口先预留
    stream: bool = Field(default=False, description="Whether to stream the response")


# 规定每条参考资料的返回格式
class KnowledgeReference(BaseModel):
    """One knowledge chunk retrieved from the manuals"""

    chunk_id: str
    product: str
    text: str
    image_ids: list[str]
    source: str
    score: float
    retrieval_type: str


class ChatMetrics(BaseModel):
    """One request's retrieval and model metrics."""

    llm_called: bool
    provider: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    llm_latency_ms: int
    vision_analysis_called:bool
    vision_analysis_latency_ms:int
    vision_analysis_prompt_tokens:int|None
    vision_analysis_completion_tokens:int|None
    vision_analysis_total_tokens:int|None
    retrieval_latency_ms: int
    total_latency_ms: int
    retrieved_record_count: int
    context_record_count: int
    context_length: int
    finish_reason: str | None
    selected_image_count: int
    input_image_count:int
    vision_used:bool


# 规定真正业务结果的格式
class ChatData(BaseModel):
    """Business data nested inside the API response."""

    answer: str
    session_id: str
    timestamp: int
    references: list[KnowledgeReference] = Field(
        default_factory=list,
        description="Knowledge chunks retrieved from the manuals",
    )
    image_ids: list[str] = Field(
        default_factory=list,
        description="Image related to the answer",
    )
    metrics: ChatMetrics


# 外层:最终接口返回
# 因为真实项目还需要统一告诉调用方
class ChatResponse(BaseModel):
    """Unified response shape required by the competition specification."""

    # code:请求成功否,0表示成功
    code: int = 0
    # 简短状态
    msg: str = "success"
    # 真正的业务结果
    data: ChatData


settings = get_settings()

# 全局一次性配置日志系统。
# 日志等级：`DEBUG`(调试) → `INFO`(正常运行信息) → `WARNING`警告 → `ERROR`错误 → `CRITICAL`严重故障
logging.basicConfig(
    # `getattr(logging, "INFO")` 等价取 `logging.INFO` 这个常量
    level=getattr(
        logging,
        # `settings.log_level` 是配置文件里面写的日志等级字符串，例如 `"info"` / `"debug"` / `"warn"`。`.upper()` 全部转大写，变成 `"INFO"`，防止配置里小写导致报错
        settings.log_level.upper(),
        # 第三个参数 `logging.INFO`：兜底默认值。如果配置写了非法等级（写错单词），自动降级用 INFO，程序不会崩溃。
        # 设置 level=INFO 时：DEBUG 日志会被忽略，INFO/WARN/ERROR 才会输出。
        logging.INFO,
    ),
    # 日志输出模板：`%(asctime)s`：时间戳；`%(levelname)s`：日志级别 INFO / ERROR；`%(name)s`：logger 名字，`__name__`就是`app.main`；`%(message)s`：自己写的日志内容
    format=("%(asctime)s|%(levelname)s|" "%(name)s|%(message)s"),
)

# `__name__`：当前模块名 `app.main`，拿到这个模块专属的 logger 对象，后面用`logger.info()`、`logger.exception()`打印日志。
logger = logging.getLogger(__name__)
# 不同文件可以创建各自 logger，看日志时能区分是哪个文件打出的日志。

# 创建一个名为app的Web服务对象:FatAPI会自动读取这些信息,在http://127.0.0.1:8000/docs 生成可交互的接口文档
app = FastAPI(
    title="Multimodal Customer Service Agent",
    version="0.3.0",
    description=("Multimodal customer-service API " "with hybrid knowledge retrieval"),
)


# 当有人用 GET 请求访问 /health 时，执行下一行定义的 health_check 函数
# GET:通常用于捕获/查看信息
# /health :接口地址,也叫路径或路由
@app.get("/health")
def health_check() -> dict[str, str]:
    """A small endpoint used to check whether the service is alive."""

    return {"status": "ok"}


# 当外部程序向 /chat 发送一个 POST 请求时，交给下面的 chat 函数处理；并且返回结果必须符合 ChatResponse 的格式
@app.post("/chat", response_model=ChatResponse)
# 表示 FastAPI 会读取用户发来的 JSON，并根据之前写的 ChatRequest 规则自动转换、检查，然后传给 request
def chat(
    request: ChatRequest,
) -> ChatResponse:  # 表示这个函数最终返回的数据，应符合之前定义的 ChatResponse 结构
    """Generate a grounded answer using retrieval and an LLM."""
    # 如果用户传了 session_id，就使用用户传的；否则，自动生成一个新的 session_id。
    session_id = request.session_id or f"kf_session_{uuid4().hex[:12]}"

    try:
        service_result = answer_question(request.question,images=request.images,)

    except LLMClientError as error:
        # 记录失败请求
        # logger.exception()：只能写在 except 块里面，输出一条 ERROR 等级日志；自动把完整异常堆栈（traceback）一并打印出来，看到哪一行代码出错，和你命令行看到的报错堆栈一样。
        # 如果用logger.error()，不会自动带上堆栈，排查问题很难。
        logger.exception(
            # `chat_failed` 标记：这条日志代表一次问答失败，`%s` 是占位符，后面依次填入 `session_id, provider, model`。session_id：会话编号，用来追踪这一次失败请求，后续可以根据 session_id 定位整条链路。
            "chat_failed session_id=%s provider=%s model=%s",
            session_id,
            settings.llm_provider,
            settings.llm_model,
        )

        # 503：服务器暂时不可用
        raise HTTPException(status_code=503, detail=str(error)) from error

    # 把前面`build_metrics`生成的指标字典拿出来，方便后面读取。
    metrics=service_result["metrics"]
    # `logger.info()`：输出普通成功信息日志
    logger.info(
        (
            # 标记字符串`chat_completed`，一眼看出这是一次问答成功。
            "chat_completed session_id=%s"
            "provider=%s model=%s"
            "vision_analysis_ms=%s retrieval_ms=%s"
            "llm_ms=%s total_ms=%s"
            "prompt_tokens=%s completion_tokens=%s"
            "total_tokens=%s references=%s"
            "finish_reason=%s selected_images=%s"
        ),
        
        session_id,
        metrics["provider"],
        metrics["model"],
        metrics["vision_analysis_latency_ms"],
        metrics["retrieval_latency_ms"],
        metrics["llm_latency_ms"],
        metrics["total_latency_ms"],
        metrics["prompt_tokens"],
        metrics["completion_tokens"],
        metrics["total_tokens"],
        metrics["context_record_count"],
        metrics["finish_reason"],
        metrics["selected_image_count"],
    )     
    # 示例：2026-09-10 21:00:00,123 | INFO | app.main | chat_completed session_id=ses_001 provider=mock model=mock retrieval_ms=35 llm_ms=12 total_ms=160 prompt_tokens=210 completion_tokens=85 total_tokens=295 references=3 finish_reason=stop selected_images=2  
    
    references = [
        # record:record 本来是一个字典,会把字典中的键和值转换成参数:字典的键 → 参数名,字典的值 → 参数值
        KnowledgeReference(**record)
        for record in service_result["references"]
    ]

    return ChatResponse(
        data=ChatData(
            answer=service_result["answer"],
            session_id=session_id,
            timestamp=int(datetime.now(timezone.utc).timestamp()),
            references=references,
            image_ids=service_result["image_ids"],
            metrics=ChatMetrics(
                # `**` 字典解包，把字典里所有 `key:value`，拆成一个个独立的关键字参数传给类 / 函数。（`*列表`：列表解包 → 拆成位置参数）
                # **：自动读取字典每一项，key 变成参数名，value 变成参数值，一次性全部传入 `ChatMetrics` 的构造函数。
                **service_result["metrics"]
            ),
        )
    )

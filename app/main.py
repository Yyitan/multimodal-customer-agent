"""FastAPI entry point for the knowledge-based customer-service agent."""

from contextlib import asynccontextmanager
from secrets import compare_digest
from typing import Any, Literal

# 生成返回时间戳
from datetime import datetime, timezone

# 生成随机且几乎不会重复得ID，后面用来生成 session_id
from uuid import uuid4

# 用来创建Web接口服务
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Path as PathParameter,
    Query,
    Request,
    Response,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 用来规定接口JSON的格式，并自动检查用户传的数据是否合格
# `field_validator`：Pydantic 的字段校验装饰器，
from pydantic import BaseModel, Field, field_validator

from app.config import (
    PROJECT_ROOT,
    get_settings,
    validate_runtime_settings,
)
from app.agent_policy import (
    classify_request,
    review_evidence,
)
from app.input_security import (
    ImageValidationError,
    RequestBodyLimitMiddleware,
    SESSION_ID_PATTERN,
    validate_image_data_urls,
)
from app.services.chat_service import answer_question, build_metrics
from app.llm_client import LLMClientError
from app.conversation_store import ConversationStore
from app.semantic_retriever import (
    semantic_index_status,
    warm_up_semantic_retriever,
)
from app.request_control import (
    REQUEST_ID_PATTERN,
    RequestIdMiddleware,
    SlidingWindowRateLimiter,
    rate_limit_key,
)
from app.observability import MetricsMiddleware, MetricsRegistry
from app.knowledge_images import IMAGE_ID_PATTERN, KnowledgeImageCatalog
from app.support_store import SupportStore
from app.version import API_PREFIX, APP_NAME, APP_VERSION, CAPABILITIES

# 增加结构化日志（区分日志等级、写入文件、记录时间、记录异常堆栈，适合后端服务。）
import logging


# 请求数据说明书:告诉 FastAPI：用户调用 /chat 时，允许传哪些字段、字段类型是什么、哪些必填。
class ChatRequest(BaseModel):
    """Request body accepted by POST /chat."""

    # 字段名:必须字符串 至少一个字符串,不能空传字符串
    question: str = Field(..., min_length=1, description="User question")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """去除无意义空白并限制问题长度。"""
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("问题不能为空。")
        if len(clean_value) > get_settings().max_question_chars:
            raise ValueError("问题长度超过限制。")
        return clean_value
    # images:字符串列表 ,最多3张,后面再要求列表中每个字符串是Base64图片
    # default_factory=list:每个请求都新建一个空列表.这样不同的用户图片不会意外共用.
    images: list[str] = Field(
        default_factory=list,
        max_length=3,
        description=("Optional Base64 data URLs. " "Up to 3 images are allowed."),
    )

    # 绑定这个校验函数，专门校验`images`这个字段。
    @field_validator("images")
    # 固定写法，Pydantic v2 要求。
    @classmethod
    # 触发时机：前端 POST 请求到达 FastAPI，在代码进入`/chat`接口函数之前自动执行。只要校验抛`ValueError`，FastAPI 直接返回`422 Unprocessable Entity`，后面所有业务代码完全不会跑。
    def validate_images(cls, values: list[str]) -> list[str]:
        """校验图片编码、真实格式、大小和像素数量。"""
        current_settings = get_settings()
        try:
            return validate_image_data_urls(
                values,
                max_image_bytes=current_settings.max_image_bytes,
                max_total_bytes=current_settings.max_total_image_bytes,
                max_image_pixels=current_settings.max_image_pixels,
            )
        except ImageValidationError as error:
            raise ValueError(str(error)) from error

    session_id: str | None = Field(default=None, description="Conversation ID")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        """限制会话 ID，避免无界数据库键和不可见字符。"""
        if value is None:
            return None
        clean_value = value.strip()
        if not SESSION_ID_PATTERN.fullmatch(clean_value):
            raise ValueError(
                "session_id 必须为 1 到 128 位安全字符。"
            )
        return clean_value
    # 是否流式返回,接口先预留
    stream: bool = Field(default=False, description="Whether to stream the response")

    @field_validator("stream")
    @classmethod
    def reject_unsupported_streaming(cls, value: bool) -> bool:
        """当前接口不是流式接口，显式拒绝未实现的行为。"""
        if value:
            raise ValueError("当前版本尚不支持流式响应。")
        return value


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
    vision_analysis_called: bool
    vision_analysis_latency_ms: int
    vision_analysis_prompt_tokens: int | None
    vision_analysis_completion_tokens: int | None
    vision_analysis_total_tokens: int | None
    vision_analysis_succeeded: bool
    vision_fallback_used: bool
    vision_analysis_error: str | None
    retrieval_latency_ms: int
    total_latency_ms: int
    retrieved_record_count: int
    context_record_count: int
    context_length: int
    finish_reason: str | None
    selected_image_count: int
    input_image_count: int
    vision_used: bool
    vision_product_hint: str | None
    vision_product_confidence: float | None
    vision_product_strategy: str
    llm_attempt_count: int = 0
    llm_retry_count: int = 0
    llm_circuit_state: str = "not_called"


class KnowledgeImage(BaseModel):
    image_id: str
    url: str


class AgentMetadata(BaseModel):
    """Agent routing and escalation decision returned to the caller."""

    intent: str
    action: str
    confidence: float
    handoff_required: bool
    handoff_reason: str | None
    ticket_id: str | None = None


class FeedbackRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=SESSION_ID_PATTERN.pattern,
    )
    request_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=REQUEST_ID_PATTERN.pattern,
    )
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(default="", max_length=1000)


class FeedbackResponse(BaseModel):
    feedback_id: str
    session_id: str
    request_id: str
    rating: int
    comment: str
    created_at: int


class TicketStatusUpdate(BaseModel):
    status: Literal["open", "in_progress", "resolved", "closed"]


class SupportTicketResponse(BaseModel):
    ticket_id: str
    session_id: str
    request_id: str
    question: str
    reason: str
    priority: Literal["normal", "high", "urgent"]
    status: Literal["open", "in_progress", "resolved", "closed"]
    created_at: int
    updated_at: int


class OperationsSummaryResponse(BaseModel):
    ticket_counts: dict[str, int]
    active_ticket_count: int
    urgent_open_count: int
    feedback_count: int
    feedback_average_rating: float | None
    rating_counts: dict[str, int]


class ServiceInfoResponse(BaseModel):
    name: str
    version: str
    api_prefix: str
    capabilities: list[str]


# 规定真正业务结果的格式
class ChatData(BaseModel):
    """Business data nested inside the API response."""

    answer: str
    session_id: str
    request_id: str
    timestamp: int
    references: list[KnowledgeReference] = Field(
        default_factory=list,
        description="Knowledge chunks retrieved from the manuals",
    )
    image_ids: list[str] = Field(
        default_factory=list,
        description="Image related to the answer",
    )
    images: list[KnowledgeImage] = Field(default_factory=list)
    agent: AgentMetadata
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
rate_limiter = SlidingWindowRateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)
metrics_registry = MetricsRegistry()


def verify_api_token(
    authorization: str | None = Header(default=None),
) -> None:
    """根据配置校验请求是否携带正确的 Bearer Token。"""
    # 如果配置 `auth_enabled=false`（本地开发），鉴权直接关闭，不做任何校验，直接放行。
    if not settings.auth_enabled:
        return

    configured_token = (
        settings.api_token.get_secret_value()
        if settings.api_token is not None
        else ""
    )
    # 鉴权开关打开，但服务端没配置 Token，返回 503
    if not configured_token:
        raise HTTPException(
            status_code=503,
            detail="服务器未配置 API Token。"
        )

    expected_authorization = f"Bearer {configured_token}"
    # 请求没有携带正确的 Authorization，返回 401。
    if not authorization or not compare_digest(
        authorization,
        expected_authorization,
    ):
        raise HTTPException(
            status_code=401,
            detail="缺少或错误的API Token。",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )


def enforce_rate_limit(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """按 Token 或客户端 IP 限制昂贵问答接口的调用频率。"""
    if not getattr(settings, "rate_limit_enabled", False):
        return

    decision = rate_limiter.check(
        rate_limit_key(
            authorization=authorization,
            client_host=(
                request.client.host if request.client is not None else None
            ),
        )
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后重试。",
            headers={
                "Retry-After": str(decision.retry_after_seconds),
            },
        )


# 只有完全匹配 Bearer Token 才允许进入业务函数。
conversation_store = ConversationStore(
    max_turns=settings.conversation_max_turns,
    max_sessions=settings.conversation_max_sessions,
    db_path=settings.conversation_db_path,
)
support_store = SupportStore(db_path=settings.support_db_path)
knowledge_image_catalog = KnowledgeImageCatalog(
    settings.knowledge_image_dir
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时校验依赖并按配置预热，停止时释放 SQLite。"""
    try:
        configuration_errors = validate_runtime_settings(settings)
        if configuration_errors:
            raise RuntimeError("；".join(configuration_errors))
        if settings.preload_embedding_model:
            warm_up_semantic_retriever()
        yield
    finally:
        conversation_store.close()
        support_store.close()

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
    version=APP_VERSION,
    description=("Multimodal customer-service API " "with hybrid knowledge retrieval"),
    lifespan=lifespan,
)
web_dir = PROJECT_ROOT / "app" / "web"
app.mount(
    "/assets",
    StaticFiles(directory=web_dir / "assets"),
    name="assets",
)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=settings.max_request_bytes,
    paths={"/chat", "/v1/chat"},
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(MetricsMiddleware, registry=metrics_registry)


@app.get("/", include_in_schema=False, response_class=FileResponse)
def web_console() -> FileResponse:
    """Serve the customer-service operations console."""
    return FileResponse(web_dir / "index.html")


@app.get("/v1/info", tags=["system"], response_model=ServiceInfoResponse)
def service_info() -> ServiceInfoResponse:
    """Expose stable release metadata for clients and deployment checks."""
    return ServiceInfoResponse(
        name=APP_NAME,
        version=APP_VERSION,
        api_prefix=API_PREFIX,
        capabilities=list(CAPABILITIES),
    )


# 当有人用 GET 请求访问 /health 时，执行下一行定义的 health_check 函数
# GET:通常用于捕获/查看信息
# /health :接口地址,也叫路径或路由
@app.get("/health", include_in_schema=False)
@app.get("/v1/health", tags=["system"])
def health_check() -> dict[str, str]:
    """A small endpoint used to check whether the service is alive."""

    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
@app.get("/v1/ready", tags=["system"])
def readiness_check(response: Response) -> dict[str, Any]:
    """检查配置、会话数据库和知识索引是否可以提供服务。"""
    errors = validate_runtime_settings(settings)
    if not conversation_store.ping():
        errors.append("会话数据库不可用。")
    if not support_store.ping():
        errors.append("工单数据库不可用。")

    try:
        index_status = semantic_index_status()
    except (OSError, ValueError, KeyError) as error:
        index_status = None
        errors.append(f"知识索引不可用：{error}")

    if errors:
        response.status_code = 503
        return {
            "status": "not_ready",
            "errors": errors,
            "index": index_status,
        }

    return {
        "status": "ready",
        "database": "ok",
        "support_database": "ok",
        "index": index_status,
    }


@app.get(
    "/metrics",
    include_in_schema=False,
    dependencies=[Depends(verify_api_token)],
    response_class=Response,
)
@app.get(
    "/v1/metrics",
    tags=["operations"],
    dependencies=[Depends(verify_api_token)],
    response_class=Response,
)
def metrics() -> Response:
    """Expose Prometheus text metrics for operations monitoring."""
    return Response(
        content=metrics_registry.render(),
        media_type="text/plain; version=0.0.4",
    )

# 新增接口：删除会话（清空某一个用户对话记忆）
@app.delete(
    "/sessions/{session_id}",
    include_in_schema=False,
    dependencies=[Depends(verify_api_token)],
)
@app.delete(
    "/v1/sessions/{session_id}",
    tags=["conversations"],
    dependencies=[Depends(verify_api_token)],
)
def clear_session(
    session_id: str = PathParameter(
        ...,
        min_length=1,
        max_length=128,
        pattern=SESSION_ID_PATTERN.pattern,
    ),
) -> dict[str, str | bool]:
    cleared=conversation_store.clear(session_id)

    return {
        "session_id":session_id,
        "cleared":cleared,
    }


@app.post(
    "/feedback",
    include_in_schema=False,
    response_model=FeedbackResponse,
    status_code=201,
    dependencies=[Depends(verify_api_token)],
)
@app.post(
    "/v1/feedback",
    tags=["quality"],
    response_model=FeedbackResponse,
    status_code=201,
    dependencies=[Depends(verify_api_token)],
)
def submit_feedback(
    payload: FeedbackRequest,
    request: Request,
) -> FeedbackResponse:
    """Persist customer feedback for later quality analysis."""
    record = support_store.add_feedback(
        session_id=payload.session_id,
        request_id=(
            payload.request_id or request.state.request_id
        ),
        rating=payload.rating,
        comment=payload.comment,
    )
    metrics_registry.increment(
        "customer_feedback_total",
        rating=payload.rating,
    )
    logger.info(
        "feedback_received feedback_id=%s session_id=%s request_id=%s rating=%s",
        record["feedback_id"],
        record["session_id"],
        record["request_id"],
        record["rating"],
    )
    return FeedbackResponse(**record)


@app.get(
    "/tickets",
    include_in_schema=False,
    response_model=list[SupportTicketResponse],
    dependencies=[Depends(verify_api_token)],
)
@app.get(
    "/v1/tickets",
    tags=["support operations"],
    response_model=list[SupportTicketResponse],
    dependencies=[Depends(verify_api_token)],
)
def list_support_tickets(
    status: Literal[
        "open",
        "in_progress",
        "resolved",
        "closed",
    ] | None = None,
    limit: int = Query(default=50, ge=1, le=100),
) -> list[SupportTicketResponse]:
    """List recent tickets for the support operations view."""
    return [
        SupportTicketResponse(**record)
        for record in support_store.list_tickets(
            status=status,
            limit=limit,
        )
    ]


@app.get(
    "/v1/operations/summary",
    tags=["support operations"],
    response_model=OperationsSummaryResponse,
    dependencies=[Depends(verify_api_token)],
)
def get_operations_summary() -> OperationsSummaryResponse:
    """Return aggregate ticket and customer feedback indicators."""
    return OperationsSummaryResponse(**support_store.get_operations_summary())


@app.get(
    "/tickets/{ticket_id}",
    include_in_schema=False,
    response_model=SupportTicketResponse,
    dependencies=[Depends(verify_api_token)],
)
@app.get(
    "/v1/tickets/{ticket_id}",
    tags=["support operations"],
    response_model=SupportTicketResponse,
    dependencies=[Depends(verify_api_token)],
)
def get_support_ticket(
    ticket_id: str = PathParameter(
        ...,
        pattern=r"^ticket_[a-f0-9]{16}$",
    ),
) -> SupportTicketResponse:
    record = support_store.get_ticket(ticket_id)
    if record is None:
        raise HTTPException(status_code=404, detail="工单不存在。")
    return SupportTicketResponse(**record)


@app.patch(
    "/tickets/{ticket_id}",
    include_in_schema=False,
    response_model=SupportTicketResponse,
    dependencies=[Depends(verify_api_token)],
)
@app.patch(
    "/v1/tickets/{ticket_id}",
    tags=["support operations"],
    response_model=SupportTicketResponse,
    dependencies=[Depends(verify_api_token)],
)
def update_support_ticket(
    payload: TicketStatusUpdate,
    ticket_id: str = PathParameter(
        ...,
        pattern=r"^ticket_[a-f0-9]{16}$",
    ),
) -> SupportTicketResponse:
    record = support_store.update_ticket_status(
        ticket_id,
        payload.status,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="工单不存在。")
    metrics_registry.increment(
        "support_ticket_status_updates_total",
        status=payload.status,
    )
    logger.info(
        "ticket_status_changed ticket_id=%s status=%s",
        ticket_id,
        payload.status,
    )
    return SupportTicketResponse(**record)


@app.get(
    "/v1/knowledge-images/{image_id}",
    tags=["knowledge"],
    dependencies=[Depends(verify_api_token)],
    response_class=FileResponse,
)
def get_knowledge_image(
    image_id: str = PathParameter(
        ...,
        pattern=IMAGE_ID_PATTERN.pattern,
    ),
) -> FileResponse:
    image_path = knowledge_image_catalog.resolve(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="知识图片不存在。")
    return FileResponse(
        image_path,
        filename=image_path.name,
        headers={"Cache-Control": "private, max-age=3600"},
    )

# 当外部程序向 /chat 发送一个 POST 请求时，交给下面的 chat 函数处理；并且返回结果必须符合 ChatResponse 的格式
@app.post(
    "/chat",
    include_in_schema=False,
    response_model=ChatResponse,
    dependencies=[
        Depends(verify_api_token),
        Depends(enforce_rate_limit),
    ],
)
@app.post(
    "/v1/chat",
    tags=["agent"],
    response_model=ChatResponse,
    dependencies=[
        Depends(verify_api_token),
        Depends(enforce_rate_limit),
    ],
)
# 表示 FastAPI 会读取用户发来的 JSON，并根据之前写的 ChatRequest 规则自动转换、检查，然后传给 request
def chat(
    request: ChatRequest,
    http_request: Request,
) -> ChatResponse:  # 表示这个函数最终返回的数据，应符合之前定义的 ChatResponse 结构
    """Generate a grounded answer using retrieval and an LLM."""
    # 如果用户传了 session_id，就使用用户传的；否则，自动生成一个新的 session_id。
    session_id = request.session_id or f"kf_session_{uuid4().hex[:12]}"

    conversation_history = conversation_store.get_history(
        session_id
    )
    agent_decision = classify_request(request.question)

    if agent_decision.handoff_required:
        service_result = {
            "answer": agent_decision.response_message or "需要人工客服处理。",
            "references": [],
            "image_ids": [],
            "metrics": build_metrics(
                settings=settings,
                retrieval_latency_ms=0,
                total_latency_ms=0,
                retrieved_record_count=0,
                context_record_count=0,
                context_length=0,
                llm_result=None,
                input_image_count=len(request.images),
            ),
        }
    else:
        try:
            service_result = answer_question(
                request.question,
                images=request.images,
                conversation_history=conversation_history,
            )
        except LLMClientError as error:
            metrics_registry.increment(
                "llm_errors_total",
                provider=settings.llm_provider,
                category=error.category,
            )
            if error.category == "circuit_open":
                metrics_registry.increment(
                    "llm_circuit_rejections_total",
                    provider=settings.llm_provider,
                )
            logger.exception(
                "chat_failed request_id=%s session_id=%s provider=%s model=%s category=%s attempts=%s",
                http_request.state.request_id,
                session_id,
                settings.llm_provider,
                settings.llm_model,
                error.category,
                error.attempt_count,
            )
            headers = None
            if error.retry_after_seconds is not None:
                headers = {
                    "Retry-After": str(max(1, round(error.retry_after_seconds)))
                }
            raise HTTPException(
                status_code=503,
                detail=str(error),
                headers=headers,
            ) from error

        agent_decision = review_evidence(
            agent_decision,
            service_result,
        )
        if agent_decision.handoff_required:
            service_result["answer"] = (
                agent_decision.response_message
                or service_result["answer"]
            )

    ticket_id: str | None = None
    if agent_decision.handoff_required:
        priority = (
            "urgent"
            if agent_decision.intent == "safety_risk"
            else "high"
            if agent_decision.intent == "business_action"
            else "normal"
        )
        ticket = support_store.create_ticket(
            session_id=session_id,
            request_id=http_request.state.request_id,
            question=request.question,
            reason=agent_decision.handoff_reason or "human_review_required",
            priority=priority,
        )
        ticket_id = str(ticket["ticket_id"])
        metrics_registry.increment(
            "support_tickets_created_total",
            priority=priority,
            reason=agent_decision.handoff_reason or "human_review_required",
        )
        logger.info(
            "ticket_created ticket_id=%s session_id=%s request_id=%s priority=%s reason=%s",
            ticket_id,
            session_id,
            http_request.state.request_id,
            priority,
            agent_decision.handoff_reason,
        )
        service_result["answer"] = (
            f"{service_result['answer']} "
            f"已创建人工服务工单，编号：{ticket_id}。"
        )

    conversation_store.append_turn(
        session_id=session_id,
        question=request.question,
        answer=service_result["answer"]
    )

    # 把前面`build_metrics`生成的指标字典拿出来，方便后面读取。
    metrics = service_result["metrics"]
    handoff_label = str(agent_decision.handoff_required).lower()
    metrics_registry.increment(
        "agent_chat_requests_total",
        intent=agent_decision.intent,
        action=agent_decision.action,
        handoff=handoff_label,
    )
    metrics_registry.observe(
        "agent_pipeline_duration_seconds",
        metrics["total_latency_ms"] / 1000,
        action=agent_decision.action,
    )
    metrics_registry.observe(
        "agent_retrieval_duration_seconds",
        metrics["retrieval_latency_ms"] / 1000,
    )
    metrics_registry.observe(
        "agent_llm_duration_seconds",
        metrics["llm_latency_ms"] / 1000,
        provider=metrics["provider"],
    )
    llm_retry_count = int(metrics.get("llm_retry_count", 0))
    llm_circuit_state = str(
        metrics.get("llm_circuit_state", "unknown")
    )
    if llm_retry_count:
        metrics_registry.increment(
            "llm_retries_total",
            amount=llm_retry_count,
            provider=metrics["provider"],
        )
    metrics_registry.increment(
        "llm_circuit_state_total",
        provider=metrics["provider"],
        state=llm_circuit_state,
    )
    if metrics["vision_fallback_used"]:
        metrics_registry.increment("vision_fallback_total")
    # `logger.info()`：输出普通成功信息日志
    logger.info(
        (
            "chat_completed session_id=%s "
            "request_id=%s "
            "provider=%s model=%s "
            "vision_analysis_ms=%s vision_fallback=%s "
            "retrieval_ms=%s llm_ms=%s total_ms=%s "
            "prompt_tokens=%s completion_tokens=%s "
            "total_tokens=%s references=%s "
            "finish_reason=%s selected_images=%s "
            "intent=%s action=%s handoff=%s ticket_id=%s"
        ),

        session_id,
        http_request.state.request_id,
        metrics["provider"],
        metrics["model"],
        metrics["vision_analysis_latency_ms"],
        metrics["vision_fallback_used"],
        metrics["retrieval_latency_ms"],
        metrics["llm_latency_ms"],
        metrics["total_latency_ms"],
        metrics["prompt_tokens"],
        metrics["completion_tokens"],
        metrics["total_tokens"],
        metrics["context_record_count"],
        metrics["finish_reason"],
        metrics["selected_image_count"],
        agent_decision.intent,
        agent_decision.action,
        agent_decision.handoff_required,
        ticket_id,
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
            request_id=http_request.state.request_id,
            timestamp=int(datetime.now(timezone.utc).timestamp()),
            references=references,
            image_ids=service_result["image_ids"],
            images=[
                KnowledgeImage(
                    image_id=image_id,
                    url=f"/v1/knowledge-images/{image_id}",
                )
                for image_id in service_result["image_ids"]
                if knowledge_image_catalog.available(image_id)
            ],
            agent=AgentMetadata(
                intent=agent_decision.intent,
                action=agent_decision.action,
                confidence=agent_decision.confidence,
                handoff_required=agent_decision.handoff_required,
                handoff_reason=agent_decision.handoff_reason,
                ticket_id=ticket_id,
            ),
            metrics=ChatMetrics(
                # `**` 字典解包，把字典里所有 `key:value`，拆成一个个独立的关键字参数传给类 / 函数。（`*列表`：列表解包 → 拆成位置参数）
                # **：自动读取字典每一项，key 变成参数名，value 变成参数值，一次性全部传入 `ChatMetrics` 的构造函数。
                **service_result["metrics"]
            ),
        )
    )

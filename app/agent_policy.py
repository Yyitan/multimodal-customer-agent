from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class AgentDecision:
    """Deterministic routing decision made before or after retrieval."""

    intent: str
    action: str
    confidence: float
    handoff_required: bool
    handoff_reason: str | None = None
    response_message: str | None = None


SAFETY_TERMS = (
    "起火",
    "冒烟",
    "漏电",
    "触电",
    "爆炸",
    "烧焦",
    "受伤",
    "烫伤",
)

HUMAN_HANDOFF_TERMS = (
    "人工客服",
    "转人工",
    "真人客服",
    "人工处理",
    "客服人员",
    "我要投诉",
)

BUSINESS_ACTION_TERMS = (
    "退款",
    "退货",
    "换货",
    "订单",
    "物流",
    "发票",
    "支付",
    "扣款",
    "账户",
    "账号",
    "保修申请",
    "预约维修",
)

TROUBLESHOOTING_TERMS = (
    "故障",
    "报错",
    "不能",
    "无法",
    "不工作",
    "异常",
    "闪烁",
    "异响",
    "过热",
)

PRODUCT_GUIDANCE_TERMS = (
    "怎么用",
    "如何使用",
    "怎么安装",
    "如何安装",
    "怎么更换",
    "如何更换",
    "设置",
    "操作步骤",
)


def classify_request(question: str) -> AgentDecision:
    """Classify a request and decide whether knowledge search is appropriate."""
    clean_question = question.strip().lower()

    if any(term in clean_question for term in SAFETY_TERMS):
        return AgentDecision(
            intent="safety_risk",
            action="handoff_to_human",
            confidence=0.98,
            handoff_required=True,
            handoff_reason="potential_safety_risk",
            response_message=(
                "该情况可能涉及安全风险。请立即停止继续使用设备，"
                "在确保人身安全的前提下断开电源，并由人工客服进一步处理。"
            ),
        )

    if any(term in clean_question for term in HUMAN_HANDOFF_TERMS):
        return AgentDecision(
            intent="human_handoff",
            action="handoff_to_human",
            confidence=1.0,
            handoff_required=True,
            handoff_reason="user_requested_human",
            response_message="已识别到您希望联系人工客服。",
        )

    if any(term in clean_question for term in BUSINESS_ACTION_TERMS):
        return AgentDecision(
            intent="business_action",
            action="handoff_to_human",
            confidence=0.95,
            handoff_required=True,
            handoff_reason="business_system_action_required",
            response_message=(
                "该请求涉及订单、支付或售后业务操作，"
                "需要由人工客服核验信息后处理。"
            ),
        )

    if any(term in clean_question for term in TROUBLESHOOTING_TERMS):
        return AgentDecision(
            intent="troubleshooting",
            action="search_knowledge",
            confidence=0.85,
            handoff_required=False,
        )

    if any(term in clean_question for term in PRODUCT_GUIDANCE_TERMS):
        return AgentDecision(
            intent="product_guidance",
            action="search_knowledge",
            confidence=0.8,
            handoff_required=False,
        )

    return AgentDecision(
        intent="knowledge_question",
        action="search_knowledge",
        confidence=0.6,
        handoff_required=False,
    )


def review_evidence(
    decision: AgentDecision,
    service_result: dict[str, Any],
) -> AgentDecision:
    """Review retrieved evidence before allowing a knowledge-based response."""
    if decision.handoff_required:
        return decision

    references = service_result.get("references") or []
    metrics = service_result.get("metrics") or {}
    context_record_count = int(metrics.get("context_record_count", 0))
    if not references or context_record_count < 1:
        return replace(
            decision,
            action="handoff_to_human",
            confidence=0.9,
            handoff_required=True,
            handoff_reason="insufficient_knowledge_evidence",
            response_message=(
                "当前知识库没有足够资料支持可靠回答，"
                "需要由人工客服进一步确认。"
            ),
        )

    return replace(
        decision,
        action="respond_with_knowledge",
    )

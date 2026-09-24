from app.agent_policy import (
    classify_request,
    review_evidence,
)


def main() -> None:
    checks: dict[str, bool] = {}

    safety = classify_request("设备突然冒烟并且有烧焦味，怎么办？")
    checks["安全风险直接转人工"] = (
        safety.intent == "safety_risk"
        and safety.handoff_required
        and safety.handoff_reason == "potential_safety_risk"
    )

    human = classify_request("我要转人工客服")
    checks["明确要求人工时直接转接"] = (
        human.intent == "human_handoff"
        and human.handoff_required
    )

    business = classify_request("我的订单需要退款")
    checks["业务系统操作不交给模型处理"] = (
        business.intent == "business_action"
        and business.handoff_required
    )

    troubleshooting = classify_request("充电指示灯一直闪烁是什么故障？")
    checks["故障问题进入知识检索"] = (
        troubleshooting.intent == "troubleshooting"
        and troubleshooting.action == "search_knowledge"
        and not troubleshooting.handoff_required
    )

    reviewed = review_evidence(
        troubleshooting,
        {
            "references": [{"chunk_id": "record-1"}],
            "metrics": {"context_record_count": 1},
        },
    )
    checks["有知识证据时允许生成回答"] = (
        reviewed.action == "respond_with_knowledge"
        and not reviewed.handoff_required
    )

    insufficient = review_evidence(
        troubleshooting,
        {
            "references": [],
            "metrics": {"context_record_count": 0},
        },
    )
    checks["知识证据不足时转人工"] = (
        insufficient.handoff_required
        and insufficient.handoff_reason
        == "insufficient_knowledge_evidence"
    )

    print("====Agent决策测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\nAgent决策测试：全部通过")


if __name__ == "__main__":
    main()

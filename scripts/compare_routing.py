# -*- coding: utf-8 -*-
"""三档路由模式对比实验：rules / llm / hybrid 在同一评测集上的表现。

回答"路由该用规则还是大模型"：
- rules  ：确定性关键词规则（零成本、零延迟、可解释）
- llm    ：纯大模型 function-calling 决策
- hybrid ：安全/业务/转人工硬规则先短路作护栏，其余交给大模型

指标：案例通过率、意图/动作准确率、转人工 P/R/F1、
平均路由延迟、平均路由 token、真实 LLM 调用次数。

知识类案例沿用 agent 评测的 has_evidence 模拟证据复核，
三档模式使用完全相同的判定与后处理，保证可比性。
结果写入 artifacts/evaluation/routing_comparison.json。
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent_policy import classify_request, review_evidence
from app.config import get_settings
from app.llm_router import needs_llm_routing, route_with_llm
from scripts.evaluation_support import load_cases

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "agent_cases.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "evaluation" / "routing_comparison.json"
)


def route_one(
    mode: str,
    question: str,
    settings: Any,
) -> tuple[Any, dict[str, Any]]:
    """按指定模式路由单个问题，返回 (decision, routing_meta)。

    llm/hybrid 模式下路由调用失败（未返回工具调用、超时、参数非法等）
    回退确定性规则并标记 routed_by=rules_fallback，与生产行为一致。
    """
    if mode == "rules":
        return classify_request(question), {"routed_by": "rules", "tokens": 0}

    if mode in ("llm", "hybrid"):
        if mode == "hybrid":
            guard = classify_request(question)
            if not needs_llm_routing(guard):
                return guard, {"routed_by": "rules_guardrail", "tokens": 0}
        try:
            result = route_with_llm(question, settings)
        except Exception:
            return classify_request(question), {
                "routed_by": "rules_fallback",
                "tokens": 0,
                "latency_ms": 0,
            }
        return result["decision"], {
            "routed_by": "llm",
            "tokens": result["usage"].get("total_tokens") or 0,
            "latency_ms": result["latency_ms"],
        }

    raise ValueError(f"未知路由模式：{mode}")


def evaluate_case(
    case: dict[str, Any],
    mode: str,
    settings: Any,
) -> dict[str, Any]:
    decision, meta = route_one(mode, case["question"], settings)
    if "has_evidence" in case:
        has_evidence = bool(case["has_evidence"])
        decision = review_evidence(
            decision,
            {
                "references": (
                    [{"chunk_id": "eval-evidence"}] if has_evidence else []
                ),
                "metrics": {"context_record_count": 1 if has_evidence else 0},
            },
        )

    checks = {
        "intent": decision.intent == case["expected_intent"],
        "action": decision.action == case["expected_action"],
        "handoff": decision.handoff_required == case["expected_handoff"],
        "handoff_reason": decision.handoff_reason
        == case.get("expected_handoff_reason"),
    }
    return {
        "case_id": case["case_id"],
        "expected_intent": case["expected_intent"],
        "expected_action": case["expected_action"],
        "actual_intent": decision.intent,
        "actual_action": decision.action,
        "routed_by": meta["routed_by"],
        "tokens": meta.get("tokens", 0),
        "latency_ms": meta.get("latency_ms", 0),
        "checks": checks,
        "passed": all(checks.values()),
    }


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    true_positive = sum(
        1
        for r in results
        if r["checks"]["handoff"]
        and r["expected_action"] == "handoff_to_human"
    )
    false_positive = sum(
        1
        for r in results
        if r["checks"]["handoff"] is False
        and r["actual_action"] == "handoff_to_human"
        and r["expected_action"] != "handoff_to_human"
    )
    false_negative = sum(
        1
        for r in results
        if r["expected_action"] == "handoff_to_human"
        and r["actual_action"] != "handoff_to_human"
    )
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    llm_routed = [r for r in results if r["routed_by"] == "llm"]
    fallbacks = [r for r in results if r["routed_by"] == "rules_fallback"]
    return {
        "case_count": total,
        "pass_rate": round(safe_ratio(sum(r["passed"] for r in results), total), 4),
        "intent_accuracy": round(
            safe_ratio(sum(r["checks"]["intent"] for r in results), total), 4
        ),
        "action_accuracy": round(
            safe_ratio(sum(r["checks"]["action"] for r in results), total), 4
        ),
        "handoff_precision": round(precision, 4),
        "handoff_recall": round(recall, 4),
        "handoff_f1": round(f1, 4),
        "llm_call_count": len(llm_routed),
        "fallback_count": len(fallbacks),
        "fallback_case_ids": [r["case_id"] for r in fallbacks],
        "total_tokens": sum(r["tokens"] for r in results),
        "avg_latency_ms": round(
            safe_ratio(
                sum(r["latency_ms"] for r in llm_routed), len(llm_routed)
            ),
            1,
        )
        if llm_routed
        else 0.0,
        "failed_case_ids": [r["case_id"] for r in results if not r["passed"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="三档路由模式对比。")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--mode",
        action="append",
        choices=["rules", "llm", "hybrid"],
        help="只跑指定模式，可重复；默认三档全跑。",
    )
    args = parser.parse_args()

    cases = load_cases(CASES_PATH)
    settings = get_settings()
    if settings.llm_provider == "mock":
        raise SystemExit("路由对比需要真实模型，请先配置 AGENT_LLM_PROVIDER。")

    modes = tuple(args.mode) if args.mode else ("rules", "llm", "hybrid")
    results: dict[str, Any] = {}
    for mode in modes:
        print(f"\n====路由模式：{mode}====")
        case_results = [
            evaluate_case(case, mode, settings) for case in cases
        ]
        summary = summarize(case_results)
        results[mode] = {"summary": summary, "per_case": case_results}
        print(
            f"通过率 {summary['pass_rate']:.2%} | 意图 {summary['intent_accuracy']:.2%} "
            f"| 动作 {summary['action_accuracy']:.2%} | 转人工F1 {summary['handoff_f1']:.2%}"
        )
        print(
            f"LLM调用 {summary['llm_call_count']} 次 | token {summary['total_tokens']} "
            f"| 平均路由延迟 {summary['avg_latency_ms']}ms"
        )
        if summary["failed_case_ids"]:
            print("失败案例：", summary["failed_case_ids"])

    output = {
        "comparison_type": "routing_modes",
        "case_count": len(cases),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {"mode": mode, **payload} for mode, payload in results.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("\n报告路径：", args.output)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from typing import Any

from app.agent_policy import classify_request, review_evidence
from scripts.evaluation_support import load_cases, select_cases, write_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "agent_cases.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "evaluation" / "agent.json"


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    decision = classify_request(case["question"])
    if "has_evidence" in case:
        has_evidence = bool(case["has_evidence"])
        decision = review_evidence(
            decision,
            {
                "references": ([{"chunk_id": "eval-evidence"}] if has_evidence else []),
                "metrics": {"context_record_count": 1 if has_evidence else 0},
            },
        )

    checks = {
        "intent": decision.intent == case["expected_intent"],
        "action": decision.action == case["expected_action"],
        "handoff": decision.handoff_required == case["expected_handoff"],
        "handoff_reason": decision.handoff_reason == case.get("expected_handoff_reason"),
    }
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "category": case["category"],
        "expected": {
            "intent": case["expected_intent"],
            "action": case["expected_action"],
            "handoff_required": case["expected_handoff"],
            "handoff_reason": case.get("expected_handoff_reason"),
        },
        "actual": {
            "intent": decision.intent,
            "action": decision.action,
            "handoff_required": decision.handoff_required,
            "handoff_reason": decision.handoff_reason,
            "confidence": decision.confidence,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    total = len(results)
    intent_correct = sum(result["checks"]["intent"] for result in results)
    action_correct = sum(result["checks"]["action"] for result in results)
    true_positive = sum(
        result["expected"]["handoff_required"]
        and result["actual"]["handoff_required"]
        for result in results
    )
    false_positive = sum(
        not result["expected"]["handoff_required"]
        and result["actual"]["handoff_required"]
        for result in results
    )
    false_negative = sum(
        result["expected"]["handoff_required"]
        and not result["actual"]["handoff_required"]
        for result in results
    )
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return {
        "intent_accuracy": safe_ratio(intent_correct, total),
        "action_accuracy": safe_ratio(action_correct, total),
        "handoff_precision": precision,
        "handoff_recall": recall,
        "handoff_f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Agent 路由与转人工评测。")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="只运行指定案例，可重复传入。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON 评测报告输出路径。",
    )
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=1.0,
        help="意图、动作和转人工 F1 的最低通过阈值。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.minimum_score <= 1:
        raise SystemExit("--minimum-score 必须位于 0 到 1 之间。")

    cases = select_cases(load_cases(CASES_PATH), args.case_id)
    results = [evaluate_case(case) for case in cases]
    metrics = calculate_metrics(results)
    report = write_report(
        args.output,
        evaluation_type="agent_policy",
        results=results,
        summary_fields=metrics,
    )

    print("====Agent 专项评测====")
    for result in results:
        print(
            f"{'通过' if result['passed'] else '失败'}|"
            f"{result['case_id']}|{result['category']}|"
            f"{result['actual']['intent']} -> {result['actual']['action']}"
        )

    print("\n评测指标")
    print("案例通过率：", f"{report['summary']['pass_rate']:.2%}")
    print("意图准确率：", f"{metrics['intent_accuracy']:.2%}")
    print("动作准确率：", f"{metrics['action_accuracy']:.2%}")
    print("转人工 Precision：", f"{metrics['handoff_precision']:.2%}")
    print("转人工 Recall：", f"{metrics['handoff_recall']:.2%}")
    print("转人工 F1：", f"{metrics['handoff_f1']:.2%}")
    print("报告路径：", args.output)

    score_fields = ("intent_accuracy", "action_accuracy", "handoff_f1")
    if report["summary"]["failed_count"] or any(
        metrics[field] < args.minimum_score for field in score_fields
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

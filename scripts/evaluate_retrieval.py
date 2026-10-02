import argparse
from pathlib import Path
from typing import Any

from app.context_builder import build_context_package
from app.hybrid_retriever import hybrid_search
from scripts.evaluation_support import (
    load_cases,
    select_cases,
    write_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "evaluation" / "retrieval.json"
)


def collect_image_ranks(
    results: list[dict[str, Any]],
) -> dict[str, int]:
    """Record the first rank at which each image appears."""
    image_ranks: dict[str, int] = {}
    for rank, record in enumerate(results, start=1):
        for image_id in record["image_ids"]:
            image_ranks.setdefault(image_id, rank)
    return image_ranks


def evaluate_case(
    case: dict[str, Any],
    max_results: int = 20,
) -> dict[str, Any]:
    """Evaluate retrieval quality for one case."""
    results = hybrid_search(
        case["question"],
        max_results=max_results,
    )
    context_package = build_context_package(results)
    selected_records = context_package["records"]

    expected_product = case.get("expected_product")
    wrong_product_records = [
        record
        for record in selected_records
        if expected_product and record["product"] != expected_product
    ]
    product_consistent = not wrong_product_records

    image_ranks = collect_image_ranks(selected_records)
    expected_image_ids = set(case["expected_image_ids"])
    returned_image_ids = set(image_ranks)
    found_image_ids = expected_image_ids & returned_image_ids
    missing_image_ids = expected_image_ids - returned_image_ids
    found_ranks = [image_ranks[image_id] for image_id in found_image_ids]
    passed = found_image_ids == expected_image_ids and product_consistent
    known_failure = bool(case.get("known_failure", False))

    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "expected_count": len(expected_image_ids),
        "found_count": len(found_image_ids),
        "missing_image_ids": sorted(missing_image_ids),
        "all_found": found_image_ids == expected_image_ids,
        "first_rank": min(found_ranks) if found_ranks else None,
        "result_count": len(results),
        "context_length": context_package["context_length"],
        "expected_product": expected_product,
        "product_consistent": product_consistent,
        "wrong_product_chunk_ids": [
            record["chunk_id"] for record in wrong_product_records
        ],
        "known_failure": known_failure,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行检索质量评测。")
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="只运行指定案例，可重复传入。",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="混合检索最多返回的记录数。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON 评测报告输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_results < 1:
        raise SystemExit("--max-results 必须大于等于 1。")

    cases = select_cases(load_cases(CASES_PATH), args.case_id)
    results: list[dict[str, Any]] = []
    for case in cases:
        result = evaluate_case(case, max_results=args.max_results)
        results.append(result)

        print("\n" + "=" * 60)
        print("案例：", result["case_id"])
        print("问题：", result["question"])
        print(
            "找到图片：",
            f"{result['found_count']}/{result['expected_count']}",
        )
        print("产品范围正确：", result["product_consistent"])
        print("缺少图片：", result["missing_image_ids"])
        print("第一张目标图片排名：", result["first_rank"])
        print("案例是否通过：", result["passed"])

    report = write_report(
        args.output,
        evaluation_type="retrieval",
        results=results,
        summary_fields={
            "known_failure_count": sum(
                1 for result in results if result["known_failure"]
            ),
            "known_failure_passed_count": sum(
                1
                for result in results
                if result["known_failure"] and result["passed"]
            ),
        },
    )
    summary = report["summary"]
    print("\n" + "=" * 60)
    print("检索评测总结")
    print("案例总数：", summary["total_count"])
    print("通过案例数：", summary["passed_count"])
    print("通过率：", f"{summary['pass_rate']:.2%}")
    print("已知失败案例（不阻断）：", summary["known_failure_count"])
    if summary["known_failure_passed_count"]:
        print(
            "警告：已知失败案例通过了，应移除 known_failure 标记转正：",
            [
                result["case_id"]
                for result in results
                if result["known_failure"] and result["passed"]
            ],
        )
    print("报告路径：", args.output)

    unexpected_failures = [
        result
        for result in results
        if not result["passed"] and not result["known_failure"]
    ]
    if unexpected_failures:
        print(
            "意外失败案例：",
            [result["case_id"] for result in unexpected_failures],
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

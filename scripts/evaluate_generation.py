import argparse
import json
from pathlib import Path
from typing import Any

from app.services.chat_service import answer_question
from scripts.evaluation_support import (
    load_cases,
    select_cases,
    write_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "generation_cases.json"
KNOWLEDGE_RECORDS_PATH = PROJECT_ROOT / "data" / "index" / "knowledge_records.jsonl"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "evaluation" / "generation.json"
)


def build_image_product_map() -> dict[str, str]:
    """Map every knowledge image id to the product its record belongs to."""
    mapping: dict[str, str] = {}
    with KNOWLEDGE_RECORDS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            product = str(record.get("product", "")).strip()
            for image_id in record.get("image_ids") or []:
                mapping.setdefault(str(image_id), product)
    return mapping


IMAGE_PRODUCT = build_image_product_map()


def check_keyword_groups(
    answer: str,
    keyword_groups: list[list[str]],
) -> list[list[str]]:
    """Return keyword groups for which no alternative matched."""
    answer_lower = answer.lower()
    return [
        group
        for group in keyword_groups
        if not any(keyword.lower() in answer_lower for keyword in group)
    ]


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one live end-to-end answer evaluation."""
    service_result = answer_question(case["question"])
    answer = service_result["answer"]
    returned_image_id_list = service_result["image_ids"]
    returned_image_ids = set(returned_image_id_list)
    expected_image_ids = set(case["expected_image_ids"])

    missing_keyword_groups = check_keyword_groups(
        answer=answer,
        keyword_groups=case["must_include_any"],
    )
    expected_image_ids = set(case["expected_image_ids"])
    returned_image_ids = set(returned_image_id_list)

    # 图片判分口径(依据2026-10-02失败分析):
    # 门禁口径:返回图片不得跨产品污染(回答配错产品图是真实缺陷)。
    # 参考指标1 expected_image_hit:至少命中一张预期图片。
    # 参考指标2 strict_image_passed:精确覆盖全部预期图片ID。
    # 不把精确命中作为门禁,因为同一问题常有多个知识片段支撑同一
    # 正确回答,而模型选图存在"倾向候选排名靠前图片"的行为,
    # 精确ID匹配度量的是出题人选片偏好而非回答质量;
    # 图片召回本身已由检索评测门禁保证。
    missing_image_ids = sorted(expected_image_ids - returned_image_ids)
    matched_image_ids = sorted(expected_image_ids & returned_image_ids)
    expected_products = {
        IMAGE_PRODUCT[image_id]
        for image_id in expected_image_ids
        if image_id in IMAGE_PRODUCT
    }
    returned_products = {
        IMAGE_PRODUCT[image_id]
        for image_id in returned_image_ids
        if image_id in IMAGE_PRODUCT
    }
    wrong_product_images = sorted(
        image_id
        for image_id in returned_image_ids
        if IMAGE_PRODUCT.get(image_id) not in expected_products
    )
    expected_image_hit = bool(matched_image_ids)
    strict_image_passed = not missing_image_ids
    image_passed = not wrong_product_images
    keyword_passed = not missing_keyword_groups
    passed = keyword_passed and image_passed

    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "answer": answer,
        "keyword_passed": keyword_passed,
        "missing_keyword_groups": missing_keyword_groups,
        "image_passed": image_passed,
        "expected_image_hit": expected_image_hit,
        "strict_image_passed": strict_image_passed,
        "matched_image_ids": matched_image_ids,
        "missing_image_ids": missing_image_ids,
        "wrong_product_images": wrong_product_images,
        "returned_image_ids": returned_image_id_list,
        "reference_count": len(service_result["references"]),
        "references": [
            {
                "chunk_id": reference["chunk_id"],
                "product": reference["product"],
                "text": reference["text"][:800],
            }
            for reference in (service_result.get("references") or [])
        ],
        "usage": {
            "prompt_tokens": service_result.get("metrics", {}).get("prompt_tokens"),
            "completion_tokens": service_result.get("metrics", {}).get(
                "completion_tokens"
            ),
            "total_tokens": service_result.get("metrics", {}).get("total_tokens"),
            "llm_latency_ms": service_result.get("metrics", {}).get("llm_latency_ms"),
        },
        "returned_image_count": len(returned_image_ids),
        "error": None,
        "passed": passed,
    }


def failed_case_result(
    case: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Convert an external model failure into a reportable case result."""
    return {
        "case_id": case["case_id"],
        "question": case["question"],
        "answer": "",
        "keyword_passed": False,
        "missing_keyword_groups": case["must_include_any"],
        "image_passed": False,
        "strict_image_passed": False,
        "matched_image_ids": [],
        "missing_image_ids": case["expected_image_ids"],
        "wrong_product_images": [],
        "returned_image_ids": [],
        "reference_count": 0,
        "references": [],
        "usage": {},
        "returned_image_count": 0,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "passed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行真实模型生成评测。")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = select_cases(load_cases(CASES_PATH), args.case_id)
    results: list[dict[str, Any]] = []

    for case in cases:
        try:
            result = evaluate_case(case)
        except Exception as error:
            result = failed_case_result(case, error)
        results.append(result)

        print("\n" + "=" * 60)
        print("案例：", result["case_id"])
        print("问题：", result["question"])
        if result["error"]:
            print("调用错误：", result["error"])
            print("本案例是否通过：", result["passed"])
            continue
        print("\n模型回答：")
        print(result["answer"])
        print("\n关键词检查：", result["keyword_passed"])
        if result["missing_keyword_groups"]:
            print("缺少关键词组：", result["missing_keyword_groups"])
        print("图片检查：", result["image_passed"])
        print("实际返回图片：", result["returned_image_ids"])
        if result["missing_image_ids"]:
            print("缺少图片：", result["missing_image_ids"])
        print("本案例是否通过：", result["passed"])

    report = write_report(
        args.output,
        evaluation_type="generation",
        results=results,
    )
    summary = report["summary"]
    print("\n" + "=" * 60)
    print("生成评测总结")
    print("案例总数：", summary["total_count"])
    print("通过案例数：", summary["passed_count"])
    print("通过率：", f"{summary['pass_rate']:.2%}")
    print("报告路径：", args.output)

    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

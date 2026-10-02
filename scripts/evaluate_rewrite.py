# -*- coding: utf-8 -*-
"""多轮检索改写消融：对比三种检索查询构造方式在追问场景下的表现。

- question_only：只用追问原文（无历史，作为下界基线）
- concat       ：历史原文 + 追问拼接（旧行为）
- rewrite      ：大模型把追问改写成可独立检索的完整问题（新行为）

指标：预期图片Recall@20、平均命中比例、产品一致性、改写延迟与token。
结果写入 artifacts/evaluation/query_rewrite_ablation.json。
"""
import argparse
import json
import time as time_module
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.conversation_store import ConversationTurn
from app.hybrid_retriever import hybrid_search
from app.query_rewriter import rewrite_query_with_history
from app.services.chat_service import build_history_aware_retrieval_query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "query_rewrite_cases.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "artifacts" / "evaluation" / "query_rewrite_ablation.json"
)


def load_multi_turn_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    for case in cases:
        for field in (
            "case_id",
            "turn1_question",
            "turn1_answer",
            "turn2_question",
            "expected_product",
            "expected_image_ids",
        ):
            if field not in case:
                raise ValueError(f"案例缺少字段 {field}：{case}")
    return cases


def build_history(case: dict[str, Any]) -> list[ConversationTurn]:
    return [
        ConversationTurn(
            question=case["turn1_question"],
            answer=case["turn1_answer"],
        )
    ]


def build_query(
    mode: str,
    case: dict[str, Any],
    settings: Any,
) -> tuple[str, dict[str, Any]]:
    """按模式构造检索查询，返回 (query, meta)。"""
    question = case["turn2_question"]
    if mode == "question_only":
        return question, {}
    history = build_history(case)
    if mode == "concat":
        return build_history_aware_retrieval_query(question, history), {}
    if mode == "rewrite":
        result = rewrite_query_with_history(question, history, settings)
        return result["query"], {
            "latency_ms": result["latency_ms"],
            "total_tokens": result["total_tokens"],
        }
    raise ValueError(f"未知模式：{mode}")


def evaluate_mode(
    mode: str,
    cases: list[dict[str, Any]],
    settings: Any,
    max_results: int,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    total_tokens = 0
    rewrite_latency: list[int] = []

    for case in cases:
        try:
            query, meta = build_query(mode, case, settings)
            rewrite_failed = False
        except Exception as error:
            query, meta = case["turn2_question"], {}
            rewrite_failed = True
            print(f"改写失败|{case['case_id']}|{error}")
        total_tokens += meta.get("total_tokens") or 0
        if "latency_ms" in meta:
            rewrite_latency.append(meta["latency_ms"])

        started = time_module.perf_counter()
        results = hybrid_search(query, max_results=max_results)
        retrieval_ms = (time_module.perf_counter() - started) * 1000

        image_ranks: dict[str, int] = {}
        for rank, record in enumerate(results, start=1):
            for image_id in record.get("image_ids") or []:
                image_ranks.setdefault(image_id, rank)
        expected_images = set(case["expected_image_ids"])
        found = expected_images & set(image_ranks)
        first_rank = min(
            (image_ranks[i] for i in found), default=None
        )
        wrong_products = sorted(
            {
                record["product"]
                for record in results
                if record["product"] != case["expected_product"]
            }
        )
        per_case.append(
            {
                "case_id": case["case_id"],
                "query": query,
                "rewrite_failed": rewrite_failed,
                "found_image_count": len(found),
                "expected_image_count": len(expected_images),
                "all_found": found == expected_images,
                "first_rank": first_rank,
                "wrong_products": wrong_products[:5],
                "retrieval_ms": round(retrieval_ms, 2),
            }
        )
        status = "通过" if per_case[-1]["all_found"] else "失败"
        print(
            f"{status}|{case['case_id']}|query={query[:40]}|"
            f"{len(found)}/{len(expected_images)}|rank={first_rank}"
        )

    total_expected = sum(c["expected_image_count"] for c in per_case)
    total_found = sum(c["found_image_count"] for c in per_case)
    return {
        "mode": mode,
        "case_count": len(cases),
        "all_found_rate": round(
            sum(1 for c in per_case if c["all_found"]) / len(cases), 4
        )
        if cases
        else 0.0,
        "image_recall": round(total_found / total_expected, 4)
        if total_expected
        else 0.0,
        "product_consistency": round(
            sum(1 for c in per_case if not c["wrong_products"]) / len(cases), 4
        )
        if cases
        else 0.0,
        "total_tokens": total_tokens,
        "avg_rewrite_latency_ms": round(
            sum(rewrite_latency) / len(rewrite_latency), 1
        )
        if rewrite_latency
        else None,
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮检索改写消融实验。")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    cases = load_multi_turn_cases(CASES_PATH)
    settings = get_settings()
    if settings.llm_provider == "mock":
        raise SystemExit("改写消融需要真实模型，请先配置 AGENT_LLM_PROVIDER。")

    results = [
        evaluate_mode(mode, cases, settings, args.max_results)
        for mode in ("question_only", "concat", "rewrite")
    ]

    output = {
        "ablation_type": "query_rewrite",
        "case_count": len(cases),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("\n====多轮检索改写消融总结====")
    header = "| 模式 | 全部命中 | 图片Recall | 产品一致性 | token |"
    print(header)
    print("| --- | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result['mode']} "
            f"| {result['all_found_rate']:.2%} "
            f"| {result['image_recall']:.2%} "
            f"| {result['product_consistency']:.2%} "
            f"| {result['total_tokens']} |"
        )
    print("报告路径：", args.output)


if __name__ == "__main__":
    main()

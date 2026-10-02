# -*- coding: utf-8 -*-
"""检索消融实验：用同一评测集对比混合检索各设计层的贡献。

回答"每个设计层到底有没有用"：
1. keyword_only          仅关键词召回
2. keyword_only_neighbor 关键词召回 + 同手册相邻扩展
3. vector_only           仅 BGE 向量召回
4. hybrid_no_filter      关键词+向量合并打分，关闭产品检测
5. hybrid_full           当前完整管线（合并 + 产品检测过滤 + 邻居扩展）

指标：图片 Recall@10/@20、首个命中图片的 MRR、产品一致性、平均延迟。
结果写入 artifacts/ablation/retrieval_ablation.json。
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

from app.hybrid_retriever import hybrid_search
from app.retriever import KNOWLEDGE_RECORDS, search_records
from app.semantic_retriever import semantic_search
from scripts.evaluation_support import load_cases

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "ablation" / "retrieval_ablation.json"


def merge_semantic_keyword(
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """关键词+向量合并打分，但不做产品检测过滤、不做邻居扩展。"""
    semantic_results = semantic_search(query, top_k=max_results * 2)
    keyword_results = search_records(KNOWLEDGE_RECORDS, query)

    candidates: dict[str, dict[str, Any]] = {}
    for record in semantic_results:
        candidate = record.copy()
        semantic_score = candidate.pop("score", 0.0)
        candidate["semantic_score"] = semantic_score
        candidate["keyword_hit"] = False
        candidates[record["chunk_id"]] = candidate
    for record in keyword_results:
        chunk_id = record["chunk_id"]
        if chunk_id in candidates:
            candidates[chunk_id]["keyword_hit"] = True
        else:
            candidate = record.copy()
            candidate["semantic_score"] = 0.0
            candidate["keyword_hit"] = True
            candidates[chunk_id] = candidate

    for candidate in candidates.values():
        candidate["score"] = (
            candidate["semantic_score"] + (1.0 if candidate["keyword_hit"] else 0.0)
        )
    ranked = sorted(candidates.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:max_results]


def run_config(
    config: str,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    if config == "keyword_only":
        return list(search_records(KNOWLEDGE_RECORDS, query))[:max_results]
    if config == "keyword_only_neighbor":
        # 复用 hybrid_search 的邻居扩展：先关键词取种子，再交给完整管线
        # 不合适——这里直接用 expand 语义，简化为对命中记录手工扩展。
        matches = search_records(KNOWLEDGE_RECORDS, query)
        return expand_keyword_neighbors(matches, max_results)
    if config == "vector_only":
        return semantic_search(query, top_k=max_results)
    if config == "hybrid_no_filter":
        return merge_semantic_keyword(query, max_results)
    if config == "hybrid_full":
        return hybrid_search(query, max_results=max_results)
    raise ValueError(f"未知消融配置：{config}")


def expand_keyword_neighbors(
    matches: list[dict[str, Any]],
    max_results: int,
) -> list[dict[str, Any]]:
    """给关键词命中记录补充同一手册的相邻记录（与 hybrid 的邻居逻辑一致）。"""
    position_by_chunk = {
        record["chunk_id"]: index for index, record in enumerate(KNOWLEDGE_RECORDS)
    }
    expanded: dict[str, dict[str, Any]] = {}
    for seed in matches:
        seed_index = position_by_chunk[seed["chunk_id"]]
        for index in range(
            max(0, seed_index - 1),
            min(len(KNOWLEDGE_RECORDS), seed_index + 2),
        ):
            record = KNOWLEDGE_RECORDS[index]
            if record["source"] != seed["source"] or record["product"] != seed["product"]:
                continue
            expanded.setdefault(record["chunk_id"], record.copy())
    ranked = list(expanded.values())
    return ranked[:max_results]


def evaluate_config(
    config: str,
    cases: list[dict[str, Any]],
    max_results: int,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    total_latency_ms = 0.0

    for case in cases:
        started = time.perf_counter()
        results = run_config(config, case["question"], max_results)
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_latency_ms += elapsed_ms

        image_ranks: dict[str, int] = {}
        for rank, record in enumerate(results, start=1):
            for image_id in record.get("image_ids") or []:
                image_ranks.setdefault(image_id, rank)

        expected_images = set(case["expected_image_ids"])
        found_images = expected_images & set(image_ranks)
        missing_images = sorted(expected_images - set(image_ranks))
        first_rank = min((image_ranks[i] for i in found_images), default=None)

        expected_product = case.get("expected_product")
        wrong_products = sorted(
            {
                record["product"]
                for record in results
                if expected_product and record["product"] != expected_product
            }
        )

        per_case.append(
            {
                "case_id": case["case_id"],
                "known_failure": bool(case.get("known_failure", False)),
                "expected_image_count": len(expected_images),
                "found_image_count": len(found_images),
                "missing_image_ids": missing_images,
                "first_rank": first_rank,
                "wrong_products": wrong_products,
                "latency_ms": round(elapsed_ms, 2),
            }
        )

    gated = [c for c in per_case if not c["known_failure"]]
    recall_at_10 = _image_recall(per_case, 10)
    recall_at_20 = _image_recall(per_case, 20)
    mrr = _mrr(per_case)
    consistent = sum(1 for c in gated if not c["wrong_products"])

    return {
        "config": config,
        "case_count": len(per_case),
        "image_recall_at_10": round(recall_at_10, 4),
        "image_recall_at_20": round(recall_at_20, 4),
        "mrr": round(mrr, 4),
        "product_consistency": round(consistent / len(gated), 4) if gated else 0.0,
        "avg_latency_ms": round(total_latency_ms / len(per_case), 2)
        if per_case
        else 0.0,
        "per_case": per_case,
    }


def _image_recall(per_case: list[dict[str, Any]], top_n: int) -> float:
    """expected 图片出现在前 top_n 结果中的比例（含已知失败案例，观察排序）。"""
    total = sum(c["expected_image_count"] for c in per_case)
    if not total:
        return 0.0
    found = sum(
        c["expected_image_count"] - len(c["missing_image_ids"])
        for c in per_case
        if c["first_rank"] is None or c["first_rank"] <= top_n
    )
    return found / total


def _mrr(per_case: list[dict[str, Any]]) -> float:
    reciprocal = 0.0
    for case in per_case:
        if case["first_rank"]:
            reciprocal += 1.0 / case["first_rank"]
    return reciprocal / len(per_case) if per_case else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行检索消融实验。")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


CONFIGS = (
    "keyword_only",
    "keyword_only_neighbor",
    "vector_only",
    "hybrid_no_filter",
    "hybrid_full",
)


def main() -> None:
    args = parse_args()
    cases = load_cases(CASES_PATH)

    # 预热向量模型，避免首个案例计入加载耗时
    semantic_search("预热查询", top_k=1)

    results = [
        evaluate_config(config, cases, args.max_results) for config in CONFIGS
    ]
    report = {
        "ablation_type": "retrieval",
        "case_count": len(cases),
        "max_results": args.max_results,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    header = (
        "| 配置 | 图片Recall@10 | 图片Recall@20 | MRR | 产品一致性 | 平均延迟ms |"
    )
    print(header)
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result['config']} "
            f"| {result['image_recall_at_10']:.2%} "
            f"| {result['image_recall_at_20']:.2%} "
            f"| {result['mrr']:.3f} "
            f"| {result['product_consistency']:.2%} "
            f"| {result['avg_latency_ms']:.1f} |"
        )
    print("报告路径：", args.output)


if __name__ == "__main__":
    main()

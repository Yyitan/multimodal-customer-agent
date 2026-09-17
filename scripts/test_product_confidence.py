from unittest.mock import patch
from math import isclose
from app.hybrid_retriever import build_candidates
from app.vision_analyzer import (
    choose_product_retrieval_strategy,
    normalize_product_confidence,
)

def main() -> None:
    product = "健身追踪器"
    checks = {
        "保留正常置信度": (
            normalize_product_confidence(
                0.85,
                product,
            )
            == 0.85
        ),
        "超过1时限制为1": (
            normalize_product_confidence(
                1.5,
                product,
            )
            == 1.0
        ),
        "小于0时限制为0": (
            normalize_product_confidence(
                -0.3,
                product,
            )
            == 0.0
        ),
        "字符串置信度视为无效": (
            normalize_product_confidence(
                "很确定",
                product,
            )
            == 0.0
        ),
        "布尔值置信度视为无效": (
            normalize_product_confidence(
                True,
                product,
            )
            == 0.0
        ),
        "没有产品名称时置信度为0": (
            normalize_product_confidence(
                0.95,
                "",
            )
            == 0.0
        ),
    }

    # 分别模拟高、中、低置信度，调用策略选择函数，拿到返回的策略配置
    hard_strategy = choose_product_retrieval_strategy({
        "product_hint": product,
        "product_confidence": 0.95,
    })
    soft_strategy = choose_product_retrieval_strategy({
        "product_hint": product,
        "product_confidence": 0.60,
    })
    ignored_strategy = choose_product_retrieval_strategy({
        "product_hint": product,
        "product_confidence": 0.20,
    })

    # 用 checks.update()，往原来的 checks 字典追加 3 条新的测试断言
    checks.update({
        "高置信度使用硬过滤": (
            hard_strategy["strategy"] == "hard_filter"
            and hard_strategy["include_product_hint"] is True
        ),
        "中等置信度使用软加权": (
            soft_strategy["strategy"] == "soft_boost"
            and soft_strategy["preferred_product"] == product
        ),
        "低置信度忽略产品提示": (
            ignored_strategy["strategy"] == "ignored"
            and ignored_strategy["preferred_product"] is None
        ),
    })

    # 模拟向量检索返回两条文档：相机原始分数更高，排在前面。
    semantic_results = [
        {
            "chunk_id": "fitness_001",
            "product": "健身追踪器",
            "text": "更换表带",
            "image_ids": [],
            "source": "fitness.txt",
            "score": 0.55,
        },
        {
            "chunk_id": "camera_001",
            "product": "相机",
            "text": "更换配件",
            "image_ids": [],
            "source": "camera.txt",
            "score": 0.65,
        },
    ]

    # 拦截真实的 semantic_search 函数，不去调用向量库，直接返回我们写死的semantic_results
    with (
        patch(
            "app.hybrid_retriever.semantic_search",
            return_value=semantic_results,
        ),
        patch(
            "app.hybrid_retriever.search_records",
            return_value=[],
        ),
    ):
        # 调用 build_candidates 做加分
        boosted_results = build_candidates(
            query="这个设备怎么更换配件？",
            semantic_top_k=2,
            # preferred_product="健身追踪器" → 给这个产品每一条 chunk +0.15 分
            preferred_product="健身追踪器",
            product_bonus=0.15,
        )

    # 最后增加一条检查项到 checks:排序第一名变成健身追踪器;retrieval_type 标记带上product_hint，证明加分逻辑生效
    checks["软加权能够提升目标产品排序"] = (
        boosted_results[0]["product"] == "健身追踪器"
        # isclose() 的含义是：两个数字只要足够接近，就认为相等。这是测试浮点数的标准方式。
        and isclose(
            boosted_results[0]["score"],
            0.70,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and "product_hint"
        in boosted_results[0]["retrieval_type"]
    )

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n产品置信度规范化测试：全部通过")


if __name__ == "__main__":
    main()
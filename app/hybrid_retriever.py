from typing import Any
from app.retriever import (
    KNOWLEDGE_RECORDS,
    search_records,
)

from app.semantic_retriever import semantic_search

"""RAG检索模块：混合检索：语义检索+关键词检索+上下文窗口扩展"""
# 建立“chunk_id → 记录位置”的映射。
# 以后根据chunk_id查位置时，不需要重复遍历整个列表。
# 索引映射
POSITION_BY_CHUNK_ID = {
    record["chunk_id"]: index for index, record in enumerate(KNOWLEDGE_RECORDS)
}

# 从知识库中收集所有产品名称。
# 按名称长度从长到短排序，避免短名称优先匹配造成误判。
PRODUCT_NAMES = sorted(
    {
        str(record.get("product", "")).strip()
        for record in KNOWLEDGE_RECORDS
        if str(record.get("product", "")).strip()
    },
    key=len,
    reverse=True,
)

# 产品别名词典：口语简称和产品特有部件词 -> 标准产品名。
# 动机：评测发现"鼠标用几号电池"无法命中产品名"蓝牙激光鼠标"，
# "冷藏室的温度怎么调节"无法命中"冰箱"，导致产品硬过滤失效，
# 跨产品记录混入上下文。命中别名与命中产品名同等生效。
# 注意：别名词是启发式，命中后走硬过滤；若问题同时提到多个产品，
# 仍按"标准产品名优先、别名兜底"的顺序取第一个命中。
PRODUCT_ALIASES = {
    "vr": "VR头显",
    "头显": "VR头显",
    "鼠标": "蓝牙激光鼠标",
    "键盘": "功能键盘",
    "手环": "健身追踪器",
    "手表": "健身追踪器",
    "温控器": "可编程温控器",
    "净化器": "空气净化器",
    "工学椅": "人体工学椅",
    "电竞椅": "人体工学椅",
    "电动车": "儿童电动摩托车",
    "冷藏室": "冰箱",
    "冷冻室": "冰箱",
    "水上摩托": "摩托艇",
    "发电机组": "发电机",
    "蒸汽拖把": "蒸汽清洁机",
    "清洁机": "蒸汽清洁机",
    "健身车": "健身单车",
    "单车": "健身单车",
}


def detect_product(query: str) -> str | None:
    """从用户问题中识别明确提到或通过别名指代的产品。"""
    clean_query = query.strip().lower()

    if not clean_query:
        return None

    # 标准产品名优先匹配（长名在前），避免"蓝牙激光鼠标"被别名"鼠标"抢先。
    for product in PRODUCT_NAMES:
        if product.lower() in clean_query:
            return product
    # 问题里没有标准产品名时，再尝试别名映射。
    for alias, product in PRODUCT_ALIASES.items():
        if alias in clean_query:
            return product
    # 都没有命中，返回None
    return None


def build_candidates(
    query: str,
    semantic_top_k: int = 10,
    keyword_bonus: float = 1.0,
    # 默认不加分；只有 soft_boost 策略传入产品名才触发加分。
    preferred_product: str | None = None,
    # 加分值默认 0.15
    product_bonus: float = 0.15,
) -> list[dict[str, Any]]:
    """合并关键词结果和语义结果，并计算混合分数。"""
    # 先判断问题中是否明确提到了某个产品
    target_product = detect_product(query)

    # 如果识别到了产品，语义检索先多取一些候选。
    # 原因：假设只取20条，其中有3条相机记录。如果取完前20条后过滤，相机记录会被删掉，所以先取semantic_top_k * 3 条，再过滤产品，以保留足够多的目标产品记录。
    semantic_fetch_k = semantic_top_k * 3 if target_product else semantic_top_k

    # 调用语义检索，拿到向量召回结果，自带score
    semantic_results = semantic_search(
        query,
        top_k=semantic_fetch_k,
    )

    # 如果问题中明确提到产品，只保留该产品的记录
    if target_product:
        semantic_results = [
            record for record in semantic_results if record["product"] == target_product
        ][:semantic_top_k]

    # 调用关键词检索，返回字面关键词命中记录
    keyword_results = search_records(
        KNOWLEDGE_RECORDS,
        query,
    )

    # 关键词结果也应用同样的产品过滤。
    if target_product:
        keyword_results = [
            record for record in keyword_results if record["product"] == target_product
        ]

    # 使用chunk_id作为字典的键，自动合并同一条知识（空字典，字典key唯一性，同一个chunk_id只能存在一条，如果即被语义召回又被关键词命中，就合并到同一个字典项）
    candidates = {}

    for record in semantic_results:
        chunk_id = record["chunk_id"]
        # .copy字典拷贝
        candidate = record.copy()

        # semantic_search原本把余弦相似度保存在score中。
        # .pop("score")：字典方法，取出key对应的值，同时把这个key从字典删掉（就是把原本的score拿出来存到semantic_score，原字典不再保留score）
        semantic_score = candidate.pop("score")

        candidate["semantic_score"] = semantic_score
        candidate["keyword_hit"] = False
        # 用 chunk_id 去重，字典的键不能重复。同一个 chunk_id 再出现时，会找到同一个候选项。
        candidates[chunk_id] = candidate

    # 遍历关键词命中结果
    for record in keyword_results:
        chunk_id = record["chunk_id"]

        # 就是关键词命中，但是没有出现在语义召回结果里
        if chunk_id not in candidates:
            # copy一份记录
            candidate = record.copy()
            # 没有语义分数
            candidate["semantic_score"] = 0.0
            # 标记关键词命中，存入字典
            candidate["keyword_hit"] = True
            candidates[chunk_id] = candidate
        else:
            # 即关键词命中，又语义命中
            # 已经存在candidates，不需要新建条目，只把标记`keyword_hit`改成 True，实现合并
            candidates[chunk_id]["keyword_hit"] = True
            """"截至到这：candidate字典已经存放全部（语义∪关键词）记录，自动完成去重"""
    # 拿到所有vaule，循环打分
    for candidate in candidates.values():
        semantic_score = candidate["semantic_score"]

        # 先计算关键词和语义检索得分。
        if candidate["keyword_hit"]:
            candidate["score"] = semantic_score + keyword_bonus
            candidate["retrieval_type"] = "keyword+semantic"
        else:
            # 只有语义召回
            candidate["score"] = semantic_score
            candidate["retrieval_type"] = "semantic"

        # 如果产品符合视觉识别结果，再统一增加产品分数。
        if (
            preferred_product
            and candidate["product"] == preferred_product
        ):
            candidate["score"] += product_bonus
            candidate["retrieval_type"] += "+product_hint"

    # sorted:第一个参数:待排序数据;lambda 匿名函数:排序依据,拿每条记录的score作为排序权重
    ranked_candidates = sorted(
        candidates.values(),
        key=lambda record: record["score"],
        # 默认从小到大，reverse=True 降序。
        reverse=True,
    )

    return ranked_candidates


def expand_context(
    seeds: list[dict[str, Any]],
    window_size: int = 1,
    neighbor_penalty: float = 0.05,
    max_results: int = 15,
) -> list[dict[str, Any]]:
    """给候选知识补充统一手册中的相邻记录。"""

    # 空字典,存放原始种子+扩展出来的邻居chunk,同样用chunk_id做key去重
    expanded = {}

    # 先保存直接检索到的种子记录
    for seed in seeds:
        expanded[seed["chunk_id"]] = seed.copy()

    # 再补充每条种子前后的相邻记录
    # 遍历每一条种子记录，通过我们最开始建立的`POSITION_BY_CHUNK_ID`索引，拿到这条种子在全局知识库列表的下标。
    for seed in seeds:
        seed_index = POSITION_BY_CHUNK_ID[seed["chunk_id"]]

        start_index = max(
            0,
            seed_index - window_size,
        )
        end_index = min(
            len(KNOWLEDGE_RECORDS),
            seed_index + window_size + 1,
        )  # 切片 range 是左闭右开，所以要 + 1

        for index in range(start_index, end_index):
            record = KNOWLEDGE_RECORDS[index]

            # 1必须是同一个手册文件才扩展;2必须属于同一个产品。
            if (
                record["source"] != seed["source"]
                or record["product"] != seed["product"]
            ):
                continue

            chunk_id = record["chunk_id"]

            # 如果这个 chunk 已经存在 expanded 字典，说明它本身就是种子，直接跳过，**不要覆盖原始种子的分数**。
            if chunk_id in expanded:
                continue
            # 距离种子片段有多远
            distance = abs(index - seed_index)
            # copy 邻居记录，不修改原始知识库
            neighbor = record.copy()
            # 邻居打分逻辑：`种子分数 − 惩罚系数 ×距离`。离种子越远扣得越多。
            neighbor["score"] = seed["score"] - neighbor_penalty * distance
            # 标记来源：这个片段是哪个种子的邻居扩展出来的，方便调试看结果来源。
            neighbor["retrieval_type"] = f"neighbor_of:{seed['chunk_id']}"
            # 存入expanded字典
            expanded[chunk_id] = neighbor

    # 把【种子记录 + 扩展邻居】全部拿出来，按照 score 再次降序排序。
    ranked_results = sorted(
        expanded.values(),
        key=lambda record: record["score"],
        reverse=True,
    )

    # 只把接口真正需要的字段返回出去。
    public_results = []

    # 列表切片，只取前 max_results 条，截断多余结果。
    for record in ranked_results[:max_results]:
        # 手动构造字典，过滤掉内部临时字段，只对外输出需要的字段。
        public_results.append(
            {
                "chunk_id": record["chunk_id"],
                "product": record["product"],
                "text": record["text"],
                "image_ids": record["image_ids"],
                "source": record["source"],
                # 分数保留6位小数
                "score": round(
                    float(record["score"]),
                    6,
                ),
                "retrieval_type": record["retrieval_type"],
            }
        )
    return public_results


def hybrid_search(
    query: str,
    semantic_top_k: int = 20,
    seed_top_k: int = 10,
    window_size: int = 1,
    max_results: int = 20,
    preferred_product: str | None = None,
    product_bonus: float = 0.15,
) -> list[dict[str, Any]]:
    """执行完整的混合检索流程"""
    # 调用`build_candidates`做语义 + 关键词混合召回打分
    candidates = build_candidates(
        query,
        semantic_top_k=semantic_top_k,
        preferred_product=preferred_product,
        product_bonus=product_bonus,
    )

    # 切片取前`seed_top_k`条，作为种子集合
    seeds = candidates[:seed_top_k]

    # 调用`expand_context`做同手册相邻片段扩展、二次排序、字段过滤截断
    results = expand_context(
        seeds,
        window_size=window_size,
        max_results=max_results,
    )

    return results


if __name__ == "__main__":
    test_questions = [
        "我的DCB107充电器指示灯闪烁是什么意思？",
        "电钻电池温度太高时为什么不能马上充电？",
        "健身追踪器有其他尺寸的表带可以更换吗？",
    ]

    for question in test_questions:
        print("\n========================================")
        print("问题：", question)

        results = hybrid_search(question)

        for rank, record in enumerate(
            results,
            start=1,
        ):
            preview = record["text"][:100].replace(
                "\n",
                " ",
            )

            print(f"\n第{rank}名")
            print(f"score:{record['score']:.4f}")
            print("chunk_id:", record["chunk_id"])
            print(
                "retrieval_type:",
                record["retrieval_type"],
            )
            print("text:", preview)
            print("image_ids:", record["image_ids"])

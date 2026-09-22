import re
from typing import Any

from app.hybrid_retriever import hybrid_search

def clean_text(text:str)->str:
    """清理连续空行和每行两端的空白。"""
    cleaned_lines=[]
    
    # 把文本按换行符切分成每行字符串列表。
    for line in text.splitlines():
        # 去掉一行开头结尾空格、tab
        clean_line=line.strip()
        
        # 空字符串会被判定 False；过滤掉空行，消除连续多个换行。
        if clean_line:
            cleaned_lines.append(clean_line)
    # 把清洗后的行，用单个换行拼接回去。
    return "\n".join(cleaned_lines)

def truncate_text(text:str,max_chars:int,)->str:
    """限制单条知识的最大字符数。"""
    if len(text)<=max_chars:
        return text
    
    # 如果超过字符上限：字符串切片取前`max_chars`字符，追加截断提示文字。
    return (
        text[:max_chars]
        +"\n[该知识块后续内容已经截断]"
    )
    
def format_record(record:dict[str,Any],rank:int,max_text_chars:int,)->str:
    """把一条知识记录转换为适合大模型阅读的文本。"""
    # 先拿 record 里面原始正文，调用`clean_text`清洗空格空行
    text=clean_text(record["text"])
    # 调用`truncate_text`，单条正文做长度截断。
    text=truncate_text(text,max_chars=max_text_chars,)
    
    # 取出关联图片 id 列表：
    image_ids=record["image_ids"]
    # 如果列表非空，用逗号拼接多个图片 id 成为字符串
    if image_ids:
        image_text=",".join(image_ids)
    # 如果没有图片，赋值字符串`"无"`，输出给大模型看。
    else:
        image_text="无"
    
    # f-string 拼接完整资料块文本。
    # 把元信息（id、来源、分数、检索类型）全部暴露给大模型，模型可以参考分数判断可信度，调试也方便。
    return (
        f"[资料 {rank}]\n"
        f"知识块ID：{record['chunk_id']}\n"
        f"产品：{record['product']}\n"
        f"来源：{record['source']}\n"
        f"检索方式：{record['retrieval_type']}\n"
        f"相关分数：{record['score']:.4f}\n"
        f"关联图片：{image_text}\n"
        f"正文：\n{text}"
    )


def _chunk_position(chunk_id: str) -> int | None:
    """提取知识块 ID 末尾的数字，用于判断手册中的相邻片段。"""
    match = re.search(r"(\d+)$", str(chunk_id))
    return int(match.group(1)) if match else None


def _image_position(record: dict[str, Any]) -> int | None:
    """提取记录中第一张图片 ID 末尾的数字。"""
    image_ids = record.get("image_ids") or []
    if not image_ids:
        return None

    match = re.search(r"(\d+)$", str(image_ids[0]))
    return int(match.group(1)) if match else None


def _query_terms(text: str) -> set[str]:
    """提取中英文短词，计算问题与知识块的轻量文字相关性。"""
    # 建一个空集合
    terms: set[str] = set()
    # 轻量分词器，正则 `[\u4e00-\u9fff]+|[a-z0-9][a-z0-9_-]*` 把文本切成两种段：连续中文 / 英文单词或数字串。
    for segment in re.findall(
        r"[\u4e00-\u9fff]+|[a-z0-9][a-z0-9_-]*",
        text.lower(),
    ):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            # 遇到连续中文段，生成 **2-gram、3-gram、4-gram**：比如 "更换表带" 会切出 `更换, 换表, 表带, 更换表, 换表带`。
            for size in (2, 3, 4):
                terms.update(
                    segment[index:index + size]
                    for index in range(len(segment) - size + 1)
                )
        else:
            terms.add(segment)

    return terms


def _record_relevance(
    record: dict[str, Any],
    query_terms: set[str],
    term_document_frequency: dict[str, int],
) -> float:
    """综合语义分数、稀有词和标题命中，辅助图片组排序。"""
    # 给一条 record 算 "跟问题有多相关"，分三层：
    # 取这条知识块的正文
    text = str(record.get("text", ""))
    # 正文也切成词集合
    text_terms = _query_terms(text)
    # 正文命中：问题词出现在正文里。
    matched_terms = query_terms & text_terms
    # 标题命中：正则 `(?:^|\n)\s*#+\s*([^\n]+)` 从正文里抽 markdown 标题。标题命中比正文命中信号强得多。
    headings = " ".join(
        heading[:24]
        for heading in re.findall(
            r"(?:^|\n)\s*#+\s*([^\n]+)",
            text,
        )
    )
    # 把标题也切成词集合。
    heading_terms = _query_terms(headings)
    rare_term_limit = max(
        2,
        max(term_document_frequency.values(), default=0) // 3,
    )
    # 从 "问题词∩标题词" 里，只保留稀有词
    heading_matches = {
        term
        for term in query_terms & heading_terms
        if term_document_frequency.get(term, 0) <= rare_term_limit
    }
    # IDF 加权：`weighted_match_score` 里 `1.0 / df`—— 一个词在多少条 record 出现过，出现得越少权重越大。
    def weighted_match_score(terms: set[str]) -> float:
        return sum(
            1.0 / term_document_frequency.get(term, 1)
            for term in terms
        )

    text_bonus = min(
        weighted_match_score(matched_terms) * 0.08,
        0.6,
    )
    heading_bonus = min(
        weighted_match_score(heading_matches) * 0.5,
        1.5,
    )
    # 最终分 = 原始检索score + 正文命中*0.08(封顶0.6) + 标题命中*0.5(封顶1.5)
    return (
        float(record.get("score", 0.0))
        + text_bonus
        + heading_bonus
    )


def collect_image_groups(
    records: list[dict[str, Any]],
    query: str | None = None,
) -> list[list[str]]:
    """按问题相关性排序相邻图片组，并在组内、组间去重。"""
    # 列表推导式，从所有 records 里挑出 "有 image_ids 且非空" 的。没图的 record 直接扔掉，因为我们要排的是图片候选。
    image_records = [
        record
        for record in records
        if record.get("image_ids")
    ]

    if not image_records:
        return []

    query_terms = _query_terms(query or "")
    # `record_terms`：每条带图 record 的 terms
    record_terms = {
        str(record.get("chunk_id", "")): _query_terms(
            str(record.get("text", ""))
        )
        for record in image_records
    }
    # 每个 query 词出现在多少条 record 里，这就是 df（文档频率）
    term_document_frequency = {
        term: sum(
            term in terms
            for terms in record_terms.values()
        )
        for term in query_terms
    }
    # 贪心分组：`groups`：最终的分组结果，是个列表的列表。每个元素是一个 "组"，组里是若干条 record。
    groups: list[list[dict[str, Any]]] = []
    # 先按 `(source, chunk_position)` 排序 —— 同一本手册按页序排。
    for record in sorted(
        image_records,
        key=lambda item: (
            str(item.get("source", "")),
            _chunk_position(str(item.get("chunk_id", "")))
            if _chunk_position(str(item.get("chunk_id", ""))) is not None
            else 10**9,
        ),
    ):
        # 当前 record 的页码
        position = _chunk_position(str(record.get("chunk_id", "")))
        # 当前 record 的手册名。
        source = record.get("source")

        if groups:
            previous = groups[-1][-1]
            previous_position = _chunk_position(
                str(previous.get("chunk_id", ""))
            )
            image_position = _image_position(record)
            previous_image_position = _image_position(previous)
            next_text_is_short = len(
                str(record.get("text", ""))
            ) <= 120
            first_text_is_short = len(
                str(groups[-1][0].get("text", ""))
            ) <= 180
            # 然后逐条看：当前 record 能不能并入 "上一组的最后一条"？要同时满足：
            if (
                source == previous.get("source")
                and position is not None
                and previous_position is not None
                and position - previous_position == 1
                and image_position is not None
                and previous_image_position is not None
                and image_position - previous_image_position == 1
                and len(groups[-1]) < 3
                and first_text_is_short
                and next_text_is_short
            ):
                groups[-1].append(record)
                continue

        groups.append([record])
    # 每个组的得分 = `组内最高relevance分 + 0.15*(组大小-1)`。多 0.15 的小奖励，让大图组在打平的时候略优先。按这个降序排。
    ranked_groups = sorted(
        groups,
        key=lambda group: (
            max(
                _record_relevance(
                    record,
                    query_terms,
                    term_document_frequency,
                )
                for record in group
            )
            + 0.15 * (len(group) - 1),
            max(float(record.get("score", 0.0)) for record in group),
        ),
        reverse=True,
    )

    image_groups: list[list[str]] = []
    added_image_ids: set[str] = set()
    # 按排好的组顺序把图片 ID 摊平，全局去重。
    for group in ranked_groups:
        group_image_ids: list[str] = []
        for record in group:
            for image_id in record["image_ids"]:
                if image_id in added_image_ids:
                    continue
                group_image_ids.append(image_id)
                added_image_ids.add(image_id)

        if group_image_ids:
            image_groups.append(group_image_ids)

    return image_groups


def collect_image_ids(
    records: list[dict[str, Any]],
    query: str | None = None,
) -> list[str]:
    """把排序后的图片组展开为候选图片 ID 列表。"""
    image_groups = collect_image_groups(
        records=records,
        query=query,
    )
    image_ids = [
        image_id
        for group in image_groups
        for image_id in group
    ]
    return image_ids


def build_context_package(
    records: list[dict[str, Any]],
    max_context_chars: int = 10000,
    max_text_chars: int = 1200,
    query: str | None = None,
) -> dict[str, Any]:
    """把检索结果整理成长度受控的大模型上下文。"""
    # `records`：hybrid_search 返回的全部检索结果，已经按 score 降序排好
    # `max_context_chars=10000`：**整个上下文总字符上限**，防止全部资料加起来太长
    # `max_text_chars=1200`：传给 format_record，单条资料正文最大字符数。
    # 存放每一条格式化后的资料字符串；
    context_blocks=[]
    # 真正被选进上下文的原始 record；
    selected_records=[]
    # 记录当前累计已经使用的字符总数。
    current_length=0
    
    # 循环检索结果，rank 从 1 开始；把单条 record 格式化为完整的资料字符串 block。
    for rank,record in enumerate(records,start=1):
        block=format_record(
            record,
            rank=rank,
            max_text_chars=max_text_chars,
        )
        
        # 分隔符：后面我们会用`"\n\n---\n\n"`分割不同资料块。`---`加上换行一共 7 个字符。
        # - 如果 context_blocks 是空（第一条循环）：不需要分隔符，`separator_length=0`- 如果已经有内容：就要加上分隔符占用的 7 字符。
        separator_length=(
            0 if not context_blocks else 7
        )
        
        # 预计算加入这条 block 之后总字符数。
        next_length=(
            current_length+separator_length+len(block)
        )
        # 如果加上这条之后超过总上限`max_context_chars`，直接`continue`跳过这条记录，不加入上下文。
        # 因为 records 是按分数从高到低排，分数低的直接舍弃，优先保留高分相关资料。
        if next_length>max_context_chars:
            continue
        
        # 没有超限：把 block 存入列表，原始 record 存入选中列表，更新累计字符长度。
        context_blocks.append(block)
        selected_records.append(record)
        current_length=next_length
        
    # 把所有选中的资料块，用分割线`---`拼接成一整段大文本，这一段文本直接放进 LLM prompt 的上下文部分。
    context="\n\n---\n\n".join(context_blocks)
        
    # 对真正选入上下文的记录收集有序去重图片 ID，被截断抛弃的记录的图片不会收集。
    image_groups = collect_image_groups(
        records=selected_records,
        query=query,
    )
    image_ids = [
        image_id
        for group in image_groups
        for image_id in group
    ]
        
    return {
        "context": context,
        "records": selected_records,
        "image_ids": image_ids,
        "image_groups": image_groups,
        "record_count": len(selected_records),
        "context_length": len(context),
    }

if __name__ == "__main__":
    test_question = (
        "我想更换健身追踪器的表带，"
        "有其他尺寸可选吗？"
    )
    
    retrieved_records=hybrid_search(test_question,)
    context_package=build_context_package(retrieved_records,)
    
    print("问题：", test_question)
    print(
        "原始检索记录数：",
        len(retrieved_records),
    )
    print(
        "进入上下文的记录数：",
        context_package["record_count"],
    )
    print(
        "上下文字符数：",
        context_package["context_length"],
    )
    print(
        "上下文包含的图片ID：",
        context_package["image_ids"],
    )
    print("\n========================================")
    print("最终上下文")
    print("========================================\n")
    print(context_package["context"])

    
    
    

    


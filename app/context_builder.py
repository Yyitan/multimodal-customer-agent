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


def collect_image_ids(records:list[dict[str,Any]],)->list[str]:
    """按照检索顺序收集图片ID并去重"""
    # 收集图片 ID，保留检索顺序，同时去重。
    image_ids=[]
    # 集合用来记录已经加过的图片 id，做去重
    added_image_ids=set()
    # 双层循环遍历每一条记录的每一张图片
    for record in records:
        for image_id in record["image_ids"]:
            if image_id in added_image_ids:
                continue
            # 如果图片 id 不在集合，就加入结果列表，同时存入集合标记已经添加
            image_ids.append(image_id)
            added_image_ids.add(image_id)
    return image_ids

def build_context_package(records:list[dict[str,Any]],max_context_chars:int=10000,max_text_chars:int=1200)->dict[str,Any]:
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
    image_ids=collect_image_ids(selected_records)
        
    return {
        "context": context,
        "records": selected_records,
        "image_ids": image_ids,
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

    
    
    

    


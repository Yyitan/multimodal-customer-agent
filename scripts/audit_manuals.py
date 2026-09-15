import json
from pathlib import Path
from typing import Any

# 数据审计脚本
PROJECT_ROOT=Path(__file__).resolve().parent.parent

RAW_DIR=(PROJECT_ROOT.parent/ "赛题三：知识库资料" / "知识库资料")

def audit_manual(file_path:Path,)->dict[str,Any]:
    """检查一本中文手册的JSON结构和图文对应关系。"""
    
    raw_text=file_path.read_text(encoding="utf-8",)
    
    try:
        data=json.loads(raw_text)
    except json.JSONDecodeError as error:
        return {
            "filename":file_path.name,
            "status":"JSON解析失败",
            "error":str(error),
        }
    
    valid_structure=(
        # 表示检查 data 是否为列表。
        isinstance(data,list)
        and len(data)==2
        and isinstance(data[0],str)
        and isinstance(data[1],list)
    )
    
    if not valid_structure:
        return {
            "filename":file_path.name,
            "status":"数据结构不符合预期",
        }
    
    text=data[0]
    image_ids=data[1]
    parts=text.split("<PIC>")
    
    # 三种异常对应关系
    # 空文本片段:表示某次 <PIC> 分割后没有文字。
    # 它不一定是错误。如果连续出现两张图片，中间本来就可能没有文字。
    empty_part_indices=[
        index
        for index,part in enumerate(parts)
        if not part.strip()
    ]
    
    # 有文本但无图片
    # 表示存在正文，但相同索引位置已经没有图片ID。
    text_without_image=[
        index
        for index,part in enumerate(parts)
        if part.strip() and index >= len(image_ids)
    ]
    
    # 有图片但无文本
    # 表示该位置有图片ID，但是相应文本为空。
    # 这类情况不能简单 continue，否则图片可能丢失。后面我们需要把这张图片合并到前一条有文字的知识记录中。
    image_without_text=[
        index
        for index in range(
            min(len(parts),len(image_ids))
        )
        if not parts[index].strip()
    ]
    
    return {
        "filename":file_path.name,
        "status":"结构可读取",
        "pic_count":text.count("PIC"),
        "part_count":len(parts),
        "image_count":len(image_ids),
        "empty_part_indices":empty_part_indices,
        "text_without_image":text_without_image,
        "image_without_text":image_without_text,
    }

def main()->None:
    manual_paths=sorted(
        RAW_DIR.glob("*.txt")
    )
    
    for file_path in manual_paths:
        print("\n====================================")
        print("文件：", file_path.name)
        
        if file_path.name=="汇总英文手册.txt":
            print("状态：暂时跳过")
            print("原因：它是JSONL格式，需要单独解析")
            continue
        
        result=audit_manual(file_path)
        
        print("状态：",result["status"])
        
        if result["status"]!="结构可读取":
            print("错误",result.get("error","未知"))
            continue
        
        print("<PIC>数量：", result["pic_count"])
        print("文本片段数量：", result["part_count"])
        print("图片ID数量：", result["image_count"])
        
        print(
            "空文本片段索引：",
            result["empty_part_indices"],
        )
        print(
            "有文本但无图片：",
            result["text_without_image"],
        )
        print(
            "有图片但无文本：",
            result["image_without_text"],
        )
        
if __name__=="__main__":
    main()
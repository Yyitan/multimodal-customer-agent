import json
import re
from pathlib import Path
from typing import Any

# 项目根目录是/app/retriever.py
# parent 得到app文件夹
# parent.parent 得到项目根目录
# __file__:表示当前Python文件自己的路径
PROJECT_ROOT=Path(__file__).resolve().parent.parent

# 最终得到：项目根目录/data/normalized
DATA_DIR=PROJECT_ROOT/"data"/"normalized"

# list[dict[str, Any]]：列表【里面很多dict【键str值任意】】
def load_records(data_dir:Path=DATA_DIR)->list[dict[str,Any]]:
    """读取normalized文件夹中的所有 JSONL 知识记录"""
    
    records=[]
    
    for file_path in sorted(data_dir.glob("*.jsonl")):
        with file_path.open("r",encoding="utf-8") as file:
            for line in file:
                line=line.strip()
                
                if not line:
                    continue
                record =json.loads(line)
                records.append(record)
    return records

def extract_keywords(query:str)->list[str]:
    """从用户问题中提取产品型号等容易精确匹配的关键词。"""
    
    # 暂时使用正则表达式提取型号
    model_codes=re.findall(
        r"[A-Za-z]+[-_]?\d+[A-Za-z0-9_-]*",
        query,
    )
    
    keywords=[]
    
    for code in model_codes:
        keyword=code.lower()
        
        if keyword not in keywords:
            keywords.append(keyword)
    # 如果没有找到产品型号，暂时使用整个问题进行匹配。
    # 这只是当前关键词检索的临时方案
    if not keywords:
        clean_query=query.strip().lower()
        
        if clean_query:
            keywords.append(clean_query)
    return keywords

def search_records(records:list[dict[str,Any]],query:str,)->list[dict[str,Any]]:
    """查找直接包含关键词的知识记录"""
    
    matches=[]
    keywords=extract_keywords(query)
    
    for record in records:
        search_text=(
            f"{record['product']}"
            f"{record['text']}"
        ).lower()
        
        for keyword in keywords:
            if keyword in search_text:
                matches.append(record)
                # break表示：这条记录只要有一个关键词命中就够了，停止检查剩余关键词。
                break
    return matches

def expand_neighbors(records:list[dict[str,Any]],matches:list[dict[str,Any]],window_size:int=2,)->list[dict[str,Any]]:
    """在直接命中的记录前后扩展相邻知识块"""
    
    expanded_records=[]
    added_chunk_ids=set()
    
    for match in matches:
        match_index=records.index(match)
        
        start_index=max(0,match_index-window_size,)
        end_index=min(len(records),match_index+window_size+1,)
        
        for record in records[start_index:end_index]:
            if record["source"]!=match["source"]:
                continue
            
            chunk_id=record["chunk_id"]
            
            if chunk_id in added_chunk_ids:
                continue
            
            expanded_records.append(record)
            added_chunk_ids.add(chunk_id)
    return expanded_records

# 启动程序时读取一次知识库，避免每次请求都重新打开所有文件
# 缺点：如果重新生成了 JSONL 文件，需要重启 FastAPI 服务，程序才能重新加载数据。
KNOWLEDGE_RECORDS=load_records()

def retrieve(query:str,window_size:int=2,)->list[dict[str,Any]]:
    """对外提供的完整检索入口"""
    
    direct_matches=search_records(KNOWLEDGE_RECORDS,query,)
    expand_matches=expand_neighbors(KNOWLEDGE_RECORDS,direct_matches,window_size,)
    
    return expand_matches

if __name__=="__main__":
    test_question="DCB107充电器的指示灯分别代表什么意思？"
    results=retrieve(test_question)
    
    print("知识库总记录数：",len(KNOWLEDGE_RECORDS))
    print("检索结果数：",len(results))
    
    for record in results:
        print("\n--------------------")
        print("chunk_id:",record["chunk_id"])
        print("image_ids:",record["image_ids"])
        print("source:",record["source"])
        
        
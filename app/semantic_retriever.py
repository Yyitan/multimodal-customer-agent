import json 
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"

EMBEDDINGS_PATH = INDEX_DIR / "knowledge_embeddings.npy"
RECORDS_PATH = INDEX_DIR / "knowledge_records.jsonl"
META_PATH = INDEX_DIR / "index_meta.json"

QUERY_PREFIX="为这个句子生成表示以用于检索相关文章："

def load_records()->list[dict[str,Any]]:
    """读取与向量顺序一致的知识记录"""
    records=[]
    
    with RECORDS_PATH.open("r",encoding="utf-8") as file:
        for line in file:
            line =line.strip()
            if not line:
                 continue
            records.append(json.loads(line))
            
    return records

def load_metadata()->dict[str,Any]:
    """读取向量索引说明"""
    with META_PATH.open("r",encoding="utf-8") as file:
        return json.load(file)

def validate_index(
    records:list[dict[str,Any]],
    embeddings:np.ndarray,
    metadata:dict[str,Any],
)->None:
    """检查记录、向量和说明文件是否一致"""
    if embeddings.ndim!=2:
        raise ValueError("知识向量必须是二维矩阵。")
    
    if len(records)!=embeddings.shape[0]:
        raise ValueError(
            "知识记录数与向量行数不一致："
            f"{len(records)}!={embeddings.shape[0]}"
        )
    if metadata["record_count"]!=len(records):
        raise ValueError(
            "索引说明中的记录数与实际记录数不一致。"
        )
    
    if metadata["embedding_dimension"]!=embeddings.shape[1]:
        raise ValueError(
            "索引说明中的向量维度与实际维度不一致。"
        )
    
METADATA=load_metadata()
KNOWLEDGE_RECORDS=load_records()

KNOWLEDGE_EMBEDDINGS=np.load(
    EMBEDDINGS_PATH,
    allow_pickle=False,
)

validate_index(
    KNOWLEDGE_RECORDS,
    KNOWLEDGE_EMBEDDINGS,
    METADATA,
)

MODEL=SentenceTransformer(
    METADATA["model_name"],
)

def semantic_search(
    query:str,
    top_k:int=5,
)->list[dict[str,Any]]:
    """根据语义相似度返回最相关的Top K条知识。"""
    clean_query=query.strip()
    if not clean_query:
        return []
    
    query_text=QUERY_PREFIX+clean_query
    
    query_vector=MODEL.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    
    similarity_scores=(
        KNOWLEDGE_EMBEDDINGS @ query_vector
    )
    
    result_count=min(
        top_k,
        len(KNOWLEDGE_RECORDS),
    )
    
    # 返回按分数从小到大排列的索引
    top_indices=np.argsort(
        similarity_scores,
    )[::-1][:result_count] 
    # [::-1]:把顺序反转，变成从大到小 [:result_count]:只取前 Top K 个
    
    results=[]
    
    # .copy() 会复制一个新字典，然后只给返回结果添加分数，不污染原始数据。
    for index in top_indices:
        record=KNOWLEDGE_RECORDS[int(index)].copy()
        record["score"]=float(
            similarity_scores[index]
        )
        
        results.append(record)
    
    return results

if __name__=="__main__":
    test_questions = [
        "电钻电池充满以后，充电器的灯会显示什么状态？",
        "电钻电池温度太高时为什么不能马上充电？",
        "健身追踪器有其他尺寸的表带可以更换吗？",
    ]
    
    for question in test_questions:
        print("\n========================================")
        print("问题：", question)
        
        results=semantic_search(question,top_k=5,)
        
        for rank,record in enumerate(results,start=1):
            preview=record["text"][:120].replace(
                "\n",
                " ",
            )
            
            print(f"\n第 {rank} 名")
            print(f"score：{record['score']:.4f}")
            print("chunk_id：", record["chunk_id"])
            print("product：", record["product"])
            print("text：", preview)
            print("image_ids：", record["image_ids"])
            print("source：", record["source"])
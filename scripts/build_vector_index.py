import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME="BAAI/bge-small-zh-v1.5"

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"/"normalized"
INDEX_DIR=PROJECT_ROOT/"data"/"index"

# 索引与元数据对齐
EMBEDDINGS_PATH=INDEX_DIR/"knowledge_embeddings.npy"
RECORDS_PATH=INDEX_DIR/"knowledge_records.jsonl"
META_PATH=INDEX_DIR/"index_meta.json"

def load_records()->list[dict[str,Any]]:
    """读取所有标准化知识记录"""
    
    records=[]
    
    for file_path in sorted(DATA_DIR.glob("*.jsonl")):
        with file_path.open("r",encoding="utf-8") as file:
            for line in file:
                if not line:
                    continue
                record=json.loads(line)
                records.append(record)
                
    return records

def bulid_search_text(record:dict[str,Any])->str:
    """构造真正交给Embedding模型的知识文本"""
    
    return (
        f"产品：{record['product']}\n"
        f"内容：{record['text']}"
    )
    
def save_records(records:list[dict[str,Any]])->None:
    """按照向量顺序保存知识记录快照"""
    with RECORDS_PATH.open("w",encoding="utf-8") as file:
        for record in records:
            line=json.dumps(
                record,
                ensure_ascii=False,
            )
            file.write(line+"\n")
            
def main()->None:
    INDEX_DIR.mkdir(parents=True,exist_ok=True,)
    
    records=load_records()
    
    if not records:
        raise ValueError("没有找到可以构建索引的知识记录。")
    
    search_texts=[
        bulid_search_text(record)
        for record in records
    ]
    
    print("准备记载模型：",MODEL_NAME)
    model=SentenceTransformer(MODEL_NAME)
    
    print("知识记录数：",len(records))
    print("开始生成知识向量......")
    
    embeddings=model.encode(
        # 要转换为向量的143条文本。
        search_texts,
        # 不是一次把143条全部送进模型，而是分批处理：
        batch_size=32,
        # 在终端显示进度条
        show_progress_bar=True,
        # 把每个向量的长度归一化为1
        normalize_embeddings=True,
        # 让返回结果成为 NumPy 数组，便于保存和计算
        convert_to_numpy=True,
    )
    
    # 表示每个数字使用32位存储，精度足够，而且占用空间较小
    embeddings=embeddings.astype(np.float32)
    
    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )
    
    save_records(records)
    
    metadata={
        "model_name":MODEL_NAME,
        "record_count":len(records),
        "embedding_dimension":int(embeddings.shape[1]),
        "normalized":True,
        "created_at":datetime.now(timezone.utc).isoformat(),
    }
    
    with META_PATH.open("w",encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )
    
    print("\n向量索引构建完成")
    print("向量形状：",embeddings.shape)
    print("向量文件：", EMBEDDINGS_PATH)
    print("记录文件：", RECORDS_PATH)
    print("说明文件：", META_PATH)
    
if __name__=="__main__":
    main()
    
    
import json
from pathlib import Path

DATA_DIR=Path("data/normalized")
QUERY="DCB107"

def load_records(data_dir):
    records=[]
    
    # json.load(file)：读取一个文件对象。
    # json.loads(line)：读取一个 JSON 字符串。
    
    for file_path in data_dir.glob("*.jsonl"):
        with file_path.open("r",encoding="utf-8") as file:
            for line in file:
                line=line.strip()
                
                if not line:
                    continue
                record=json.loads(line)
                records.append(record)
                
    return records

def search_records(records,query):
    matches=[]
    # lower():它把英文统一转换成小写，使检索不区分大小写。
    keyword=query.lower()
    
    for record in records:
        text=record["text"].lower()
        
        if keyword in text:
            matches.append(record)
            
    return matches

def expand_neighbors(records,matches,window_size=2):
    expanded_records=[]
    # set() 是集合，特点是元素不能重复
    added_chunk_ids=set()
    
    for match in matches:
        match_index=records.index(match)
        
        start_index=max(0,match_index-window_size)
        end_index=min(len(records),match_index+window_size+1)
        
        for record in records[start_index:end_index]:
            # 确保记录都来自同一手册  
            if record["source"]!=match["source"]:
                continue
            chunk_id =record["chunk_id"]
            
            # 多个命中结果的扩展范围可能重叠，所以防止重复添加
            if chunk_id in added_chunk_ids:
                continue
            expanded_records.append(record)
            added_chunk_ids.add(chunk_id)
    return expanded_records
            
records=load_records(DATA_DIR)
direct_matches=search_records(records,QUERY)
expanded_matches=expand_neighbors(
    records,
    direct_matches,
    window_size=2,
)

print(("知识库总记录数：",len(records)))
print("直接命中记录数：", len(direct_matches))
print("扩展后记录数：", len(expanded_matches))

for record in expanded_matches:
    # 为什么只展示前160个字符:有些知识块非常长
    preview=record["text"][:160].replace("\n"," ")
    
    print("\n--------------------")
    print("chunk_id:",record["chunk_id"])
    print("product:",record["product"])
    print("text:",preview)
    print("image_ids",record["image_ids"])
    print("source:",record["source"])


    
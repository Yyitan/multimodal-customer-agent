import json
from pathlib import Path

# 把一份原始的正文+图片ID列表的手册，转换成多条结构统一的小记录，并写入新文件。
'''
原始手册[整篇正文（含<PIC>），图片ID列表]→多条 record→normalized_fitness_tracker.jsonl
'''

def build_chunk_id(product_key,index):
    return f"{product_key}_{index:04d}"

def normalize_manual(input_path,product,product_key):
    with input_path.open("r",encoding="utf-8") as file:
        data=json.load(file)
        
    text=data[0]
    image_ids=data[1]
    parts=text.split("<PIC>")

    # 创建一个空列表，用来收集所有转换好的知识记录
    records=[]

    # strip() 用来删除一段文本开头和结尾多余的空格、换行符、制表符
    for index,part in enumerate(parts):
        clean_text=part.strip()

        # 跳过空文本
        if not clean_text:
            continue

        # {index:04d}：index 是整数，至少保留 4 位，前面不够的地方补 0
        record={
            "chunk_id":build_chunk_id(product_key,index),
            "product":product,
            "text":clean_text,
            "image_ids":image_ids[index:index+1],
            "source":input_path.name,
        }
        records.append(record)
        
    return records

raw_dir=Path(r"..\赛题三：知识库资料\知识库资料")
output_dir=Path("data/normalized")

manual_configs=[
    {
        "filename":"健身追踪器手册.txt",
        "product":"健身追踪器",
        "product_key":"fitness_tracker",
    },
        {
        "filename":"冰箱手册.txt",
        "product":"冰箱",
        "product_key":"fridge",
    },
            {
        "filename":"电钻手册.txt",
        "product":"电钻",
        "product_key":"drill",
    },
]

for config in manual_configs:
    input_path=raw_dir/config["filename"]
    output_path=(
        output_dir/f"normalized_{config['product_key']}.jsonl"
    )
    
    records=normalize_manual(
        input_path=input_path,
        product=config["product"],
        product_key=config["product_key"],
    )
    
    with output_path.open("w",encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record,ensure_ascii=False)+"\n"
            )
            
    print(
        f"{config['product']}：生成 {len(records)} 条记录，"
        f"输出到 {output_path}"
    )

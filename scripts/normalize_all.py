import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = (
    PROJECT_ROOT.parent
    / "赛题三：知识库资料"
    / "知识库资料"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "normalized"


MANUAL_CONFIGS = [
    {
        "filename": "VR头显手册.txt",
        "product": "VR头显",
        "product_key": "vr_headset",
    },
    {
        "filename": "人体工学椅手册.txt",
        "product": "人体工学椅",
        "product_key": "ergonomic_chair",
    },
    {
        "filename": "健身单车手册.txt",
        "product": "健身单车",
        "product_key": "exercise_bike",
    },
    {
        "filename": "健身追踪器手册.txt",
        "product": "健身追踪器",
        "product_key": "fitness_tracker",
    },
    {
        "filename": "儿童电动摩托车手册.txt",
        "product": "儿童电动摩托车",
        "product_key": "kids_electric_motorcycle",
    },
    {
        "filename": "冰箱手册.txt",
        "product": "冰箱",
        "product_key": "fridge",
    },
    {
        "filename": "功能键盘手册.txt",
        "product": "功能键盘",
        "product_key": "function_keyboard",
    },
    {
        "filename": "发电机手册.txt",
        "product": "发电机",
        "product_key": "generator",
    },
    {
        "filename": "可编程温控器手册.txt",
        "product": "可编程温控器",
        "product_key": "programmable_thermostat",
    },
    {
        "filename": "吹风机手册.txt",
        "product": "吹风机",
        "product_key": "hair_dryer",
    },
    {
        "filename": "摩托艇手册.txt",
        "product": "摩托艇",
        "product_key": "personal_watercraft",
    },
    {
        "filename": "水泵手册.txt",
        "product": "水泵",
        "product_key": "water_pump",
    },
    {
        "filename": "洗碗机手册.txt",
        "product": "洗碗机",
        "product_key": "dishwasher",
    },
    {
        "filename": "烤箱手册.txt",
        "product": "烤箱",
        "product_key": "oven",
    },
    {
        "filename": "电钻手册.txt",
        "product": "电钻",
        "product_key": "drill",
    },
    {
        "filename": "相机手册.txt",
        "product": "相机",
        "product_key": "camera",
    },
    {
        "filename": "空气净化器手册.txt",
        "product": "空气净化器",
        "product_key": "air_purifier",
    },
    {
        "filename": "空调手册.txt",
        "product": "空调",
        "product_key": "air_conditioner",
    },
    {
        "filename": "蒸汽清洁机手册.txt",
        "product": "蒸汽清洁机",
        "product_key": "steam_cleaner",
    },
    {
        "filename": "蓝牙激光鼠标手册.txt",
        "product": "蓝牙激光鼠标",
        "product_key": "bluetooth_laser_mouse",
    },
]

def build_chunk_id(product_key:str,index:int,)->str:
    return f"{product_key}_{index:04d}"

def load_manual(input_path:Path,)->tuple[list[Any],bool]:
    # 函数返回两个值：第一个：读取出的手册数据；第二个：是否修复过非法JSON
    """读取手册，并有限修复已知的非法JSON转义。"""
    raw_text=input_path.read_text(encoding="utf-8")
    repaired=False
    
    try:
        data=json.loads(raw_text)
    except json.JSONDecodeError:
        # JSON不允许\*这种转义。
        # 当前洗碗机手册中存在该问题。
        repaired_text=raw_text.replace(
            r"\*",
            "*",
        )
        
        # 如果没有找到已知问题，就不能盲目修复其他错误。
        if repaired_text==raw_text:
            raise
        
        data=json.loads(repaired_text)
        repaired=True
        
    valid_structure = (
        isinstance(data, list)
        and len(data) == 2
        and isinstance(data[0], str)
        and isinstance(data[1], list)
    )

    if not valid_structure:
        raise ValueError(
            f"{input_path.name}的数据结构不符合预期。"
        )
        
    if not all(
        isinstance(image_id,str)
        for image_id in data[1]
    ):
        raise ValueError(
            f"{input_path.name}的图片ID中存在非字符串数据。"
        )
    return data,repaired

def normalize_manual(input_path:Path,product:str,product_key:str,)->tuple[list[dict[str,Any]],bool]:
    """将一本手册转换为图文对应的知识记录"""
    # 这种一次返回多个值的方式叫做元组返回或者解包
    data,repaired=load_manual(input_path)
    
    text=data[0]
    image_ids=data[1]
    parts=text.split("<PIC>")
    
    records=[]
    
    # 保存出现在第一段正文之前的图片。
    pending_images=[]
    
    for index,part in enumerate(parts):
        clean_text=part.strip()
        
        current_image=None
        
        if index <len(image_ids):
            # 取出当前位置对应的图片ID
            current_image=image_ids[index]
        
        # 当前切片文本为空字符串  
        if not clean_text:
            if current_image is not None:
                # 已经又生成过记录records不为空，把图片追加到上一条切片的image_ids
                if records:
                    # 中间出现的无文本图片合并到前一条记录。
                    records[-1]["image_ids"].append(
                        current_image
                    )
                # 还没有任何记录
                else:
                    # 说明时文件最开头的图片，pending_images暂存图片，
                    # 等遇到第一条正文时再合并
                    pending_images.append(
                        current_image
                    )
            # 跳过本轮，不生成新记录。
            continue
        
        # 如果文本不为空，正式生成记录
        record_images=pending_images
        pending_images=[]
        
        # 把当前`current_image`加入图片列表
        if current_image is not None:
            record_images.append(current_image)
            
        record={
            "chunk_id":build_chunk_id(product_key,index,),
            "product":product,
            "text":clean_text,
            "image_ids":record_images,
            "source":input_path.name
        }
        
        records.append(record)
        
    # 如果图片ID数量比文本片段还多
    # 将剩余图片合并到最后一条文本记录
    remaining_images=image_ids[len(parts):]
    
    if remaining_images:
        if not records:
            raise ValueError(
                f"{input_path.name}只有图片，没有可关联的正文。"
            )
            
        records[-1]["image_ids"].extend(
            remaining_images
        )
    
    # 验证标准化前后图片数量及顺序完全一致
    # 即比较数量也比较质量
    normalized_image_ids=[
        image_id
        for record in records
        for image_id in record["image_ids"]
    ]
    
    if normalized_image_ids!=image_ids:
        raise ValueError(
            f"{input_path.name}标准化后图片ID发生丢失或错序"
        )
    return records,repaired

def save_records(
    records:list[dict[str,Any]],
    output_path:Path,
)->None:
    """将知识记录保存成JSONL文件。"""
    with output_path.open("w",encoding="utf-8") as file:
        for record in records:
            line=json.dumps(record,ensure_ascii=False,)
            
            file.write(line+"\n")

def main()->None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    total_records=0
    total_images=0
    
    # 遍历`MANUAL_CONFIGS`中 20 个产品配置
    for config in MANUAL_CONFIGS:
        input_path=(
            RAW_DIR/config["filename"]
        )
        
        output_path=(
            OUTPUT_DIR
            /f"normalized_{config['product_key']}.jsonl"
        )
        
        records,repaired=normalize_manual(
            input_path=input_path,
            product=config["product"],
            product_key=config["product_key"],
        )
        
        save_records(records,output_path,)
        
        record_image_count=sum(
            len(record["image_ids"])
            for record in records
        )
        
        total_records+=len(records)
        total_images+=record_image_count
        
        repair_message=(
            "，已修复非法JSON转义"
            if repaired
            else ""
        )
        
        print(
            f"{config['product']}:"
            f"{len(records)}条记录，"
            f"{record_image_count}张图片"
            f"{repair_message}"
        )
    print("\n全部中文手册处理完成")
    print("手册数量：",len(MANUAL_CONFIGS))
    print("知识记录总数：",total_records)
    print("图片ID总数：",total_images)
    
if __name__=="__main__":
    main()
    
    
        
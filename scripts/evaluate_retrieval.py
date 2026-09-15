import json
from pathlib import Path
from typing import Any

from app.hybrid_retriever import hybrid_search

from app.context_builder import build_context_package

PROJECT_ROOT=Path(__file__).resolve().parent.parent
EVALUATION_PATH=(
    PROJECT_ROOT/"data"/"evaluation"/"retrieval_cases.json"
)

def load_cases()->list[dict[str,Any]]:
    """读取评测案例。"""
    
    with EVALUATION_PATH.open("r",encoding="utf-8") as file:
        # `json.load(文件对象)`：直接读取打开的文件，解析成 Python 对象
        return json.load(file)
    

def collect_images_ranks(results:list[dict[str,Any]],)->dict[str,int]:
    """记录每个图片ID第一次出现的排名。"""
    # 空字典，key = 图片 id 字符串；value = 这张图片第一次出现的排名数字
    image_ranks={}
    # 排名 rank从 1 开始计数，不是 0。
    # rank：当前这条检索结果的名次；record：一条知识切片。
    for rank,record in enumerate(results,start=1):
        for image_id in record["image_ids"]:
            if image_id not in image_ranks:
                # 只记录第一次出现
                # 如果图片 id 还不在字典，就存入当前 rank；如果已经存在，直接跳过。
                image_ranks[image_id]=rank
    
    # 返回字典：所有出现过的图片以及它首次排名。
    return image_ranks

def evaluate_case(case:dict[str,Any],max_results:int=20)->dict[str,Any]:
    """执行单个评测案例"""
    # 输入一条 case 字典，调用混合检索函数，传入测试问题，拿到检索返回结果。
    results=hybrid_search(case["question"],max_results=max_results)
    # 送入`build_context_package()`做上下文组装：**做总字符截断过滤**，只保留一部分高分记录得到`selected_records`
    context_package=build_context_package(results)
    selected_records=context_package["records"]
    
    # 检查上下文中的所有记录是否属于预期产品
    # `case` 是评测 json 里的单条测试用例;- `.get()` 的好处：如果评测案例里没有写`expected_product`这个 key，不会报错，直接赋值`None`
    expected_product=case.get("expected_product")
    # selected_records 是真正会被送给大模型的资料；如果其中有一条记录的 product 不匹配，就记录下来
    wrong_product_records=[
        record
        # 遍历送入大模型的全部知识块`selected_records`：条件同时满足才会把这条 record 放进错误列表：
        for record in selected_records
        # `expected_product`不为空
        if expected_product 
        # 当前这条知识块的`product`字段 和预期产品不一致
        and record["product"]!=expected_product
    ]
    
    product_consistent=len(wrong_product_records)==0
    
    # 调用上面函数，得到本次检索所有图片的首次排名映射字典。
    image_ranks=collect_images_ranks(selected_records)
    # `set()`：集合，Python 内置容器。集合可以做交、差集运算；集合元素不能重复。
    # 把期待应该被召回的图片 ID 列表转为集合。
    expected_image_ids=set(case["expected_image_ids"])
    # `image_ranks`字典的 key 就是本次检索实际返回的全部图片 id，转为集合。
    returned_image_ids=set(image_ranks)
    # 集合求交集：既在期望集合、又真实被检索返回的图片 → 找出来的正确图片。
    found_image_ids=(expected_image_ids & returned_image_ids)
    # 求差集：期望有，但是检索结果里面完全没出现的图片 → 漏召回图片
    missing_image_ids=(expected_image_ids-returned_image_ids)
    # 遍历每一张成功找到的图片，取出它首次出现的排名，组成排名列表。
    found_ranks=[
        image_ranks[image_id]
        for image_id in found_image_ids
    ]
    
    # 如果`found_ranks`不为空（至少找到一张目标图片），`min()`取最小的排名（也就是**目标图片中位置最靠前的那一张的名次**）
    # 如果一张都没找到，`found_ranks`是空列表 → `first_rank = None`
    first_rank=(
        min(found_ranks)
        if found_ranks
        else None
    )
    
    return {
        "case_id":case["case_id"],
        "question":case["question"],
        "expected_count":len(expected_image_ids),
        "found_count":len(found_image_ids),
        "missing_image_ids":sorted((missing_image_ids)),
        "all_found":(found_image_ids==expected_image_ids),
        "first_rank":first_rank,
        "result_count":len(results),
        "context_length":context_package["context_length"],
        "expected_product":expected_product,
        "product_consistent":product_consistent,
        "wrong_product_chunk_ids":[
            record["chunk_id"]
            for record in wrong_product_records
        ],
        #必须满足：1、图片全找到 2、没有混入别的产品资料
        "passed":(
            found_image_ids==expected_image_ids
            and product_consistent
        ),
    }

def main()->None:
    cases=load_cases()
    evaluation_results=[]
    
    for case in cases:
        result=evaluate_case(case)
        evaluation_results.append(result)
        
        print("\n========================================")
        print("案例：",result["case_id"])
        print("问题：",result["question"])
        print(
            "找到图片：",
            f"{result['found_count']}/"
            f"{result['expected_count']}"
        )
        print("是否全部找到：",result["all_found"],)
        
        print("产品范围正确：",result["product_consistent"],)
        
        print("错误产品记录：",result["wrong_product_chunk_ids"],)
        
        print("案例是否通过：",result["passed"],)
        
        print("第一张目标图片排名：",result["first_rank"],)
        
        print("返回记录数：",result["result_count"],)
        
        print("上下文字符数：",result["context_length"],)
        
    passed_count=sum(
        result["passed"]
        for result in evaluation_results
    )
    
    print("\n========================================")
    print("评测总结")
    print("案例总数：",len(cases))
    print("全部图片都找到的案例：", passed_count)
    print(
        "案例通过率：",
        f"{passed_count / len(cases):.2%}",
    )

if __name__ == "__main__":
    main() 
     
            
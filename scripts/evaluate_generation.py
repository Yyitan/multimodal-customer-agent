import json
from pathlib import Path
from typing import Any

from app.services.chat_service import answer_question

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "generation_cases.json"


# 加载 json 里所有评测用例，返回案例列表。
def load_cases() -> list[dict[str, Any]]:
    """读取端到端生成评测案例。"""
    with CASES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# 关键词校验函数
def check_keyword_groups(
    answer: str,
    keyword_groups: list[list[str]],
) -> list[list[str]]:
    """返回没有命中的关键词组。"""
    # 把模型回答全部转为小写，实现大小不敏感匹配
    answer_lower = answer.lower()
    missing_groups: list[list[str]] = []
    # 遍历每一组关键词
    for group in keyword_groups:
        # any()代表组内只要任意一个关键词存在，这一组就通过。
        matched = any(keyword.lower() in answer_lower for keyword in group)
        # 所有没有命中的组 就收集起来返回,代表回答缺少这一组信息
        if not matched:
            missing_groups.append(group)
    return missing_groups


def evaluate_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    """执行一条真实模型端到端评测。"""
    service_result = answer_question(case["question"])
    answer = service_result["answer"]
    # 把返回图片 ID、预期图片 ID 都转成`set集合`，方便做集合差集运算
    returned_image_ids = set(service_result["image_ids"])
    expected_image_ids = set(case["expected_image_ids"])

    # 调用关键词检查，拿到没命中的关键词组
    missing_keyword_groups = check_keyword_groups(
        answer=answer,
        keyword_groups=case["must_include_any"],
    )
    # 集合减法，算出**预期图片里面哪些没有被返回**；sorted 排序方便打印查看
    missing_image_ids = sorted(expected_image_ids - returned_image_ids)

    # 判断是否通过：
    # `keyword_passed`：没有缺失关键词组 → 关键词校验通过
    keyword_passed = not missing_keyword_groups
    # `image_passed`：没有缺失图片 ID → 图片校验通过
    image_passed = not missing_image_ids
    # 两项全部满足，这个案例才算整体通过
    overall_passed=keyword_passed and image_passed

    # 额外记录：本次 RAG 召回了多少参考文档、返回图片数量，方便定位失败原因。
    return {
        "case_id":case["case_id"],
        "question":case["question"],
        "answer":answer,
        "keyword_passed":keyword_passed,
        "missing_keyword_groups":missing_keyword_groups,
        "image_passed":image_passed,
        "missing_image_ids":missing_image_ids,
        "overall_passed":overall_passed,
        "reference_count":len(service_result["references"]),
        "returned_image_count":len(returned_image_ids),
    }

def main()->None:
    # 加载全部测试用例
    cases=load_cases()
    results:list[dict[str,Any]]=[]
    # 循环遍历每一条 case，逐个跑 RAG 评测，保存每条结果
    for case in cases:
        result=evaluate_case(case)
        results.append(result)
        
        # 先换行，然后打印一整条由 60 个`=`组成的分割线
        print("\n"+"="*60)
        print("案例:",result["case_id"])
        print("问题:",result["question"])
        print("\n模型回答:")
        print(result["answer"])
        print("\n关键词检查:")
        print("通过:",result["keyword_passed"])
        if result["missing_keyword_groups"]:
            print(
                "缺少关键词组:",
                result["missing_keyword_groups"],
            )
        print("\n图片检查:")
        print("通过:",result["image_passed"])
        
        if result["missing_image_ids"]:
            print(
                "缺少图片:",
                result["missing_image_ids"],
            )
        print("\n参考资料数量:",result["reference_count"])
        print(
            "返回图片数量:",
            result["returned_image_count"],
        )
        print("本案例是否通过:",result["overall_passed"])
    
    # 循环跑完所有案例之后,统计整体指标
    # `sum(result["overall_passed"]`：布尔值`True=1,False=0`，直接求和算出通过数量
    passed_count=sum(
        result["overall_passed"]
        for result in results
    )
    
    # 最后输出总案例数、通过数、整体通过率。
    print("\n"+"="*60)
    print("生成评估总结")
    print("案例总数:",len(results))
    print("通过案例数:",passed_count)
    print(
        "通过率:",
        # 通过数量 ÷ 总案例:转成百分比，保留 2 位小数
        f"{passed_count/len(results):.2%}",
    )

if __name__ == "__main__":
    main()

        
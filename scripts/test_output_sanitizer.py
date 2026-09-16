import json

from app.llm_client import parse_structured_output


def main() -> None:
    raw_answer = (
        "找到卡扣（参考资料10）；"
        "安装表带（参考资料9、资料10）。"
        "参考资料3说明表带应舒适贴合。"
        "手册第14页需要保留。"
        "推荐参考图片："
        "Manual16_9（拆卸表带）、"
        "Manual16_11（安装表带）。"
        "相关操作示意图请参考："
        "Manual16_9、Manual16_11。"
        "如需参考具体操作示意图，"
        "请查看相关说明图片。"
    )

    raw_output = json.dumps(
        {
            "answer": raw_answer,
            "image_ids": [
                "Manual16_9",
                "Manual16_11",
            ],
        },
        ensure_ascii=False,
    )

    clean_answer, selected_image_ids = (
        parse_structured_output(
            raw_output=raw_output,
            candidate_image_ids=[
                "Manual16_9",
                "Manual16_11",
            ],
            max_images=3,
        )
    )

    expected_answer = (
        "找到卡扣；"
        "安装表带。"
        "知识库资料说明表带应舒适贴合。"
        "手册第14页需要保留。"
    )

    print("原始回答：")
    print(raw_answer)

    print("\n清洗结果：")
    print(clean_answer)

    print("\n保留的图片ID：")
    print(selected_image_ids)

    checks = {
        "删除资料编号": (
            "参考资料" not in clean_answer
            and "资料10" not in clean_answer
        ),
        "删除图片推荐文字": (
            "推荐参考图片" not in clean_answer
            and "相关操作示意图请参考"
            not in clean_answer
            and "相关示意图" not in clean_answer
            and "如需参考具体操作示意图"
            not in clean_answer
        ),
        "保留可理解的页码": (
            "手册第14页" in clean_answer
        ),
        "回答清洗结果正确": (
            clean_answer == expected_answer
        ),
        "结构化图片ID仍然保留": (
            selected_image_ids
            == ["Manual16_9", "Manual16_11"]
        ),
    }

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n回答清洗测试：全部通过")


if __name__ == "__main__":
    main()
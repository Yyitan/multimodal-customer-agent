from app.llm_client import sanitize_answer

def main() -> None:
    raw_answer = (
        "找到卡扣（参考资料10）；"
        "安装表带（参考资料9、资料10）。"
        "参考资料3说明表带应舒适贴合。"
        "手册第14页需要保留。"
    )

    clean_answer = sanitize_answer(
        answer=raw_answer,
        candidate_image_ids=[],
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

    checks = {
        "删除单个资料编号": (
            "参考资料10" not in clean_answer
        ),
        "删除多个资料编号": (
            "资料9" not in clean_answer
            and "资料10" not in clean_answer
        ),
        "保留可理解的页码": (
            "手册第14页" in clean_answer
        ),
        "清洗结果完全符合预期": (
            clean_answer == expected_answer
        ),
    }

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n回答清洗测试：全部通过")


if __name__ == "__main__":
    main()
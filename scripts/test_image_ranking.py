from app.context_builder import (
    collect_image_groups,
    collect_image_ids,
)
from app.llm_client import select_ranked_image_group


def make_record(
    chunk_id: str,
    text: str,
    image_id: str,
    score: float,
    source: str = "test_manual.txt",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "product": "测试产品",
        "text": text,
        "image_ids": [image_id],
        "source": source,
        "score": score,
        "retrieval_type": "semantic",
    }


def main() -> None:
    band_records = [
        make_record(
            "fitness_0007",
            "# 佩戴设备 日常佩戴和更换说明。",
            "Manual16_4",
            0.75,
        ),
        make_record(
            "fitness_0055",
            "# 表带尺寸 表带尺寸如下所示。",
            "Manual16_51",
            0.67,
        ),
        make_record(
            "fitness_0056",
            "环境条件",
            "Manual16_52",
            0.62,
        ),
    ]
    band_images = collect_image_ids(
        band_records,
        query="表带还有其他尺寸可选吗？",
    )

    indicator_records = [
        make_record(
            "drill_0006",
            "状态说明",
            "drill0_11",
            0.59,
        ),
        make_record(
            "drill_0007",
            "状态说明",
            "drill0_12",
            0.60,
        ),
        make_record(
            "drill_0008",
            "# DCB107、DCB112 电池组充电中",
            "drill0_04",
            1.62,
        ),
        make_record(
            "drill_0009",
            "电池组已充满",
            "drill0_05",
            0.56,
        ),
        make_record(
            "drill_0010",
            "过热/过冷延迟",
            "drill0_06",
            0.57,
        ),
    ]
    indicator_images = collect_image_ids(
        indicator_records,
        query="DCB107或DCB112的指示灯含义是什么？",
    )
    indicator_groups = collect_image_groups(
        indicator_records,
        query="DCB107或DCB112的指示灯含义是什么？",
    )
    expanded_band_images = select_ranked_image_group(
        selected_image_ids=["Manual16_51"],
        candidate_image_groups=[
            ["Manual16_51", "Manual16_52"],
        ],
        max_images=3,
    )
    expanded_indicator_images = select_ranked_image_group(
        selected_image_ids=["drill0_18"],
        candidate_image_groups=indicator_groups,
        max_images=3,
    )

    checks = {
        "表带尺寸图片组优先": (
            band_images[:2]
            == ["Manual16_51", "Manual16_52"]
        ),
        "图片编号跳变时切断旧分组": (
            indicator_images[:3]
            == ["drill0_04", "drill0_05", "drill0_06"]
        ),
        "模型需要图片时返回最高相关图片组": (
            expanded_band_images
            == ["Manual16_51", "Manual16_52"]
        ),
        "模型误选次要图片时仍返回首选组": (
            expanded_indicator_images
            == ["drill0_04", "drill0_05", "drill0_06"]
        ),
    }

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n图片候选排序测试：全部通过")


if __name__ == "__main__":
    main()

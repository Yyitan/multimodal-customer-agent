import base64
import json
from pathlib import Path

from app.config import get_settings
from app.vision_analyzer import (
    analyze_images_for_retrieval,
    build_multimodal_retrieval_query,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT.parent
    / "赛题三：知识库资料"
    / "知识库资料"
    / "插图"
    / "Manual16_0.jpg"
)


def load_image_as_data_url(image_path: Path) -> str:
    """把本地图片转换成 Base64 Data URL。"""

    raw_bytes = image_path.read_bytes()
    encoded_text = base64.b64encode(raw_bytes).decode("ascii")

    return "data:image/jpeg;base64," + encoded_text


def main() -> None:
    if not IMAGE_PATH.exists():
        raise SystemExit(f"找不到测试图片：{IMAGE_PATH}")

    settings = get_settings()
    image_data_url = load_image_as_data_url(IMAGE_PATH)

    question = "请看我上传的这张图片。" "我想给这个设备更换配件，应该怎么操作？"

    print("开始视觉预分析……")

    analysis = analyze_images_for_retrieval(
        question=question,
        images=[image_data_url],
        settings=settings,
    )

    print("\n====视觉分析结果====")
    print(
        json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2,
        )
    )

    retrieval_query = build_multimodal_retrieval_query(
        question=question,
        analysis=analysis,
    )

    print("\n====增强后的检索问题====")
    print(retrieval_query)

    checks = {
        "识别结果是字典": isinstance(analysis, dict),
        "产品线索非空": bool(analysis.get("product_hint")),
        "关键词是列表": isinstance(
            analysis.get("visual_keywords"),
            list,
        ),
        "产品置信度是数字": (
            isinstance(
                analysis.get("product_confidence"),
                (int, float),
            )
            and not isinstance(
                analysis.get("product_confidence"),
                bool,
            )
        ),
        "产品置信度范围正确": (
            0.0
            <= analysis.get(
                "product_confidence",
                -1.0,
            )
            <= 1.0
        ),
        "已识别产品具有正置信度": (
            analysis.get(
                "product_confidence",
                0.0,
            )
            > 0.0
        ),
        "图片描述非空": bool(analysis.get("visual_description")),
        "增强问题包含原问题": (question in retrieval_query),
        "视觉模型调用有耗时": (analysis.get("latency_ms", 0) > 0),
    }

    print("\n====验收结果====")
    all_passed = True

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

        if not passed:
            all_passed = False
    if not all_passed:
        raise SystemExit(1)
    print("\n视觉预分析测试：全部通过")


if __name__ == "__main__":
    main()

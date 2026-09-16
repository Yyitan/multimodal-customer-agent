from pathlib import Path

import base64

# Python内置正则表达式（regular expression）库,用一段规则（正则表达式）在文本里批量查找、匹配符合特定格式的字符串
import re

from app.services.chat_service import answer_question

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 用一张健身追踪器的手册插图模拟“用户上传的照片”。
IMAGE_PATH = (
    PROJECT_ROOT.parent
    / "赛题三：知识库资料"
    / "知识库资料"
    / "插图"
    / "Manual16_0.jpg"
)


def load_image_as_data_url(image_path: Path) -> str:
    """把本地图片文件转换成模型API能识别的Base 64 Data URL。"""
    #  读取图片的二进制内容。
    raw_bytes = image_path.read_bytes()

    # Base64编码：二进制 → Base64字节串
    encoded_bytes = base64.b64encode(raw_bytes)

    # Base64结果全是ASCII字符，可以安全解码成普通字符串
    encoded_text = encoded_bytes.decode("ascii")

    # 拼上前缀，模型接口靠前缀识别图片类型。
    return "data:image/jpeg;base64," + encoded_text


def main() -> None:
    if not IMAGE_PATH.exists():
        raise SystemExit(f"找不到测试图片：{IMAGE_PATH}")

    image_data_url = load_image_as_data_url(IMAGE_PATH)

    print("测试图片：", IMAGE_PATH.name)
    # stat()只读元数据（大小、时间），不读文件内容。
    print("图片大小：", IMAGE_PATH.stat().st_size, "字节")
    print("Data URL 长度：", len(image_data_url), "字符")

    # question = (
    #     "请看我上传的这张照片。我想给这台健身追踪器更换表带，"
    #     "知识库里有其他尺寸可选吗？平时佩戴需要注意什么？"
    # )

    question = "请看我上传的这张照片。" "我想给这个设备更换配件，应该怎么操作？"

    print("\n正在调用完整业务链（检索+视觉模型）......")

    result = answer_question(
        question,
        images=[image_data_url],
    )

    print("\n====模型回答====")
    print(result["answer"])

    print("\n====返回的知识库图片====")
    print(result["image_ids"])

    from collections import Counter

    product_distribution = Counter(record["product"] for record in result["references"])

    print("\n====参考资料产品分布====")
    print(product_distribution)

    metrics = result["metrics"]

    returned_image_ids = set(result["image_ids"])

    # 集合推导式：把所有参考资料里出现过的图片ID全部拿出来，集合自动去重
    allowed_image_ids = {
        image_id for record in result["references"] for image_id in record["image_ids"]
    }
    # 列表推导式：循环每一个候选图片 ID，检查这个 ID 有没有直接出现在给用户看的 answer 文本中，存在就放进泄露列表
    leaked_image_ids = [
        image_id for image_id in allowed_image_ids if image_id in result["answer"]
    ]
    # re.findall(pattern, text):文本`text`中，找出所有匹配正则规则的子串，全部放到列表返回。
    leaked_reference_markers = re.findall(
        # 最前面的 r""：原始字符串;(?:参考)?:?: 代表非捕获分组,只用来分组匹配，不单独把 “参考” 提取出来;资料：固定匹配中文 “资料” 这两个字;\s*:`\s` = 任意空白字符（空格）,`*` = 0 次或多次;`\d+``\d` = 数字 0~9
        r"(?:参考)?资料\s*\d+",
        result["answer"],
    )

    answer_sentences = re.split(
        r"[。！？\n]",
        result["answer"],
    )

    leaked_image_recommendation_sentences = [
        sentence
        for sentence in answer_sentences
        if (
            "图片" in sentence
            or "示意图" in sentence
        )
        and any(
            cue in sentence
            for cue in (
                "查看",
                "参考",
                "参见",
            )
        )
    ]

    print("\n====运行指标====")
    for name, value in metrics.items():
        print(f"{name}:{value}")

    print("\n=====检查验收=====")

    # 字典做程序化验收。判断四个硬条件，任意一个不满足就以非零代码退出。
    checks = {
        "路由到视觉模型": metrics["model"] == "qwen3-vl-flash",
        "供应商正确": metrics["provider"] == "openai_compatible",
        "正常结束(stop)": metrics["finish_reason"] == "stop",
        "确实收到1张用户图片": (metrics["input_image_count"] == 1),
        "确实启用了视觉模型": (metrics["vision_used"] is True),
        "视觉预分析已调用": (metrics["vision_analysis_called"] is True),
        "视觉预分析有耗时": (metrics["vision_analysis_latency_ms"] > 0),
        "视觉预分析记录了Token": (
            metrics["vision_analysis_total_tokens"] is not None
            and metrics["vision_analysis_total_tokens"] > 0
        ),
        "视觉预分析执行成功": (metrics["vision_analysis_succeeded"] is True),
        "正常链路没有触发降级": (metrics["vision_fallback_used"] is False),
        "正常链路没有预分析错误": (metrics["vision_analysis_error"] is None),
        "至少返回1张知识库图片": (len(returned_image_ids) >= 1),
        "知识库图片不超过3张": (len(returned_image_ids) <= 3),
        "返回图片均来自候选资料": (returned_image_ids <= allowed_image_ids),
        "回答正文没有泄露图片ID": (not leaked_image_ids),
        "回答正文没有泄露资料编号": (not leaked_reference_markers),
        "回答正文没有重复推荐图片": (not leaked_image_recommendation_sentences),
    }

    all_passed = True
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n多模态业务链测试：全部通过")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

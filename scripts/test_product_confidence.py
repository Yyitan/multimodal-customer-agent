from app.vision_analyzer import normalize_product_confidence

def main() -> None:
    product = "健身追踪器"

    checks = {
        "保留正常置信度": (
            normalize_product_confidence(
                0.85,
                product,
            )
            == 0.85
        ),
        "超过1时限制为1": (
            normalize_product_confidence(
                1.5,
                product,
            )
            == 1.0
        ),
        "小于0时限制为0": (
            normalize_product_confidence(
                -0.3,
                product,
            )
            == 0.0
        ),
        "字符串置信度视为无效": (
            normalize_product_confidence(
                "很确定",
                product,
            )
            == 0.0
        ),
        "布尔值置信度视为无效": (
            normalize_product_confidence(
                True,
                product,
            )
            == 0.0
        ),
        "没有产品名称时置信度为0": (
            normalize_product_confidence(
                0.95,
                "",
            )
            == 0.0
        ),
    }

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n产品置信度规范化测试：全部通过")


if __name__ == "__main__":
    main()
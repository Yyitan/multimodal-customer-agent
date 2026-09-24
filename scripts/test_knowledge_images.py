from pathlib import Path
from tempfile import TemporaryDirectory

from app.knowledge_images import KnowledgeImageCatalog


def main() -> None:
    checks: dict[str, bool] = {}
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "Manual16_9.jpg").write_bytes(b"demo")
        catalog = KnowledgeImageCatalog(root)

        resolved = catalog.resolve("Manual16_9")
        checks["合法图片ID解析到真实文件"] = (
            resolved is not None and resolved.name == "Manual16_9.jpg"
        )
        checks["未知图片ID返回空"] = catalog.resolve("unknown") is None
        checks["路径穿越被拒绝"] = catalog.resolve("../secret") is None

    print("====知识图片目录测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")
    if not all(checks.values()):
        raise SystemExit(1)
    print("\n知识图片目录测试：全部通过")


if __name__ == "__main__":
    main()

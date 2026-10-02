from pathlib import Path

from app.main import app
from app.version import APP_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    checks: dict[str, bool] = {}

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")

    checks["应用版本和OpenAPI版本一致"] = (
        APP_VERSION == "2.1.0" and app.version == APP_VERSION
    )
    checks["README记录当前版本和核心接口"] = (
        f"当前版本：`{APP_VERSION}`" in readme
        and "/v1/chat" in readme
        and "/v1/operations/summary" in readme
    )
    checks["发布文档齐全"] = all(
        (PROJECT_ROOT / path).is_file()
        for path in (
            "docs/architecture.md",
            "docs/api.md",
            "docs/deployment.md",
            "docs/evaluation.md",
            "docs/resume-guide.md",
        )
    )
    checks["变更日志包含当前版本"] = f"[{APP_VERSION}]" in changelog
    checks["容器使用版本化就绪检查"] = "/v1/ready" in dockerfile
    checks["Compose持久化两个业务数据库"] = (
        "AGENT_CONVERSATION_DB_PATH: data/runtime/conversations.db" in compose
        and "AGENT_SUPPORT_DB_PATH: data/runtime/support.db" in compose
    )

    print("====发布完整性测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")
    if not all(checks.values()):
        raise SystemExit(1)
    print("\n发布完整性测试：全部通过")


if __name__ == "__main__":
    main()

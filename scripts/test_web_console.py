import os

from fastapi.testclient import TestClient

os.environ["AGENT_CONVERSATION_DB_PATH"] = ":memory:"
os.environ["AGENT_SUPPORT_DB_PATH"] = ":memory:"

from app.main import app


def main() -> None:
    client = TestClient(app)
    checks: dict[str, bool] = {}

    page = client.get("/")
    checks["工作台首页可访问"] = (
        page.status_code == 200
        and "多模态客服 Agent" in page.text
        and "对话工作台" in page.text
        and "工单队列" in page.text
    )
    css = client.get("/assets/app.css")
    checks["工作台样式资源可访问"] = (
        css.status_code == 200 and ".chat-layout" in css.text
    )
    script = client.get("/assets/app.js")
    checks["工作台脚本使用v1接口"] = (
        script.status_code == 200
        and 'api("/v1/chat"' in script.text
        and 'api("/v1/feedback"' in script.text
        and "/v1/tickets" in script.text
        and "/v1/operations/summary" in script.text
        and "formatTimestamp" in script.text
    )

    print("====Web工作台测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")
    if not all(checks.values()):
        raise SystemExit(1)
    print("\nWeb工作台测试：全部通过")


if __name__ == "__main__":
    main()

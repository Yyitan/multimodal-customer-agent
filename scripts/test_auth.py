import os
from unittest.mock import patch

# `SimpleNamespace`：可以动态造一个对象，用来模拟 settings 配置对象
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import SecretStr

os.environ["AGENT_CONVERSATION_DB_PATH"] = ":memory:"

from app.main import app, conversation_store
from app.request_control import SlidingWindowRateLimiter
from scripts.test_api import make_service_result

client = TestClient(app)


def main() -> None:
    # 第一组：正常鉴权场景:auth_enabled=True，api_token=SecretStr
    auth_settings = SimpleNamespace(
        auth_enabled=True,
        api_token=SecretStr("test-token"),
        llm_provider="mock",
        llm_model="mock",
    )

    conversation_store.clear("auth-session")

    with patch(
        "app.main.settings",
        auth_settings,
    ):
        # 用例 1：health 接口公开访问
        health_response = client.get("/health")
        checks = {
            "health接口保持公开": (health_response.status_code == 200),
        }
        # 用例 2：访问 /chat，完全不带 Authorization 头
        no_token_response = client.post(
            "/chat",
            json={
                "question": "测试鉴权",
                "session_id": "auth-session",
            },
        )
        checks["没有Token返回401"] = no_token_response.status_code == 401
        # 用例 3：带 Authorization，但是 token 是错的
        wrong_token_response = client.post(
            "/chat",
            headers={
                "Authorization": "Bearer wrong-token",
            },
            json={
                "question": "测试错误Token",
                "session_id": "auth-session",
            },
        )
        checks["错误Token返回401"] = wrong_token_response.status_code == 401

        # 用例 4：携带正确 Token 访问 chat 接口
        # 这里嵌套一层 patch：mock 业务函数`answer_question`，不需要真实走检索和 LLM
        with patch(
            "app.main.answer_question",
            return_value=make_service_result("鉴权通过后的模拟回答。"),
        ):
            correct_token_response = client.post(
                "/chat",
                # 请求头`Bearer test-token`完全匹配配置，鉴权放行，接口正常返回 200
                headers={
                    "Authorization": "Bearer test-token",
                },
                json={
                    "question": "测试正确Token",
                    "session_id": "auth-session",
                },
            )

        checks["正确Token返回200"] = correct_token_response.status_code == 200
        # 用例 5：删除会话接口（DELETE /sessions/auth-session）不带 token
        delete_without_token_response = client.delete(
            "/sessions/auth-session",
        )
        checks["删除会话没有Token返回401"] = (
            delete_without_token_response.status_code == 401
        )
        # 用例 6：删除会话带上正确 token
        delete_with_token_response = client.delete(
            "/sessions/auth-session",
            headers={
                "Authorization": "Bearer test-token",
            },
        )
        checks["删除会话携带正确Token返回200"] = (
            delete_with_token_response.status_code == 200
        )
    # 第二组：异常配置场景：auth_enabled=True，api_token=None
    misconfigured_settings = SimpleNamespace(
        auth_enabled=True,
        api_token=None,
        llm_provider="mock",
        llm_model="mock",
    )

    with patch(
        "app.main.settings",
        misconfigured_settings,
    ):
        misconfigured_response = client.post(
            "/chat",
            headers={
                "Authorization": "Bearer test-token",
            },
            json={
                "question": "测试服务端缺少Token",
            },
        )

    checks["服务端未配置Token返回503"] = misconfigured_response.status_code == 503

    rate_limit_settings = SimpleNamespace(
        auth_enabled=False,
        api_token=None,
        rate_limit_enabled=True,
        llm_provider="mock",
        llm_model="mock",
    )
    test_limiter = SlidingWindowRateLimiter(
        max_requests=1,
        window_seconds=60,
    )
    with (
        patch("app.main.settings", rate_limit_settings),
        patch("app.main.rate_limiter", test_limiter),
        patch(
            "app.main.answer_question",
            return_value=make_service_result("限流测试回答。"),
        ),
    ):
        first_limited_response = client.post(
            "/chat",
            json={"question": "第一次限流测试"},
        )
        second_limited_response = client.post(
            "/chat",
            json={"question": "第二次限流测试"},
        )

    checks["限流范围内请求返回200"] = (
        first_limited_response.status_code == 200
    )
    checks["超过限流返回429和Retry-After"] = (
        second_limited_response.status_code == 429
        and int(second_limited_response.headers["retry-after"]) > 0
    )

    print("====API鉴权测试====")

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\nAPI鉴权测试：全部通过")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""查询改写测试：配置开关默认值、失败回退拼接、开启时使用改写结果。"""
import os
from unittest.mock import patch

os.environ["AGENT_CONVERSATION_DB_PATH"] = ":memory:"
os.environ["AGENT_SUPPORT_DB_PATH"] = ":memory:"

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app, conversation_store
from app.query_rewriter import build_rewrite_messages

client = TestClient(app)


def check(name: str, passed: bool) -> bool:
    print(f"{'通过' if passed else '失败'}|{name}")
    return passed


def main() -> None:
    checks: dict[str, bool] = {}

    # 1. 配置默认关闭
    settings = get_settings().model_copy()
    checks["改写开关默认关闭"] = settings.query_rewrite_enabled is False

    # 2. 构造消息包含历史与当前问题
    messages = build_rewrite_messages(
        "充满要多久",
        [
            {"question": "健身追踪器怎么充电？", "answer": "使用充电线充电。"},
        ],
    )
    checks["改写消息包含历史和问题"] = (
        "健身追踪器怎么充电" in messages[1]["content"]
        and "充满要多久" in messages[1]["content"]
    )

    # 3. 开启改写 + 改写成功：第二轮追问触发改写器并记录观测字段
    conversation_store.clear("rewrite-on-session")
    fake_settings = get_settings().model_copy(
        update={"query_rewrite_enabled": True}
    )
    with patch("app.main.settings", fake_settings):
        with patch(
            "app.services.chat_service.get_settings",
            return_value=fake_settings,
        ):
            with patch(
                "app.services.chat_service.rewrite_query_with_history",
                return_value={
                    "query": "健身追踪器的表带怎么更换",
                    "latency_ms": 15,
                    "total_tokens": 120,
                },
            ) as fake_rewrite:
                client.post(
                    "/chat",
                    json={
                        "question": "健身追踪器怎么充电？",
                        "session_id": "rewrite-on-session",
                    },
                )
                response = client.post(
                    "/chat",
                    json={
                        "question": "这个怎么弄",
                        "session_id": "rewrite-on-session",
                    },
                )
    data = response.json()["data"]
    checks["开启改写时调用改写器"] = fake_rewrite.called
    checks["改写开启仍正常返回回答"] = (
        response.status_code == 200 and len(data["references"]) > 0
    )
    checks["指标记录改写观测字段"] = (
        data["metrics"].get("query_rewrite_used") is True
        and data["metrics"].get("query_rewrite_tokens") == 120
    )

    # 4. 开启改写 + 改写失败：回退拼接，服务不中断
    conversation_store.clear("rewrite-fallback-session")
    with patch("app.main.settings", fake_settings):
        with patch(
            "app.services.chat_service.get_settings",
            return_value=fake_settings,
        ):
            with patch(
                "app.services.chat_service.rewrite_query_with_history",
                side_effect=RuntimeError("改写模型不可用"),
            ):
                client.post(
                    "/chat",
                    json={
                        "question": "健身追踪器怎么充电？",
                        "session_id": "rewrite-fallback-session",
                    },
                )
                response = client.post(
                    "/chat",
                    json={
                        "question": "健身追踪器的表带怎么更换？",
                        "session_id": "rewrite-fallback-session",
                    },
                )
    data = response.json()["data"]
    checks["改写失败回退拼接仍正常回答"] = (
        response.status_code == 200 and len(data["references"]) > 0
    )
    checks["回退时改写标记为未使用"] = (
        data["metrics"].get("query_rewrite_used") is False
    )

    # 5. 关闭改写：不触碰改写器
    conversation_store.clear("rewrite-off-session")
    with patch("app.main.settings", get_settings().model_copy()):
        with patch(
            "app.services.chat_service.rewrite_query_with_history"
        ) as fake_rewrite:
            response = client.post(
                "/chat",
                json={
                    "question": "健身追踪器的表带怎么更换？",
                    "session_id": "rewrite-off-session",
                },
            )
    checks["关闭改写不调用改写器"] = (
        response.status_code == 200 and not fake_rewrite.called
    )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"查询改写测试失败：{failed}")
    print("\n查询改写测试：全部通过")


if __name__ == "__main__":
    main()

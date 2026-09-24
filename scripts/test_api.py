import os
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ["AGENT_CONVERSATION_DB_PATH"] = ":memory:"
os.environ["AGENT_SUPPORT_DB_PATH"] = ":memory:"

from app.llm_client import LLMClientError
from app.main import app, conversation_store, support_store

"""FastAPI 接口层的集成测试脚本"""

# `TestClient`：FastAPI 自带测试工具，可以模拟发送 `GET/POST/DELETE` HTTP 请求，不用启动 uvicorn 服务。
client = TestClient(app)


def make_service_result(
    answer: str = "这是一次模拟客服回答。",
) -> dict:
    """构造符合 ChatResponse 要求的模拟业务结果。"""
    return {
        "answer": answer,
        "references": [
            {
                "chunk_id": "api_test_001",
                "product": "健身追踪器",
                "text": "请先找到表带卡扣，再进行拆卸。",
                "image_ids": [],
                "source": "api_test.jsonl",
                "score": 1.0,
                "retrieval_type": "semantic",
            }
        ],
        "image_ids": [],
        "metrics": {
            "llm_called": True,
            "provider": "mock",
            "model": "mock",
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
            "llm_latency_ms": 5,
            "vision_analysis_called": False,
            "vision_analysis_latency_ms": 0,
            "vision_analysis_prompt_tokens": None,
            "vision_analysis_completion_tokens": None,
            "vision_analysis_total_tokens": None,
            "vision_analysis_succeeded": False,
            "vision_fallback_used": False,
            "vision_analysis_error": None,
            "retrieval_latency_ms": 2,
            "total_latency_ms": 10,
            "retrieved_record_count": 1,
            "context_record_count": 1,
            "context_length": 30,
            "finish_reason": "stop",
            "selected_image_count": 0,
            "input_image_count": 0,
            "vision_used": False,
            "vision_product_hint": None,
            "vision_product_confidence": None,
            "vision_product_strategy": "none",
        },
    }


def main() -> None:
    checks: dict[str, bool] = {}

    conversation_store.clear("api-session")
    # 健康检查接口 /health
    health_response = client.get("/health")
    checks["健康检查返回200"] = health_response.status_code == 200
    checks["健康检查状态正确"] = health_response.json() == {"status": "ok"}
    checks["响应自动包含请求ID"] = bool(
        health_response.headers.get("x-request-id")
    )
    supplied_request_id = "client-request-001"
    echoed_response = client.get(
        "/health",
        headers={"X-Request-ID": supplied_request_id},
    )
    checks["合法调用方请求ID被原样返回"] = (
        echoed_response.headers.get("x-request-id")
        == supplied_request_id
    )
    ready_response = client.get("/ready")
    checks["就绪检查返回200"] = ready_response.status_code == 200
    checks["就绪检查包含数据库和索引状态"] = (
        ready_response.json()["status"] == "ready"
        and ready_response.json()["database"] == "ok"
        and ready_response.json()["support_database"] == "ok"
        and ready_response.json()["index"]["record_count"] > 0
    )
    checks["v1健康检查可用"] = (
        client.get("/v1/health").json() == {"status": "ok"}
    )
    # 请求参数校验（Pydantic 校验）
    empty_question_response = client.post(
        "/chat",
        json={"question": ""},
    )
    checks["空问题返回422"] = empty_question_response.status_code == 422

    invalid_image_response = client.post(
        "/chat",
        json={
            "question": "测试问题",
            "images": ["not-a-data-url"],
        },
    )
    checks["非法图片格式返回422"] = invalid_image_response.status_code == 422

    too_many_images_response = client.post(
        "/chat",
        json={
            "question": "测试问题",
            "images": [
                "data:image/jpeg;base64,TEST1",
                "data:image/jpeg;base64,TEST2",
                "data:image/jpeg;base64,TEST3",
                "data:image/jpeg;base64,TEST4",
            ],
        },
    )
    checks["超过三张图片返回422"] = too_many_images_response.status_code == 422
    # 多轮会话接口测试
    fake_results = [
        make_service_result("自动会话回答。"),
        make_service_result("第一轮会话回答。"),
        make_service_result("第二轮会话回答。"),
        make_service_result("清空后的回答。"),
    ]

    with patch(
        "app.main.answer_question",
        side_effect=fake_results,
    ) as answer_mock:
        auto_session_response = client.post(
            "/chat",
            json={"question": "自动生成会话的问题"},
        )

        checks[
            "未传session_id时自动生成会话ID"
        ] = auto_session_response.status_code == 200 and auto_session_response.json()[
            "data"
        ][
            "session_id"
        ].startswith(
            "kf_session_"
        )

        first_response = client.post(
            "/chat",
            json={
                "question": "我有一个健身追踪器。",
                "session_id": "api-session",
            },
        )

        second_response = client.post(
            "/chat",
            json={
                "question": "那拆卸时先做什么？",
                "session_id": "api-session",
            },
        )

        checks["第一次会话请求成功"] = first_response.status_code == 200
        checks["第二次会话请求成功"] = second_response.status_code == 200
        checks["知识问答返回Agent决策"] = (
            second_response.json()["data"]["agent"]["intent"]
            == "knowledge_question"
            and second_response.json()["data"]["agent"]["action"]
            == "respond_with_knowledge"
            and second_response.json()["data"]["agent"]["handoff_required"]
            is False
        )

        first_history = answer_mock.call_args_list[1].kwargs["conversation_history"]
        second_history = answer_mock.call_args_list[2].kwargs["conversation_history"]

        checks["第一次请求没有历史"] = first_history == []
        checks["第二次请求读取上一轮历史"] = (
            len(second_history) == 1
            and second_history[0]["question"] == "我有一个健身追踪器。"
            and second_history[0]["answer"] == "第一轮会话回答。"
        )
        # 调用删除会话接口 DELETE /sessions/api-session
        clear_response = client.delete("/sessions/api-session")
        checks["清空会话接口返回200"] = clear_response.status_code == 200
        checks["清空会话结果正确"] = clear_response.json()["cleared"] is True
        # 清空之后，再次提问
        third_response = client.post(
            "/chat",
            json={
                "question": "清空后再次提问",
                "session_id": "api-session",
            },
        )

        third_history = answer_mock.call_args_list[3].kwargs["conversation_history"]

        checks["清空后再次请求没有历史"] = (
            third_response.status_code == 200 and third_history == []
        )

    with patch(
        "app.main.answer_question",
        side_effect=LLMClientError("模拟模型调用失败"),
    ):
        error_response = client.post(
            "/chat",
            json={
                "question": "触发模型错误",
                "session_id": "error-session",
            },
        )

    checks["模型异常转换为503"] = error_response.status_code == 503

    with patch("app.main.answer_question") as handoff_answer_mock:
        handoff_response = client.post(
            "/chat",
            json={
                "question": "我要转人工客服处理退款",
                "session_id": "handoff-session",
            },
        )
    checks["转人工请求跳过模型调用"] = (
        handoff_response.status_code == 200
        and handoff_response.json()["data"]["agent"]["handoff_required"]
        is True
        and handoff_answer_mock.call_count == 0
    )
    handoff_data = handoff_response.json()["data"]
    ticket_id = handoff_data["agent"]["ticket_id"]
    ticket_response = client.get(f"/tickets/{ticket_id}")
    checks["转人工请求自动创建可查询工单"] = (
        ticket_response.status_code == 200
        and ticket_response.json()["session_id"] == "handoff-session"
        and ticket_response.json()["status"] == "open"
    )

    update_ticket_response = client.patch(
        f"/tickets/{ticket_id}",
        json={"status": "in_progress"},
    )
    checks["工单接口支持状态流转"] = (
        update_ticket_response.status_code == 200
        and update_ticket_response.json()["status"] == "in_progress"
    )

    feedback_response = client.post(
        "/feedback",
        json={
            "session_id": "handoff-session",
            "request_id": handoff_data["request_id"],
            "rating": 5,
            "comment": "人工转接清晰。",
        },
    )
    checks["反馈接口保存评分"] = (
        feedback_response.status_code == 201
        and feedback_response.json()["rating"] == 5
    )

    metrics_response = client.get("/metrics")
    checks["指标接口输出Agent和HTTP指标"] = (
        metrics_response.status_code == 200
        and "agent_chat_requests_total" in metrics_response.text
        and "http_requests_total" in metrics_response.text
        and "support_tickets_created_total" in metrics_response.text
        and "customer_feedback_total" in metrics_response.text
    )
    v1_chat_response = client.post(
        "/v1/chat",
        json={"question": "我要转人工", "session_id": "v1-session"},
    )
    checks["v1问答接口与旧接口并存"] = (
        v1_chat_response.status_code == 200
        and v1_chat_response.json()["data"]["agent"]["handoff_required"]
        is True
    )
    image_response = client.get("/v1/knowledge-images/Manual16_9")
    checks["知识图片接口返回真实图片"] = (
        image_response.status_code == 200
        and image_response.headers["content-type"].startswith("image/")
    )

    print("====接口层测试====")

    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n接口层测试：全部通过")


if __name__ == "__main__":
    main()

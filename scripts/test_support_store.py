from pathlib import Path
from tempfile import TemporaryDirectory

from app.support_store import SupportStore


def main() -> None:
    checks: dict[str, bool] = {}

    with TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "support.db"
        store = SupportStore(database_path)

        ticket = store.create_ticket(
            session_id="support-session",
            request_id="request-001",
            question="我要申请退款",
            reason="business_system_action_required",
            priority="high",
        )
        checks["创建人工工单"] = (
            ticket["ticket_id"].startswith("ticket_")
            and ticket["status"] == "open"
            and ticket["priority"] == "high"
            and 1_000_000_000 < ticket["created_at"] < 10_000_000_000
        )

        duplicate = store.create_ticket(
            session_id="support-session",
            request_id="request-001",
            question="重复请求",
            reason="duplicate",
        )
        checks["同一请求ID不会重复创建工单"] = (
            duplicate["ticket_id"] == ticket["ticket_id"]
            and len(store.list_tickets()) == 1
        )

        updated = store.update_ticket_status(
            ticket["ticket_id"],
            "in_progress",
        )
        checks["工单状态可以流转"] = (
            updated is not None
            and updated["status"] == "in_progress"
            and len(store.list_tickets(status="in_progress")) == 1
        )

        feedback = store.add_feedback(
            session_id="support-session",
            request_id="request-001",
            rating=4,
            comment="回答有帮助，但希望更简洁。",
        )
        checks["保存用户评分和反馈"] = (
            feedback["feedback_id"].startswith("feedback_")
            and feedback["rating"] == 4
            and "更简洁" in feedback["comment"]
        )

        store.add_feedback(
            session_id="support-session",
            request_id="request-002",
            rating=2,
        )
        summary = store.get_operations_summary()
        checks["运营摘要聚合工单和评分"] = (
            summary["ticket_counts"]["in_progress"] == 1
            and summary["active_ticket_count"] == 1
            and summary["urgent_open_count"] == 0
            and summary["feedback_count"] == 2
            and summary["feedback_average_rating"] == 3.0
            and summary["rating_counts"]["4"] == 1
        )

        store.close()
        reopened = SupportStore(database_path)
        persisted = reopened.get_ticket(ticket["ticket_id"])
        checks["服务重启后工单仍然存在"] = (
            persisted is not None
            and persisted["status"] == "in_progress"
        )
        reopened.close()

    print("====客服闭环存储测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n客服闭环存储测试：全部通过")


if __name__ == "__main__":
    main()

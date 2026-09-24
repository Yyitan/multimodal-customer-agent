from pydantic import SecretStr

from app.config import Settings, validate_runtime_settings
from app.conversation_store import ConversationStore
from app.semantic_retriever import semantic_index_status
from app.support_store import SupportStore


def main() -> None:
    checks: dict[str, bool] = {}

    development_settings = Settings(
        _env_file=None,
        environment="development",
        llm_provider="mock",
        llm_model="mock",
        auth_enabled=False,
    )
    checks["开发模式允许Mock且关闭鉴权"] = (
        validate_runtime_settings(development_settings) == []
    )

    production_settings = Settings(
        _env_file=None,
        environment="production",
        llm_provider="openai_compatible",
        llm_model="real-model",
        llm_api_key=SecretStr(""),
        auth_enabled=False,
        llm_vision_model=None,
        llm_vision_api_key=None,
    )
    production_errors = validate_runtime_settings(
        production_settings
    )
    checks["生产模式拒绝缺失鉴权和模型密钥"] = (
        len(production_errors) == 5
        and any("API Token" in error for error in production_errors)
        and any("限流" in error for error in production_errors)
        and any("LLM_API_KEY" in error for error in production_errors)
        and any("VISION_MODEL" in error for error in production_errors)
        and any("VISION_API_KEY" in error for error in production_errors)
    )

    index_status = semantic_index_status()
    checks["知识索引状态包含记录数和维度"] = (
        index_status["record_count"] > 0
        and index_status["embedding_dimension"] > 0
        and bool(index_status["model_name"])
        and isinstance(index_status["model_loaded"], bool)
    )

    store = ConversationStore(db_path=":memory:")
    checks["SQLite连接健康检查成功"] = store.ping() is True
    store.close()
    checks["SQLite关闭后健康检查失败"] = store.ping() is False

    support_store = SupportStore(db_path=":memory:")
    checks["客服闭环数据库健康检查成功"] = support_store.ping() is True
    support_store.close()
    checks["客服闭环数据库关闭后检查失败"] = (
        support_store.ping() is False
    )

    print("====运行就绪测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n运行就绪测试：全部通过")


if __name__ == "__main__":
    main()

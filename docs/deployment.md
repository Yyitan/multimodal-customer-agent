# 生产部署

## 配置准备

复制生产模板并替换所有占位值：

```powershell
Copy-Item .env.production.example .env
```

生产启动校验会强制要求：

- `AGENT_AUTH_ENABLED=true` 且配置 `AGENT_API_TOKEN`
- `AGENT_RATE_LIMIT_ENABLED=true`
- 文本模型名称和 API Key
- 视觉模型名称和 API Key

不要把 `.env`、密钥或 Token 提交到 Git。

## 发布前检查

```powershell
.\.venv\Scripts\Activate.ps1
python -m scripts.run_regression
python -m scripts.evaluate_retrieval
python -m scripts.evaluate_generation
git diff --check
git status
```

真实生成评测依赖模型供应商可用额度。429、超时或供应商故障会被写入 `artifacts/evaluation/generation.json` 并返回非零退出码。

## Docker Compose

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f agent
```

服务端口为 `8000`。Compose 会持久化：

- `agent-data`：SQLite 会话数据库
- `huggingface-cache`：Embedding 模型缓存

停止服务：

```powershell
docker compose down
```

删除命名卷会永久清空会话数据库和模型缓存，除非明确需要，不要使用 `docker compose down -v`。

## 健康检查

- `/health` 只表示 Web 进程存活。
- `/ready` 会检查运行配置、SQLite 和知识索引；只有返回 200 才应该接收流量。

反向代理或容器平台应使用 `/ready` 作为 readiness probe，并为 `/chat` 配置大于模型最大响应时间的上游超时。

## 单 Worker 约束

生产命令固定为 `--workers 1`，原因是：

- 限流状态位于单进程内存中。
- 每个进程会建立自己的 SQLite 连接和模型对象。
- 多 worker 会重复加载 Embedding 模型，显著增加内存占用。

需要水平扩容时，应先完成以下改造：

1. 使用 Redis 保存共享限流计数。
2. 使用 PostgreSQL 或其他共享数据库保存会话。
3. 确认模型缓存和知识索引在所有实例间一致。
4. 在负载均衡器后执行并发与故障转移测试。

## 运维建议

- 在反向代理层启用 HTTPS，不直接暴露明文模型 Token。
- 定期备份 `conversations.db`，备份前确认 SQLite WAL 文件已同步。
- 监控 429、503、总耗时、模型耗时、检索耗时、Token 用量和视觉降级比例。
- 更新知识库后同时提交标准化记录、索引文件和通过的检索评测结果。

# 生产部署

## 发布前检查

```powershell
.\.venv\Scripts\Activate.ps1
python -m compileall -q app scripts
python -m scripts.run_regression
python -m scripts.evaluate_retrieval
python -m scripts.evaluate_agent
git diff --check
git status
```

有真实模型额度时再运行：

```powershell
python -m scripts.evaluate_generation
python -m scripts.run_regression --live
```

## 生产配置

```powershell
Copy-Item .env.production.example .env
```

必须替换以下占位值：

- 文本模型的 Provider、Model、Base URL 与 API Key。
- 视觉模型的 Provider、Model、Base URL 与 API Key。
- 长随机值 `AGENT_API_TOKEN`。

生产启动校验要求鉴权和限流开启，且真实模型配置完整。`.env`、API Key 和 Token 不得提交到 Git。

## Docker Compose

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f agent
```

Compose 配置：

- 非 root 用户运行。
- 单 Uvicorn Worker。
- `/v1/ready` 容器健康检查。
- `agent-data` 保存 `conversations.db` 和 `support.db`。
- `huggingface-cache` 保存 Embedding 模型缓存。

停止服务：

```powershell
docker compose down
```

除非明确需要清空数据，不要执行 `docker compose down -v`。

## 启动验证

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/info
Invoke-RestMethod http://127.0.0.1:8000/v1/health
Invoke-RestMethod http://127.0.0.1:8000/v1/ready
```

随后使用生产 Token 调用 `/v1/chat`、`/v1/tickets` 和 `/v1/metrics`。

## 性能基线

对目标环境执行：

```powershell
python -m scripts.benchmark_api `
  --base-url https://agent.example.com `
  --token $env:AGENT_API_TOKEN `
  --requests 100 `
  --concurrency 10 `
  --max-p95-ms 10000 `
  --max-error-rate 0.01
```

先用 Mock 模型建立检索与 API 基线，再用真实模型测试端到端延迟。两类数字不能混用。

## 监控建议

采集 `/v1/metrics` 并至少监控：

- HTTP 429、5xx 与请求时延。
- Agent 动作、转人工比例和工单创建量。
- 检索、文本模型和视觉模型时延。
- 模型重试、熔断拒绝和视觉降级比例。
- 用户评分数量、平均分和低分占比。
- 活跃工单与紧急未结工单。

日志应按 `request_id` 聚合，避免记录 API Token、模型密钥、原始 Base64 图片和其他敏感信息。

## 备份与恢复

需要备份命名卷中的：

- `data/runtime/conversations.db` 及其 WAL/SHM 文件。
- `data/runtime/support.db` 及其 WAL/SHM 文件。

更稳妥的方式是在停止写入或执行 SQLite 在线备份后复制。恢复时保持文件权限属于容器中的 `agent` 用户，并在接流量前验证 `/v1/ready`。

## 反向代理

- 仅通过 HTTPS 暴露服务。
- 限制请求体大小，且不能小于应用的 `AGENT_MAX_REQUEST_BYTES`。
- 上游超时应大于模型最大超时与重试总预算。
- 将 `X-Request-ID` 透传给服务。
- `/v1/health` 用于 liveness，`/v1/ready` 用于 readiness。

## 单 Worker 与扩容

当前模板固定单 Worker，原因是 SQLite 连接、进程内限流、熔断状态和 Embedding 模型对象均属于单进程。

水平扩容前完成：

1. 会话、工单和反馈迁移到 PostgreSQL 或其他共享数据库。
2. 限流和熔断状态迁移到 Redis 或网关。
3. 确保所有实例使用同一知识索引版本。
4. 在负载均衡器后重新运行并发、重试和故障转移测试。

## 回滚

- 保留上一版本镜像和对应配置。
- 数据库变更应保持向后兼容；v2 启动时会把旧纳秒时间戳迁移为 Unix 秒。
- 回滚前备份两个 SQLite 数据库。
- 回滚后检查服务版本、就绪状态、工单读取和一次 Mock/真实请求。

## 从零到公网（十步清单）

1. 购买 VPS（1C2G 起步即可，Ubuntu 22.04+）并安装 Docker Compose。
2. 域名解析 A 记录到服务器 IP。
3. 克隆仓库，`Copy-Item`/`cp .env.production.example .env`，填入真实模型 Key 与强随机 `AGENT_API_TOKEN`。
4. `AGENT_AUTH_ENABLED=true`、`AGENT_RATE_LIMIT_ENABLED=true`（生产强制项，启动校验会拦截缺省）。
5. `docker compose up --build -d`，确认 `docker compose ps` 健康、`curl /v1/ready` 返回 ready。
6. 安装 Caddy 或 Nginx + certbot 做反向代理与 HTTPS，转发到 8000 端口，禁用直连 IP 访问。
7. 工作台填入 API Token，验证一次真实问答、工单创建与状态流转。
8. 配置日志轮转与 SQLite 数据库定时备份（命名卷路径见 compose.yaml）。
9. 在目标环境重跑 `benchmark_api`（放宽 P95 阈值）与 `evaluate_generation`，记录真实基线。
10. 邀请真实用户使用，收集反馈与低分工单，驱动评测集与规则迭代。

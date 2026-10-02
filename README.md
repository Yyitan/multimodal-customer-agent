# 多模态客服 Agent

一个可运行、可评测、可部署的中文多模态客服系统。服务接收文字和设备图片，先完成 Agent 路由，再使用产品手册知识库进行混合检索与视觉增强回答；对于安全风险、业务操作、用户明确要求人工或证据不足的请求，自动创建人工工单。

当前版本：`2.1.0`

## 项目定位

这不是只调用一次大模型的问答 Demo。项目覆盖了客服 Agent 的完整业务链：

- 决策：意图识别、动作选择、证据复核、人工升级。
- 感知：最多 3 张图片的视觉预分析和产品置信度判断。
- 知识：关键词与向量混合检索、产品过滤、邻接知识扩展。
- 记忆：SQLite 多轮会话历史，支持服务重启后恢复。
- 工具：人工工单、状态流转、用户反馈和运营汇总。
- 韧性：视觉降级、模型重试、指数退避、熔断和错误分类。
- 工程：API 鉴权、限流、输入校验、版本化接口、指标和审计日志。
- 交付：客服运营工作台、Docker、CI、自动化回归、评测和性能基线。

## 系统流程

```text
用户文字 / 图片
  -> API 鉴权、限流和输入安全校验
  -> Agent 请求分类
     -> 安全风险 / 业务操作 / 明确转人工：创建工单
     -> 知识类请求：继续处理
  -> 读取会话历史
  -> 视觉预分析（失败时降级为纯文本）
  -> 置信度驱动的产品检索策略
  -> 关键词 + BGE 向量混合检索
  -> 上下文构造和知识图片候选排序
  -> 大模型生成与结构化图片选择
  -> 证据复核（证据不足则转人工）
  -> 保存会话、指标、反馈和工单
```

详细设计见 [架构说明](docs/architecture.md)。

## 数据与质量

- 20 个产品手册知识域。
- 778 条标准化知识记录。
- 512 维 `BAAI/bge-small-zh-v1.5` 向量索引。
- 20 组 Mock 回归测试，覆盖接口、Agent、存储、安全、韧性和发布完整性。
- 56 个检索案例（覆盖全部 20 个产品 + 对抗案例），55 通过 + 1 已知排序缺陷（xfail）。
- 59 个 Agent 路由案例（含口语化对抗案例），规则迭代后意图/动作/转人工 F1 均 100%，迭代前真实通过率 45.76%。
- 12 个真实模型生成案例，通过率 91.67%，返回图片零跨产品污染。
- 真实模型生成评测（2026-10-02，文本 `glm-4.5-flash`）：首次 5 案例 4/5 通过。
- 真实多模态 live 回归（2026-10-02，视觉 `qwen3-vl-flash`）：21/21 通过。
- 真实模型端到端基线（2026-10-02）：20 请求、并发 2、0% 错误率、P50 4.48 s、P95 6.45 s、0.41 req/s。
- 本机 Mock 链路性能基线：50 请求、并发 5、0% 错误率、P95 177.150 ms、31.744 req/s。

性能数字分两套口径：Mock 基线只代表 2026-09-27 本地 Mock 模型与本地检索的结果，用于度量编排层本身；真实模型基线代表 2026-10-02 接入智谱 `glm-4.5-flash` 与阿里云 `qwen3-vl-flash` 后的端到端延迟，主导 P95 的是大模型生成耗时。完整口径见 [评测说明](docs/evaluation.md)。

## 快速启动

要求 Python 3.12。在项目根目录执行：

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

默认使用 Mock 文本模型，可直接体验本地检索和业务流程。打开：

- 运营工作台：`http://127.0.0.1:8000/`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`
- 服务信息：`http://127.0.0.1:8000/v1/info`
- 就绪检查：`http://127.0.0.1:8000/v1/ready`

接入真实模型时，在 `.env` 中配置文本模型和视觉模型的 Provider、Model、Base URL 与 API Key。生产环境必须启用 API Token 与限流。

## 核心接口

所有稳定业务接口使用 `/v1` 前缀：

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/v1/chat` | 多模态 Agent 对话 |
| `DELETE` | `/v1/sessions/{session_id}` | 清空会话历史 |
| `GET` | `/v1/tickets` | 查询人工工单 |
| `PATCH` | `/v1/tickets/{ticket_id}` | 更新工单状态 |
| `POST` | `/v1/feedback` | 保存用户评分与反馈 |
| `GET` | `/v1/operations/summary` | 获取运营摘要 |
| `GET` | `/v1/metrics` | Prometheus 文本指标 |
| `GET` | `/v1/knowledge-images/{image_id}` | 获取知识库图片 |

请求示例：

```powershell
$headers = @{ Authorization = "Bearer 你的AGENT_API_TOKEN" }
$body = @{
    question = "我的健身追踪器表带应该怎样更换？"
    session_id = "demo-session"
    images = @()
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/v1/chat" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body
```

接口字段、状态码和响应示例见 [API 文档](docs/api.md)。

## 测试与评测

不消耗模型额度的完整回归：

```powershell
python -m scripts.run_regression
python -m scripts.evaluate_retrieval
python -m scripts.evaluate_agent
```

真实模型生成评测：

```powershell
python -m scripts.evaluate_generation
python -m scripts.run_regression --live
```

对已启动的服务执行并发基线：

```powershell
python -m scripts.benchmark_api `
  --base-url http://127.0.0.1:8000 `
  --requests 50 `
  --concurrency 5
```

报告写入 `artifacts/`，该目录不会提交到 Git。真实生成评测依赖外部供应商额度和稳定性，因此没有放入 CI 阻断链路。

## 部署

```powershell
Copy-Item .env.production.example .env
# 替换模型密钥、模型名称和 AGENT_API_TOKEN
docker compose up --build -d
docker compose ps
```

容器以非 root 用户运行，`/v1/ready` 作为健康检查，SQLite 运行数据和 Hugging Face 缓存使用命名卷持久化。当前模板固定单 Worker；水平扩容前需要将 SQLite 和进程内限流迁移到共享基础设施。

完整检查清单见 [部署说明](docs/deployment.md)。

## 项目结构

```text
app/                    Agent、API、检索、模型、存储、安全与前端
app/web/                客服运营工作台
data/normalized/        标准化手册知识
data/index/             知识快照与向量索引
data/evaluation/        检索、生成和 Agent 评测集
scripts/                数据处理、测试、评测和性能基线
docs/                   架构、API、部署、评测与简历讲解
```

## 文档导航

- [架构说明](docs/architecture.md)
- [API 文档](docs/api.md)
- [部署说明](docs/deployment.md)
- [评测说明](docs/evaluation.md)
- [简历与面试复盘指南](docs/resume-guide.md)
- [版本变更记录](CHANGELOG.md)

## 已知边界

- 当前限流器位于进程内，SQLite 也按单实例设计，因此生产模板固定单 Worker。
- 真实模型质量与延迟受供应商、模型版本和额度影响，需要在目标环境重新运行生成评测和压测。
- 工单系统是本项目内置闭环，真实企业接入时可替换为 CRM、Zendesk 或内部售后系统适配器。
- 当前接口采用非流式响应；长回答场景可在后续版本增加 SSE。

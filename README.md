# 多模态客服对话 Agent

一个基于 FastAPI、混合检索和视觉大模型的中文客服 Agent。它能够接收文字与图片，先从图片中提取产品线索，再结合本地产品手册知识库生成有依据的回答，并返回相关知识库图片。

## 核心能力

- 文字与最多 3 张图片的多模态问答
- 关键词检索、向量检索、产品过滤和相邻知识块扩展
- 视觉产品识别置信度策略：高置信度硬过滤、中置信度软加权、低置信度忽略
- SQLite 多轮会话记忆，支持服务重启后恢复
- 视觉预分析失败时自动降级为文字检索
- API Token 鉴权、请求体限制、图片真实性校验、限流和请求 ID
- `/health` 存活检查与 `/ready` 依赖就绪检查
- 结构化运行指标、统一回归测试、检索评测与真实生成评测
- Docker、Docker Compose 和 GitHub Actions 配置

## 处理流程

```text
文字/图片请求
  -> 输入安全校验
  -> 读取最近会话历史
  -> 图片视觉预分析（可降级）
  -> 置信度驱动的产品检索策略
  -> 关键词 + 向量混合检索
  -> 相邻知识块扩展与上下文截断
  -> 大模型生成结构化回答和图片选择
  -> 清洗内部资料标记
  -> 保存会话并返回答案、图片、参考资料和指标
```

更详细的设计见 [docs/architecture.md](docs/architecture.md)。

## Windows 快速启动

要求 Python 3.12。以下命令在项目根目录的 PowerShell 中执行。

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

默认配置使用 `mock` 文本模型，适合运行测试。要调用真实模型，请在 `.env` 中填写文本模型和视觉模型的名称、接口地址及密钥。

启动服务：

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

可访问：

- API 文档：`http://127.0.0.1:8000/docs`
- 存活检查：`http://127.0.0.1:8000/health`
- 就绪检查：`http://127.0.0.1:8000/ready`

## 调用接口

未开启鉴权时：

```powershell
$body = @{
    question = "我的健身追踪器表带应该怎样更换？"
    session_id = "demo-session"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/chat" `
    -ContentType "application/json" `
    -Body $body
```

开启鉴权后增加请求头：

```powershell
$headers = @{ Authorization = "Bearer 你的AGENT_API_TOKEN" }
```

图片需要放在 `images` 数组中，并使用完整 Data URL，例如 `data:image/jpeg;base64,...`。接口会验证 Base64、MIME、真实图片格式、单图大小、总大小和像素数量。

继续同一轮对话时复用 `session_id`。清空会话：

```powershell
Invoke-RestMethod `
    -Method Delete `
    -Uri "http://127.0.0.1:8000/sessions/demo-session" `
    -Headers $headers
```

## 测试与评测

普通回归全部使用 Mock，不消耗模型额度：

```powershell
python -m scripts.run_regression
```

追加真实多模态链路测试：

```powershell
python -m scripts.run_regression --live
```

检索评测会运行 13 个跨产品案例，并在失败时返回非零退出码：

```powershell
python -m scripts.evaluate_retrieval
```

真实生成评测会调用配置的文本模型：

```powershell
python -m scripts.evaluate_generation
python -m scripts.evaluate_generation --case-id drill_temperature
```

评测报告写入 `artifacts/evaluation/`，该目录不会提交到 Git。

## 知识库维护

标准化知识位于 `data/normalized/`，向量索引位于 `data/index/`。修改标准化知识后重建索引：

```powershell
python -m scripts.build_vector_index
python -m scripts.evaluate_retrieval
```

只有在检索评测通过后才应提交新索引。

## Docker 部署

```powershell
Copy-Item .env.production.example .env
# 编辑 .env，替换模型、密钥和 API Token 占位值
docker compose up --build -d
docker compose ps
```

Compose 使用单个 Uvicorn worker，并将 SQLite 会话数据库和 Hugging Face 模型缓存保存到命名卷。完整生产检查见 [docs/deployment.md](docs/deployment.md)。

## 目录说明

```text
app/                    API、检索、模型调用、会话和安全逻辑
data/normalized/        标准化知识记录
data/index/             向量和知识快照
data/evaluation/        检索与生成评测案例
scripts/                索引、测试、评测和数据处理脚本
docs/                   架构与部署说明
```

## 当前扩展边界

当前限流器保存在进程内，会话使用 SQLite，因此生产模板固定为单 worker。需要多实例水平扩容时，应将限流迁移到 Redis，并将会话存储迁移到共享数据库。

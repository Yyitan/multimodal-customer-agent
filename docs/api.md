# API 文档

## 约定

- 稳定接口前缀：`/v1`。
- JSON 编码：UTF-8。
- 开启鉴权后，除公开系统接口外均需 `Authorization: Bearer <token>`。
- 每个响应都包含 `X-Request-ID`；调用方也可以传入符合规则的 `X-Request-ID`。
- 旧的无版本路径仅用于兼容，已从 OpenAPI 隐藏，新客户端应使用 `/v1`。

## 系统接口

### `GET /v1/info`

返回服务名称、版本、API 前缀和能力列表。该接口公开，可用于部署检查。

### `GET /v1/health`

只表示 HTTP 进程存活，不检查模型、数据库和索引。

### `GET /v1/ready`

检查运行配置、会话数据库、工单数据库和知识索引。依赖不可用时返回 `503`，响应中包含错误列表。

### `GET /v1/metrics`

返回 Prometheus 文本格式指标，需要鉴权。

## Agent 对话

### `POST /v1/chat`

请求体：

```json
{
  "question": "请看图片，这个设备的表带怎样更换？",
  "images": ["data:image/jpeg;base64,..."],
  "session_id": "customer-001",
  "stream": false
}
```

约束：

- `question`：1 到配置上限字符，默认上限 4000。
- `images`：0 到 3 张，仅支持 JPEG、PNG、WebP Data URL。
- `session_id`：可选，1 到 128 位安全字符；不传时服务端生成。
- `stream`：当前只能为 `false`。

成功响应核心结构：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "...",
    "session_id": "customer-001",
    "request_id": "...",
    "timestamp": 1790496000,
    "references": [],
    "image_ids": [],
    "images": [],
    "agent": {
      "intent": "product_guidance",
      "action": "respond_with_knowledge",
      "confidence": 0.8,
      "handoff_required": false,
      "handoff_reason": null,
      "ticket_id": null
    },
    "metrics": {
      "retrieval_latency_ms": 35,
      "llm_latency_ms": 1200,
      "total_latency_ms": 1300,
      "llm_attempt_count": 1,
      "llm_retry_count": 0,
      "llm_circuit_state": "closed"
    }
  }
}
```

当请求需要人工处理时，`agent.handoff_required=true`，并返回 `ticket_id`。安全风险、业务操作和明确要求人工的请求会跳过最终模型调用。

常见状态码：

| 状态码 | 含义 |
| --- | --- |
| `200` | 请求成功，包括成功创建人工工单 |
| `401` | Token 缺失或错误 |
| `413` | 请求体超过限制 |
| `422` | 字段、Base64、MIME 或图片真实性校验失败 |
| `429` | 超过接口限流，响应包含 `Retry-After` |
| `503` | 配置错误、依赖不可用或模型调用失败 |

## 会话

### `DELETE /v1/sessions/{session_id}`

删除指定会话的历史。响应中的 `cleared` 表示该会话是否存在并被删除。

## 工单

### `GET /v1/tickets?status=open&limit=50`

按创建时间倒序查询工单。`status` 可为 `open`、`in_progress`、`resolved`、`closed`。

### `GET /v1/tickets/{ticket_id}`

读取单个工单。

### `PATCH /v1/tickets/{ticket_id}`

```json
{"status": "in_progress"}
```

## 反馈与运营

### `POST /v1/feedback`

```json
{
  "session_id": "customer-001",
  "request_id": "可选请求ID",
  "rating": 5,
  "comment": "步骤清晰"
}
```

评分范围为 1 到 5。

### `GET /v1/operations/summary`

返回各状态工单数、活跃工单、未结紧急工单、反馈数、平均评分和评分分布。

## 知识图片

### `GET /v1/knowledge-images/{image_id}`

返回服务端目录中已登记的知识图片。接口校验图片 ID 并阻止路径穿越，响应使用私有缓存策略。

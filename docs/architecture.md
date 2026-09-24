# 架构说明

## 模块职责

- `app/main.py`：FastAPI 路由、请求模型、鉴权、限流、日志和健康检查。
- `app/services/chat_service.py`：编排完整问答流水线。
- `app/vision_analyzer.py`：从用户图片提取产品、置信度、关键词和可见描述。
- `app/hybrid_retriever.py`：合并关键词检索与向量检索，并扩展同手册相邻知识块。
- `app/context_builder.py`：控制上下文长度并排序知识图片候选。
- `app/prompt_builder.py`：构造最终模型 Prompt。
- `app/llm_client.py`：统一模型供应商调用、结构化结果解析和回答清洗。
- `app/conversation_store.py`：SQLite 会话持久化和 LRU 会话淘汰。
- `app/input_security.py`：请求体、图片和会话 ID 校验。
- `app/request_control.py`：请求 ID 与滑动窗口限流。

## 一次请求的生命周期

1. `RequestIdMiddleware` 读取合法的 `X-Request-ID`，否则生成新 ID，并写回响应头。
2. `RequestBodyLimitMiddleware` 在 JSON 解析前限制 `/chat` 请求体大小。
3. Pydantic 校验问题、图片数量和 `session_id`；图片会被解码并验证真实格式、大小和像素。
4. API Token 校验通过后，限流器按 Token 哈希计数；未开启鉴权时按客户端 IP 计数。
5. 从 SQLite 读取最近若干轮会话，拼入检索问题和最终 Prompt。
6. 有图片时，视觉模型先返回 `product_hint`、`product_confidence`、关键词和描述。
7. 根据置信度选择检索策略：
   - `>= 0.8`：把产品名写入检索问题，触发产品硬过滤。
   - `>= 0.4` 且 `< 0.8`：不硬过滤，只对该产品知识加分。
   - `< 0.4`：忽略产品提示，避免错误识别污染检索。
8. 关键词结果和向量结果按 `chunk_id` 合并，计算混合分数，并补充同产品、同手册的相邻知识块。
9. 上下文构造器限制单块和总字符数，并生成相关图片候选。
10. 模型根据用户问题、对话历史和知识上下文生成回答，同时从候选中选择最多指定数量的图片。
11. 输出清洗器删除内部资料编号和图片 ID 文案；图片 ID 只通过结构化字段返回。
12. 成功回答写入 SQLite，并返回答案、会话 ID、参考资料、图片和完整指标。

## 降级与失败边界

- 视觉预分析失败：记录错误和耗时，使用原始文字问题继续检索。
- 无检索结果：不调用最终模型，返回要求补充产品或故障信息的提示。
- 上下文为空：不调用最终模型，返回资料不足提示。
- 模型超时、限流或连接失败：转换为 `LLMClientError`，HTTP 层返回 503。
- 配置、SQLite 或向量索引不可用：`/ready` 返回 503 和具体错误列表。

## 状态与扩容

SQLite 保存会话历史；进程内滑动窗口保存限流状态。二者都适合当前单实例部署。若部署多个实例，需要共享会话数据库和共享限流后端，否则不同实例之间的会话与计数不一致。

## 质量保障

- `scripts.run_regression`：Mock 单元和接口回归，不调用真实模型。
- `scripts.evaluate_retrieval`：跨产品检索召回、图片和产品范围评测。
- `scripts.evaluate_generation`：真实模型回答关键词和图片评测，单案例失败不会中断整份报告。
- `scripts.run_regression --live`：上传真实图片的完整多模态链路。

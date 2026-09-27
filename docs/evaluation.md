# 评测说明

## 质量门禁

项目把质量拆成四层，避免只看一条主观回答：

1. 回归测试：代码行为、安全边界、接口契约和降级逻辑。
2. 检索评测：目标产品、知识图片召回和跨产品污染。
3. Agent 评测：意图、动作和转人工策略。
4. 生成评测：真实模型回答关键词和知识图片选择。

## 2026-09-27 已验证结果

| 评测 | 数据规模 | 结果 |
| --- | ---: | --- |
| Mock 回归 | 20 组测试模块 | 20/20 通过 |
| 检索评测 | 13 个跨产品案例 | 13/13 通过，100% |
| Agent 路由 | 17 个策略案例 | 17/17 通过 |
| Agent 指标 | 意图、动作、转人工 | 准确率 100%，转人工 F1 100% |
| HTTP 性能 | 50 请求，并发 5 | 0% 错误，P95 177.150 ms，31.744 req/s |

性能测试使用本地 Mock 文本模型、本地 BGE Embedding 和本地 SQLite。首次模型加载由预热阶段排除，结果反映热服务状态下的 API、检索和存储基线。

## 回归测试

```powershell
python -m scripts.run_regression
```

覆盖：

- Agent 前置路由和证据复核。
- API 契约、会话、工单、反馈和运营摘要。
- Token 鉴权、限流、请求 ID 和输入安全。
- SQLite 持久化与就绪检查。
- 模型重试、退避和熔断。
- 视觉失败降级、产品置信度策略和输出清洗。
- 知识图片、Web 工作台资源和性能统计工具。

`--live` 会追加真实图片和真实视觉模型调用：

```powershell
python -m scripts.run_regression --live
```

## 检索评测

```powershell
python -m scripts.evaluate_retrieval
```

13 个案例覆盖电钻、健身追踪器、空气净化器、人体工学椅、VR 头显、儿童电动摩托车、温控器、水泵等产品。一个案例通过需要：

- 预期知识图片全部出现在上下文候选中。
- 上下文没有混入错误产品记录。

## Agent 评测

```powershell
python -m scripts.evaluate_agent
```

17 个案例覆盖：

- 冒烟、漏电、爆炸和人身伤害等安全风险。
- 明确转人工和投诉。
- 退款、发票、订单和维修等业务系统操作。
- 故障排查与产品操作知识请求。
- 检索证据不足后的人工升级。

报告给出案例通过率、意图准确率、动作准确率、转人工 Precision、Recall 和 F1。

## 真实生成评测

```powershell
python -m scripts.evaluate_generation
```

生成评测共有 5 个案例，会实际调用配置的文本模型，并检查必要语义和预期知识图片。外部供应商的 429、超时和余额不足会记录到报告并返回非零退出码。

本次发布环境的真实供应商曾返回 429 额度错误，因此发布门禁不把该外部失败伪装成代码通过。获得有效额度后应在目标模型上重新运行并保存报告。

## 性能基线

```powershell
python -m scripts.benchmark_api `
  --base-url http://127.0.0.1:8000 `
  --requests 50 `
  --concurrency 5 `
  --warmup 3 `
  --max-p95-ms 5000 `
  --max-error-rate 0
```

脚本输出请求数、成功数、错误率、吞吐量、P50/P95/P99/最大时延和状态码分布，并在阈值不满足时返回非零退出码。

## 报告

机器可读报告写入：

- `artifacts/evaluation/retrieval.json`
- `artifacts/evaluation/agent.json`
- `artifacts/evaluation/generation.json`
- `artifacts/benchmark/api.json`

`artifacts/` 被 Git 忽略，CI 和发布系统应把它作为构建产物保存，而不是长期提交动态结果。

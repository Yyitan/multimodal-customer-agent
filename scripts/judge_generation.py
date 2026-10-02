# -*- coding: utf-8 -*-
"""LLM-as-judge 生成质量评审。

读取 evaluate_generation 产出的报告，用大模型按四维度 rubric 对每个
案例的回答打分：忠实性 / 相关性 / 完整性 / 安全性（各 0-2 分），
并给出整体 pass/fail 与理由。

动机：关键词判分对"语义正确但措辞不同"的回答会误判
（如 vr_headset_child_age 回答正确但缺少"安全/风险/警告"字样），
judge 用于补充语义层质量信号。judge 结论不替代门禁，作为
参考指标上报，并与关键词门禁、人工标注对比一致率。

用法：
    python -m scripts.judge_generation \
        --report artifacts/evaluation/generation.json
"""
import argparse
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = PROJECT_ROOT / "artifacts" / "evaluation" / "generation.json"
HUMAN_LABELS_PATH = PROJECT_ROOT / "data" / "evaluation" / "generation_human_labels.json"

JUDGE_SYSTEM_PROMPT = """你是严格的产品客服回答质量评审员。你会看到用户问题、
客服回答和作为回答依据的知识库参考资料片段。

请按以下四个维度各打 0-2 分：
- faithfulness 忠实性：回答是否完全由参考资料支撑，没有编造资料外的型号、参数、步骤或结论（0=大量编造，1=少量存疑，2=完全有据）
- relevance 相关性：回答是否针对用户问题本身（0=答非所问，1=部分相关，2=直接回答了问题）
- completeness 完整性：关键步骤、注意事项、安全警告是否覆盖（0=遗漏关键信息，1=覆盖主要信息，2=完整且保留了安全警告）
- safety 安全性：如资料中存在安全警告，回答是否保留并优先提醒（0=丢弃警告，1=提及但未突出，2=无安全相关内容或正确突出警告）

最后给出 overall："pass" 当且仅当四个维度都不低于 1 分且忠实性为 2 分；
否则 "fail"。reason 用一句话说明最关键的判断依据。

只返回合法 JSON，不要使用 Markdown 代码块：
{"faithfulness":0,"relevance":0,"completeness":0,"safety":0,"overall":"pass","reason":"..."}
"""


def build_judge_user_prompt(result: dict[str, Any]) -> str:
    references = result.get("references") or []
    reference_text = "\n\n".join(
        f"[片段 {reference['chunk_id']} | 产品: {reference['product']}]\n"
        f"{reference['text']}"
        for reference in references
    ) or "（无参考资料）"
    return f"""用户问题：
{result['question']}

客服回答：
{result['answer'] or '（空回答）'}

知识库参考资料片段：
{reference_text}

请按系统提示的 rubric 打分，只返回合法 JSON。"""


def parse_judge_json(content: str) -> dict[str, Any] | None:
    """解析评审输出，兼容模型偶尔包裹的 Markdown 代码块。"""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def judge_result(
    client: OpenAI,
    settings: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    request_options: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(result)},
        ],
        "max_tokens": 500,
        "temperature": 0,
        # 与主链路一致：保证模型返回合法 JSON
        "response_format": {"type": "json_object"},
    }
    if settings.llm_thinking is not None:
        # 智谱 GLM 通过 extra_body 关闭思考模式，否则思考耗尽输出额度
        request_options["extra_body"] = {"thinking": {"type": settings.llm_thinking}}
    response = client.chat.completions.create(**request_options)
    content = response.choices[0].message.content or ""
    scores = parse_judge_json(content)
    return {
        "case_id": result["case_id"],
        "judge_raw": content[:500],
        "judge_scores": scores,
        "judge_tokens": response.usage.total_tokens if response.usage else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge 生成质量评审。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluation" / "generation_judged.json",
    )
    args = parser.parse_args()

    with args.report.open("r", encoding="utf-8") as file:
        report = json.load(file)

    human_labels: dict[str, dict[str, Any]] = {}
    if HUMAN_LABELS_PATH.exists():
        with HUMAN_LABELS_PATH.open("r", encoding="utf-8") as file:
            labels = json.load(file)
        human_labels = {label["case_id"]: label for label in labels}

    settings = get_settings()
    client = OpenAI(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=str(settings.llm_base_url),
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )

    judged: list[dict[str, Any]] = []
    for result in report["results"]:
        if result.get("error"):
            judged.append(
                {
                    "case_id": result["case_id"],
                    "judge_scores": None,
                    "skipped": "调用失败案例不评审",
                }
            )
            print(f"跳过|{result['case_id']}（模型调用失败）")
            continue

        judged_result = judge_result(client, settings, result)
        scores = judged_result["judge_scores"]
        gate_passed = bool(result["passed"])
        label = human_labels.get(result["case_id"], {})
        judged_result["keyword_gate_passed"] = gate_passed
        judged_result["human_passed"] = label.get("human_passed")
        judged_result["human_note"] = label.get("note")
        judged.append(judged_result)

        verdict = scores.get("overall") if scores else "parse_error"
        flag = ""
        if scores and scores.get("overall") == "pass" and not gate_passed:
            flag = " <- judge与门禁不一致(门禁判失败)"
        if scores and scores.get("overall") == "fail" and gate_passed:
            flag = " <- judge与门禁不一致(门禁判通过)"
        print(
            f"{verdict}|{result['case_id']}|"
            f"f={scores.get('faithfulness') if scores else '-'} "
            f"r={scores.get('relevance') if scores else '-'} "
            f"c={scores.get('completeness') if scores else '-'} "
            f"s={scores.get('safety') if scores else '-'}{flag}"
        )

    valid = [item for item in judged if item.get("judge_scores")]
    judge_pass = sum(
        1 for item in valid if item["judge_scores"].get("overall") == "pass"
    )
    comparable = [item for item in valid if item.get("human_passed") is not None]
    human_agree = sum(
        1
        for item in comparable
        if (item["judge_scores"].get("overall") == "pass") == item["human_passed"]
    )
    gate_comparable = valid
    gate_agree = sum(
        1
        for item in gate_comparable
        if (item["judge_scores"].get("overall") == "pass")
        == item["keyword_gate_passed"]
    )
    disagreements = [
        {
            "case_id": item["case_id"],
            "judge": item["judge_scores"].get("overall"),
            "keyword_gate": item["keyword_gate_passed"],
            "human": item.get("human_passed"),
            "reason": item["judge_scores"].get("reason"),
        }
        for item in valid
        if (item["judge_scores"].get("overall") == "pass")
        != item["keyword_gate_passed"]
    ]

    summary = {
        "judged_count": len(valid),
        "judge_pass_count": judge_pass,
        "human_labeled_count": len(comparable),
        "human_agreement": round(human_agree / len(comparable), 4)
        if comparable
        else None,
        "keyword_gate_agreement": round(gate_agree / len(gate_comparable), 4)
        if gate_comparable
        else None,
        "disagreements": disagreements,
        "judge_total_tokens": sum(
            item.get("judge_tokens") or 0 for item in valid
        ),
    }
    output = {
        "judgement_type": "generation_llm_judge",
        "source_report": str(args.report),
        "summary": summary,
        "results": judged,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("\n====LLM-as-judge 总结====")
    print("评审案例数：", summary["judged_count"])
    print("judge 判定通过：", summary["judge_pass_count"])
    print("与人工标注一致率：", summary["human_agreement"])
    print("与关键词门禁一致率：", summary["keyword_gate_agreement"])
    print("judge 消耗 token：", summary["judge_total_tokens"])
    print("报告路径：", args.output)


if __name__ == "__main__":
    main()

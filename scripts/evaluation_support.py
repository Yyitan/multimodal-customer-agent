import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate an evaluation case file."""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(f"评测文件必须包含非空案例列表：{path}")

    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("每条评测案例必须是 JSON 对象。")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("每条评测案例必须包含非空 case_id。")
        if case_id in case_ids:
            raise ValueError(f"评测案例 ID 重复：{case_id}")
        case_ids.add(case_id)

    return cases


def select_cases(
    cases: list[dict[str, Any]],
    requested_ids: list[str],
) -> list[dict[str, Any]]:
    """Select requested cases while preserving file order."""
    if not requested_ids:
        return cases

    requested = set(requested_ids)
    available = {str(case["case_id"]) for case in cases}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"未知评测案例：{', '.join(missing)}")
    return [case for case in cases if case["case_id"] in requested]


def write_report(
    path: Path,
    evaluation_type: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist a machine-readable evaluation report."""
    passed_count = sum(bool(result["passed"]) for result in results)
    total_count = len(results)
    report = {
        "evaluation_type": evaluation_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_count": total_count,
            "passed_count": passed_count,
            "failed_count": total_count - passed_count,
            "pass_rate": passed_count / total_count if total_count else 0.0,
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report

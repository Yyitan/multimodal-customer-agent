import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.evaluate_generation import (
    check_keyword_groups,
    failed_case_result,
)
from scripts.evaluate_retrieval import collect_image_ranks
from scripts.evaluation_support import (
    load_cases,
    select_cases,
    write_report,
)


def main() -> None:
    checks: dict[str, bool] = {}

    image_ranks = collect_image_ranks(
        [
            {"image_ids": ["image-a", "image-b"]},
            {"image_ids": ["image-b", "image-c"]},
        ]
    )
    checks["图片排名保留首次出现位置"] = image_ranks == {
        "image-a": 1,
        "image-b": 1,
        "image-c": 2,
    }

    missing_groups = check_keyword_groups(
        "电池过热时暂停充电，冷却后自动恢复。",
        [["高温", "过热"], ["暂停"], ["故障"]],
    )
    checks["关键词组支持组内任一匹配"] = missing_groups == [["故障"]]

    failed_result = failed_case_result(
        {
            "case_id": "failed-case",
            "question": "测试问题",
            "must_include_any": [["测试"]],
            "expected_image_ids": ["image-a"],
        },
        RuntimeError("模拟外部服务失败"),
    )
    checks["外部调用失败会转换为可报告结果"] = (
        failed_result["passed"] is False
        and failed_result["error"]["type"] == "RuntimeError"
        and failed_result["missing_image_ids"] == ["image-a"]
    )

    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        cases_path = root / "cases.json"
        report_path = root / "report.json"
        cases_path.write_text(
            json.dumps(
                [
                    {"case_id": "case-a", "question": "A"},
                    {"case_id": "case-b", "question": "B"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        cases = load_cases(cases_path)
        selected = select_cases(cases, ["case-b"])
        checks["可按案例ID选择评测子集"] = (
            len(selected) == 1 and selected[0]["case_id"] == "case-b"
        )

        report = write_report(
            report_path,
            evaluation_type="test",
            results=[
                {"case_id": "case-a", "passed": True},
                {"case_id": "case-b", "passed": False},
            ],
        )
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        checks["评测报告写入汇总和明细"] = (
            report["summary"]["pass_rate"] == 0.5
            and persisted["summary"]["failed_count"] == 1
            and len(persisted["results"]) == 2
        )

        try:
            select_cases(cases, ["missing-case"])
        except ValueError:
            unknown_case_rejected = True
        else:
            unknown_case_rejected = False
        checks["未知案例ID会被拒绝"] = unknown_case_rejected

    print("====评测工具测试====")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}|{name}")

    if not all(checks.values()):
        raise SystemExit(1)

    print("\n评测工具测试：全部通过")


if __name__ == "__main__":
    main()

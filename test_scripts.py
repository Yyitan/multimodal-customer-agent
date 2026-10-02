# -*- coding: utf-8 -*-
"""pytest 入口：把 scripts/test_*.py 的 main() 形式测试包装成 pytest 用例。

历史测试按"模块 + main() 函数"组织（`python -m scripts.run_regression`）。
本文件让同一批测试可以直接用 `pytest -q` 运行，而不必重写断言：
每个模块收集为一个用例，main() 抛 SystemExit/异常即失败。

排除项（不进 pytest 自动化）：
- test_llm_connection.py 需要真实模型连接
- test_multimodal_chain.py / test_vision_analyzer.py 依赖仓库外的
  赛题三原图目录（未随仓库分发），仅本机可运行；run_regression
  同样只在 --live 时加载 multimodal_chain
"""
import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent

EXCLUDED_MODULES = {
    "test_llm_connection.py",
    "test_multimodal_chain.py",
    "test_vision_analyzer.py",
}

TEST_MODULES = sorted(
    path.stem
    for path in (PROJECT_ROOT / "scripts").glob("test_*.py")
    if path.name not in EXCLUDED_MODULES
)


@pytest.mark.parametrize("module_name", TEST_MODULES)
def test_script_module(module_name: str) -> None:
    module = importlib.import_module(f"scripts.{module_name}")
    module.main()

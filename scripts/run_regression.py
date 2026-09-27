# `argparse`：命令行参数解析，
import argparse

# 捕获异常后打印完整报错堆栈，方便定位哪里崩了
import traceback

# 类型别名 `TestFunction`：代表无入参、无返回的测试函数（就是每个测试脚本里的 `main()`）
from collections.abc import Callable

TestFunction = Callable[[], None]


def load_tests(
    include_live: bool,
) -> list[tuple[str, TestFunction]]:
    """加载回归测试列表。"""

    # 延迟导入：`from scripts.xxx import main` 写在函数内部，不是文件顶部。
    # 好处：不加载不需要的测试模块，加快启动；而且 `--live` 不开时，不会加载调用真实 LLM 的 `test_multimodal_chain`
    from scripts.test_agent_policy import main as test_agent_policy
    from scripts.test_api import main as test_api
    from scripts.test_auth import main as test_auth
    from scripts.test_conversation import main as test_conversation
    from scripts.test_evaluation import main as test_evaluation
    from scripts.test_output_sanitizer import (
        main as test_output_sanitizer,
    )
    from scripts.test_observability import main as test_observability
    from scripts.test_model_resilience import main as test_model_resilience
    from scripts.test_knowledge_images import main as test_knowledge_images
    from scripts.test_web_console import main as test_web_console
    from scripts.test_benchmark import main as test_benchmark
    from scripts.test_image_ranking import (
        main as test_image_ranking,
    )
    from scripts.test_input_security import (
        main as test_input_security,
    )
    from scripts.test_product_confidence import (
        main as test_product_confidence,
    )
    from scripts.test_product_strategy_chain import (
        main as test_product_strategy_chain,
    )
    from scripts.test_readiness import main as test_readiness
    from scripts.test_request_control import (
        main as test_request_control,
    )
    from scripts.test_support_store import main as test_support_store
    from scripts.test_vision_fallback import (
        main as test_vision_fallback,
    )

    # `tests` 是列表，每一项是元组：`(测试名称字符串, 测试函数)`
    tests = [
        ("Agent决策测试", test_agent_policy),
        ("接口层测试", test_api),
        ("API鉴权测试", test_auth),
        ("多轮会话测试", test_conversation),
        ("评测工具测试", test_evaluation),
        ("可观测性测试", test_observability),
        ("模型韧性测试", test_model_resilience),
        ("知识图片目录测试", test_knowledge_images),
        ("Web工作台测试", test_web_console),
        ("性能基线工具测试", test_benchmark),
        ("回答清洗测试", test_output_sanitizer),
        ("图片候选排序测试", test_image_ranking),
        ("输入安全测试", test_input_security),
        ("运行就绪测试", test_readiness),
        ("请求控制测试", test_request_control),
        ("客服闭环存储测试", test_support_store),
        ("产品置信度测试", test_product_confidence),
        ("产品策略业务链测试", test_product_strategy_chain),
        ("视觉降级测试", test_vision_fallback),
    ]

    # `include_live` 就是命令行的 `--live`：不加 `--live`：只跑 mock 单元 / 集成测试;加上 `--live`：追加执行 `test_multimodal_chain`，调用真实视觉模型做端到端全链路测试
    if include_live:
        from scripts.test_multimodal_chain import (
            main as test_multimodal_chain,
        )
        tests.append(
            ("真实多模态链路测试", test_multimodal_chain)
        )

    return tests

# 调用传入的测试函数 `test_function()`
def run_one_test(
    name: str,
    test_function: TestFunction,
) -> bool:
    print("\n" + "=" * 60)
    print(f"开始测试：{name}")

    try:
        test_function()
    # 捕获两种情况：
    # `SystemExit`：很多测试脚本失败时会 `raise SystemExit(1)`，专门处理这种退出码；退出码 0 代表成功，非 0 代表失败
    except SystemExit as error:
        passed = error.code in (None, 0)
        if not passed:
            print(f"失败|{name}|测试主动退出")
        return passed
    # 普通 Exception：代码报错（断言失败、变量不存在、接口 500 等），打印完整堆栈，返回`False`标记失败
    except Exception as error:
        print(f"失败|{name}|{error}")
        traceback.print_exc()
        return False
    # 正常跑完无异常 → 返回 True，标记【通过】
    print(f"通过|{name}")
    return True

def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行项目回归测试。",
    )
    # 解析命令行参数，读取 `--live`
    parser.add_argument(
        "--live",
        action="store_true",
        help="额外运行真实视觉模型多模态测试。",
    )
    args = parser.parse_args()

    tests = load_tests(
        include_live=args.live,
    )

    failed_tests: list[str] = []

    # 拿到全部测试列表，循环逐个执行
    for name, test_function in tests:
        passed = run_one_test(
            name=name,
            test_function=test_function,
        )
        # 收集所有失败的测试名称存入 `failed_tests`
        if not passed:
            failed_tests.append(name)

    # 循环跑完后输出汇总统计表
    print("\n" + "=" * 60)
    print("回归测试总结")
    print(f"测试总数：{len(tests)}")
    print(f"通过数量：{len(tests) - len(failed_tests)}")
    print(f"失败数量：{len(failed_tests)}")
    # 如果存在失败用例：打印失败清单，`raise SystemExit(1)` 退出如果存在失败用例：打印失败清单，`raise SystemExit(1)` 退出
    if failed_tests:
        print("失败测试：")
        for name in failed_tests:
            print(f"- {name}")
        raise SystemExit(1)

    print("回归测试：全部通过")


if __name__ == "__main__":
    main()

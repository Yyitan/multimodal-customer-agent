# `lru_cache`：函数缓存装饰器，调用一次之后结果缓存，不会重复加载配置。
from functools import lru_cache
from pathlib import Path

# `Literal`：字面量类型，限定变量只能取指定字符串，写错值直接报错。
from typing import Literal

# `Field`：pydantic 字段，给配置项设置默认值、数值校验范围。`SecretStr`：密码 / 密钥专用类型，打印的时候不会明文泄露 API‑key。
from pydantic import Field, SecretStr

# `BaseSettings`：pydantic‑settings 基类，专门做环境变量 /`.env`配置加载。`SettingsConfigDict`：配置加载行为的配置字典。
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 定义配置类，继承`BaseSettings`。所有写在这里的字段，自动支持从环境变量 / `.env` 文件读取。
class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    #  运行环境，只能三选一：`development`开发 /`test`测试 /`production`生产，默认开发模式。
    # `Literal`限制：如果`.env`写错字符串，程序直接启动报错，防止非法配置。
    environment: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    log_level: str = "INFO"

    # 大模型后端类型
    llm_provider: Literal[
        "mock",
        "openai",
        "openai_compatible",
    ] = "mock"

    # 大模型模型名字
    llm_model: str = "mock"

    # 大模型接口地址
    llm_base_url: str | None = None
    # 大模型 API 密钥
    llm_api_key: SecretStr | None = None
    
    # 视觉模型供应商
    # 当前视觉模型使用阿里云百炼的OpenAI兼容接口.
    llm_vision_provider:Literal[
        "openai_compatible",
    ]="openai_compatible"
    
    # 视觉模型名称
    llm_vision_model:str|None=None
    
    # 视觉模型接口地址
    llm_vision_base_url:str|None=None
    
    # 视觉模型APIKey.
    llm_vision_api_key:SecretStr|None=None
    
    # 是否开启视觉模型的思考模式.
    # 当前先关闭,因为需要稳定返回结构化JSON.
    llm_vision_thinking:bool=False
    
    # LLM 接口请求超时时间，单位秒。默认 30 秒；greater than，必须大于 0，less equal，最大 120 秒
    llm_timeout_seconds:float=Field(
        default=30.0,
        gt=0,
        le=120,
    )
    
    # `disabled`：关闭深度思考智谱。 GLM-4.5 的深度思考（thinking）就是模型内部的推理链，开启会消耗更多 token，推理更慢
    llm_thinking:Literal["enabled","disabled"]|None="disabled"
    # 默认最多选 3 张图片，大于等于1，小于等于5
    max_answer_images:int=Field(default=3,ge=1,le=5)
    
    # 限制模型最对生成多少token，防止模型回答无限变长、控制接口响应时间，控制真实API调用费用。
    llm_max_output_tokens:int=Field(
        # 接下来要求模型输出严格 JSON 结构化结果，所以加长一点。
        default=1200,
        ge=64,
        le=8192,
    )
    # 语义检索返回多少条候选切片，`ge=1`大于等于 1，`le=100`最大 100
    semantic_top_k:int=Field(
        default=20,
        ge=1,
        le=100,
    )
    # 关键词检索返回候选数量。
    seed_top_k:int=Field(
        default=10,
        ge=1,
        le=50,
    )
    # 相邻窗口扩展参数,0 代表关闭窗口扩展，最大 5。
    neighbor_window_size: int = Field(
        default=1,
        ge=0,
        le=5,
    )
    
    # 混合检索最终输出最多返回多少条记录
    max_retrieval_results: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    # 全局上下文总字符上限
    max_context_chars: int = Field(
        default=10000,
        ge=1000,
        le=50000,
    )
    
    # 单条知识块最大字符截断
    max_text_chars: int = Field(
        default=1200,
        ge=100,
        le=10000,
    )
    
    # API 接口鉴权开关；后端对外提供接口的时候，开启鉴权，请求头带上 token 访问。本地调试默认关闭。
    auth_enabled:bool=False
    api_token:SecretStr|None=None
    
    # `model_config`：pydantic‑settings 的核心配置
    model_config=SettingsConfigDict(
        # 加载项目根目录下的`.env`文件。
        env_file=PROJECT_ROOT/".env",
        env_file_encoding="utf-8",
        # 环境变量前缀:类里面字段叫`llm_provider`，那么`.env`或者系统环境变量名字要写`AGENT_LLM_PROVIDER=mock`。
        env_prefix="AGENT_",
        # 大小写不敏感，环境变量大小写无所谓
        case_sensitive=False,
        # `.env`里面出现配置类没有定义的多余变量，直接忽略，不报错。
        extra="ignore",
    )

    # 单个会话最多保存多少轮问答历史
    conversation_max_turns:int=Field(
        default=6,
        ge=1,
        le=20,
    )

    # 全局最多同时保存多少个独立session会话
    conversation_max_sessions:int=Field(
        default=1000,
        ge=1,
        le=10000,
    )

# 缓存装饰器 
# - 第一次调用`get_settings()`，实例化`Settings()`，读取`.env`和环境变量，生成配置对象。
# - 后续所有地方再次调用`get_settings()`，**直接返回缓存好的对象，不会重复读取解析.env 文件**。
@lru_cache
def get_settings()->Settings:
    """Load and cache one Settings instance."""
    return Settings()
    
    
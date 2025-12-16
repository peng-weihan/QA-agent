"""简化的模型接口，基于 SWE-agent 的模型系统"""
from __future__ import annotations

import os
import litellm
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, Field, SecretStr

litellm.suppress_debug_info = True


class ModelConfig(BaseModel):
    """模型配置"""
    name: str = Field(description="模型名称，如 gpt-4o, claude-sonnet-4-20250514")
    api_key: SecretStr | None = Field(default=None, description="API key")
    api_base: str | None = Field(default=None, description="API base URL")
    temperature: float = 0.0
    max_tokens: int | None = None


class AbstractModel(ABC):
    """抽象模型接口"""
    
    @abstractmethod
    def forward(self, history: list[dict[str, str]]) -> tuple[str, Dict[str, Any]]:
        """调用模型
        
        Args:
            history: 对话历史，格式为 [{"role": "user", "content": "..."}, ...]
            
        Returns:
            (response_text, stats): 响应文本和统计信息（latency, input_tokens, output_tokens）
        """
        pass


class LiteLLMModel(AbstractModel):
    """基于 LiteLLM 的模型实现"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.name
        
        # 设置 API key
        if config.api_key:
            # 根据模型名称设置对应的环境变量
            if "gpt" in config.name.lower() or "openai" in config.name.lower():
                os.environ["OPENAI_API_KEY"] = config.api_key.get_secret_value()
            elif "claude" in config.name.lower() or "anthropic" in config.name.lower():
                os.environ["ANTHROPIC_API_KEY"] = config.api_key.get_secret_value()
    
    def forward(self, history: list[dict[str, str]]) -> tuple[str, Dict[str, Any]]:
        """调用模型
        
        Returns:
            (response_text, stats): 响应文本和统计信息
        """
        import time
        start_time = time.time()
        
        messages = []
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        
        response = litellm.completion(**kwargs)
        
        # 计算延迟
        latency = time.time() - start_time
        
        # 提取token统计信息
        usage = getattr(response, "usage", None)
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        stats = {
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        
        return response.choices[0].message.content, stats


def get_model(config: ModelConfig) -> AbstractModel:
    """根据配置获取模型实例"""
    return LiteLLMModel(config)


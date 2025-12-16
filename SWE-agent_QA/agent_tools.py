"""简化的工具处理系统，基于 SWE-agent 的工具框架"""
import json
import re
import logging
from pathlib import Path
from typing import Any, Optional, Dict
from pydantic import BaseModel, Field

from local_env import LocalEnv

logger = logging.getLogger(__name__)


class ToolFilterConfig(BaseModel):
    """工具过滤配置"""
    blocklist: list[str] = Field(default_factory=lambda: [
        "vim", "vi", "emacs", "nano", "python", "python3", "bash", "sh"
    ])


class ToolConfig(BaseModel):
    """工具配置"""
    filter: ToolFilterConfig = Field(default_factory=ToolFilterConfig)
    env_variables: Dict[str, Any] = Field(default_factory=lambda: {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
    })
    enable_bash_tool: bool = True


class ToolHandler:
    """工具处理器"""
    
    def __init__(self, config: ToolConfig, env: LocalEnv):
        self.config = config
        self.env = env
    
    def parse_action(self, response: str) -> Optional[Dict[str, Any]]:
        """从模型响应中解析工具调用
        
        支持两种格式：
        1. Function calling (JSON)
        2. 简单的命令格式
        """
        # 尝试解析完整的 JSON 对象（可能包含嵌套）
        try:
            # 先尝试直接解析整个响应
            action = json.loads(response.strip())
            if isinstance(action, dict) and "name" in action:
                return action
        except:
            pass
        
        # 尝试解析 JSON 对象（更宽松的匹配）
        try:
            # 查找 JSON 对象，支持多行和嵌套
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"name"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                action = json.loads(json_match.group())
                if isinstance(action, dict) and "name" in action:
                    return action
        except:
            pass
        
        # 尝试解析代码块中的 JSON
        try:
            json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_block:
                action = json.loads(json_block.group(1))
                if isinstance(action, dict) and "name" in action:
                    return action
        except:
            pass
        
        # 尝试解析简单的命令格式
        # 格式: bash: "command"
        bash_match = re.search(r'bash:\s*["\']([^"\']+)["\']', response)
        if bash_match:
            return {
                "name": "bash",
                "arguments": {"command": bash_match.group(1)}
            }
        
        # 尝试解析 grep, cat, ls 等命令
        for cmd in ["grep", "cat", "ls", "find", "tree"]:
            pattern = rf'{cmd}:\s*["\']([^"\']+)["\']'
            match = re.search(pattern, response)
            if match:
                return {
                    "name": cmd,
                    "arguments": {"path": match.group(1)}
                }
        
        return None
    
    def execute_action(self, action: Dict[str, Any], env: LocalEnv) -> str:
        """执行工具调用"""
        action_name = action.get("name", "")
        arguments = action.get("arguments", {})
        
        # 检查是否被阻止
        if self._is_blocked(action_name, arguments):
            return f"Error: Command '{action_name}' is blocked by the environment."
        
        try:
            if action_name == "bash":
                command = arguments.get("command", "")
                return env.communicate(command, check="warn")
            
            elif action_name == "read_file" or action_name == "cat":
                path = arguments.get("path", "")
                return env.read_file(path)
            
            elif action_name == "grep":
                pattern = arguments.get("pattern", "")
                path = arguments.get("path", ".")
                command = f"grep -r '{pattern}' {path}"
                return env.communicate(command, check="warn")
            
            elif action_name == "ls":
                path = arguments.get("path", ".")
                return env.communicate(f"ls -la {path}", check="warn")
            
            elif action_name == "find":
                pattern = arguments.get("pattern", "*")
                path = arguments.get("path", ".")
                return env.communicate(f"find {path} -name '{pattern}'", check="warn")
            
            else:
                return f"Error: Unknown action '{action_name}'"
        
        except Exception as e:
            return f"Error executing action: {str(e)}"
    
    def _is_blocked(self, action_name: str, arguments: Dict[str, Any]) -> bool:
        """检查动作是否被阻止"""
        blocklist = self.config.filter.blocklist
        
        # 检查完整命令
        if action_name in blocklist:
            return True
        
        # 检查 bash 命令中的阻止项
        if action_name == "bash":
            command = arguments.get("command", "")
            for blocked in blocklist:
                if command.startswith(blocked):
                    return True
        
        return False


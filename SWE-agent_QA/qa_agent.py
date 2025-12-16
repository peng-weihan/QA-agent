"""轻量化的 QA Agent，基于 SWE-agent 的核心框架"""
import logging
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field

from local_env import LocalEnv
from agent_models import ModelConfig, get_model
from agent_tools import ToolHandler, ToolConfig
from problem_statement import TextProblemStatement, ProblemStatement

logger = logging.getLogger(__name__)


class QAAgentConfig(BaseModel):
    """QA Agent 配置"""
    model: ModelConfig = Field(description="模型配置")
    tools: ToolConfig = Field(default_factory=ToolConfig, description="工具配置")
    max_steps: int = 10
    """最大执行步数（轮数上限）"""
    max_observation_length: int = 100_000
    """最大观察长度"""


class QAAgent:
    """轻量化的 QA Agent"""
    
    def __init__(
        self,
        config: QAAgentConfig,
        env: LocalEnv,
        problem_statement: ProblemStatement,
    ):
        self.config = config
        self.env = env
        self.problem_statement = problem_statement
        self.model = get_model(config.model)
        self.tool_handler = ToolHandler(config.tools, env)
        self.history = []
        self.step_count = 0
        self.stats = {
            "total_latency": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }
        
    def run(self) -> dict[str, Any]:
        """运行 QA 任务"""
        logger.info(f"Starting QA task: {self.problem_statement.id}")
        
        # 初始化环境变量
        if self.config.tools.env_variables:
            self.env.set_env_variables(self.config.tools.env_variables)
        
        # 构建初始提示
        system_prompt = self._build_system_prompt()
        instance_prompt = self._build_instance_prompt()
        
        # 添加到历史
        self.history.append({"role": "system", "content": system_prompt})
        self.history.append({"role": "user", "content": instance_prompt})
        
        # 主循环
        while self.step_count < self.config.max_steps:
            self.step_count += 1
            logger.info(f"Step {self.step_count}/{self.config.max_steps}")
            
            # 接近上限时给出警告
            if self.step_count >= self.config.max_steps - 2:
                logger.warning(f"接近轮数上限 ({self.config.max_steps})，剩余 {self.config.max_steps - self.step_count} 步")
            
            # 调用模型
            response, step_stats = self.model.forward(self.history)
            logger.debug(f"Model response: {response[:200]}...")
            
            # 累计统计信息
            self.stats["total_latency"] += step_stats["latency"]
            self.stats["total_input_tokens"] += step_stats["input_tokens"]
            self.stats["total_output_tokens"] += step_stats["output_tokens"]
            
            # 解析工具调用
            action = self.tool_handler.parse_action(response)
            
            if action is None:
                # 检查是否包含明确的结束信号
                if any(keyword in response.lower() for keyword in ["final answer", "answer:", "结论", "总结"]):
                    logger.info("Found final answer signal, treating as final answer")
                    self.history.append({"role": "assistant", "content": response})
                    break
                # 如果没有工具调用，提示模型使用工具
                logger.warning("No action parsed from response, prompting to use tools")
                self.history.append({"role": "assistant", "content": response})
                self.history.append({
                    "role": "user",
                    "content": "Please use the available tools to explore the repository. Use JSON format: {\"name\": \"tool_name\", \"arguments\": {...}}. Remember: Your final answer must be in English."
                })
                continue
            
            # 执行工具
            observation = self.tool_handler.execute_action(action, self.env)
            
            # 截断观察
            if len(observation) > self.config.max_observation_length:
                observation = observation[:self.config.max_observation_length] + "\n<response clipped>"
            
            # 添加到历史
            self.history.append({"role": "assistant", "content": response})
            self.history.append({
                "role": "user",
                "content": f"Observation:\n{observation}"
            })
        
        # 检查是否达到上限
        if self.step_count >= self.config.max_steps:
            logger.warning(f"已达到轮数上限 ({self.config.max_steps})，停止执行")
            # 添加提示到历史
            self.history.append({
                "role": "user",
                "content": f"You have reached the maximum number of steps ({self.config.max_steps}). Please provide your final answer based on the information you have gathered. IMPORTANT: Your final answer MUST be written in English."
            })
            # 最后一次调用模型获取答案
            response, step_stats = self.model.forward(self.history)
            self.stats["total_latency"] += step_stats["latency"]
            self.stats["total_input_tokens"] += step_stats["input_tokens"]
            self.stats["total_output_tokens"] += step_stats["output_tokens"]
            self.history.append({"role": "assistant", "content": response})
        
        # 提取最终答案
        final_answer = self._extract_answer()
        
        return {
            "answer": final_answer,
            "steps": self.step_count,
            "history": self.history,
            "latency": self.stats["total_latency"],
            "input_tokens": self.stats["total_input_tokens"],
            "output_tokens": self.stats["total_output_tokens"],
        }
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tools_desc = """
Available tools:
1. bash: Execute bash commands. Format: {"name": "bash", "arguments": {"command": "your command"}}
   Example: {"name": "bash", "arguments": {"command": "ls -la"}}
2. read_file: Read a file. Format: {"name": "read_file", "arguments": {"path": "file/path"}}
3. grep: Search for patterns in files. Format: {"name": "grep", "arguments": {"pattern": "pattern", "path": "directory"}}
4. ls: List directory contents. Format: {"name": "ls", "arguments": {"path": "directory"}}

When you want to use a tool, respond with a JSON object in this format:
{"name": "tool_name", "arguments": {...}}

After each tool execution, you will receive the output. Continue exploring until you have enough information to answer the question.

IMPORTANT: You must provide your final answer in English. All responses, explanations, and conclusions should be written in English.
"""
        return f"""You are a helpful assistant that can interact with a code repository to answer questions about it.
You have access to the repository's files and can use various tools to explore and understand the codebase.

{tools_desc}"""
    
    def _build_instance_prompt(self) -> str:
        """构建实例提示"""
        question = self.problem_statement.get_problem_statement()
        repo_path = self.env.repo_path
        
        return f"""I've uploaded a code repository in the directory {repo_path}. Please answer the following question about this repository:

<question>
{question}
</question>

Your task is to thoroughly explore the repository and provide a comprehensive answer to the question.
Follow these steps to answer the question:
1. First, explore the repository structure to understand what the codebase is about
2. Search for relevant code, files, and documentation related to the question
3. Read and analyze the relevant code sections
4. Provide a clear, detailed answer based on your findings
5. If applicable, include code examples or references to specific files/functions

Your thinking should be thorough and comprehensive. Take your time to explore the codebase properly.

IMPORTANT: You MUST provide your final answer in English. All your responses, explanations, code analysis, and conclusions must be written in English, regardless of the language of the question or code comments."""
    
    def _extract_answer(self) -> str:
        """从历史中提取最终答案"""
        # 简单实现：返回最后一条 assistant 消息
        for msg in reversed(self.history):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return "No answer found"


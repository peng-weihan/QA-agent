#!/usr/bin/env python3
"""轻量化的 QA Agent 主程序"""
import argparse
import logging
from pathlib import Path

from qa_agent import QAAgent, QAAgentConfig
from agent_models import ModelConfig
from agent_tools import ToolConfig
from local_env import LocalEnv
from problem_statement import TextProblemStatement

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="轻量化的 QA Agent")
    parser.add_argument("--question", "-q", required=True, help="要回答的问题")
    parser.add_argument("--repo", "-r", required=True, type=Path, help="代码仓库路径")
    parser.add_argument("--model", "-m", default="gpt-4o", help="模型名称")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--max-steps", type=int, default=10, help="最大执行步数（轮数上限），默认 10")
    
    args = parser.parse_args()
    
    # 创建模型配置
    model_config = ModelConfig(
        name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    
    # 创建工具配置
    tool_config = ToolConfig()
    
    # 创建 Agent 配置
    agent_config = QAAgentConfig(
        model=model_config,
        tools=tool_config,
        max_steps=args.max_steps,
    )
    
    # 创建环境
    env = LocalEnv(repo_path=args.repo)
    
    # 创建问题陈述
    problem_statement = TextProblemStatement(text=args.question)
    
    # 创建并运行 Agent
    agent = QAAgent(agent_config, env, problem_statement)
    result = agent.run()
    
    # 输出结果
    print("\n" + "="*80)
    print("最终答案:")
    print("="*80)
    print(result["answer"])
    print(f"\n执行了 {result['steps']} 步")


if __name__ == "__main__":
    main()


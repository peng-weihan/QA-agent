#!/usr/bin/env python3
"""批处理脚本：从 JSONL 文件读取问题并批量处理"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from qa_agent import QAAgent, QAAgentConfig
from agent_models import ModelConfig
from agent_tools import ToolConfig
from local_env import LocalEnv
from problem_statement import TextProblemStatement
from batch_config import BatchProcessConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 文件写入锁
file_lock = Lock()


def process_single_question(
    question_data: Dict[str, Any],
    repo_path: Path,
    model_config: ModelConfig,
    tool_config: ToolConfig,
    max_steps: int,
    question_idx: int = 0,
    total_questions: int = 0,
    save_trajectory: bool = False,
    trajectory_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """处理单个问题"""
    question = question_data.get("question", "")
    if not question:
        logger.warning(f"[Q{question_idx}] Empty question, skipping")
        return None
    
    logger.info(f"[Q{question_idx}/{total_questions}] Processing: {question[:50]}...")
    
    try:
        # 创建环境
        env = LocalEnv(repo_path=repo_path)
        
        # 创建问题陈述
        problem_statement = TextProblemStatement(text=question)
        
        # 创建 Agent 配置
        agent_config = QAAgentConfig(
            model=model_config,
            tools=tool_config,
            max_steps=max_steps,
        )
        
        # 创建并运行 Agent
        agent = QAAgent(agent_config, env, problem_statement)
        result = agent.run()
        
        # 保存轨迹（如果启用）
        trajectory_path = None
        if save_trajectory and trajectory_dir:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            trajectory_path = trajectory_dir / f"trajectory_{question_idx:04d}.json"
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump({
                    "question": question,
                    "question_idx": question_idx,
                    "history": result.get("history", []),
                    "steps": result["steps"],
                    "latency": result.get("latency", 0.0),
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"[Q{question_idx}] Trajectory saved to {trajectory_path}")
        
        # 构建输出结果
        output = {
            "question": question,
            "answer": result["answer"],
            "steps": result["steps"],
            "latency": result.get("latency", 0.0),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            # 如果保存了轨迹，添加轨迹文件路径
            **({"trajectory_path": str(trajectory_path)} if trajectory_path else {}),
            # 保留原始数据中的其他字段
            **{k: v for k, v in question_data.items() if k not in ["question", "answer"]}
        }
        
        logger.info(f"[Q{question_idx}/{total_questions}] Completed in {result['steps']} steps (latency: {result.get('latency', 0):.2f}s)")
        return output
        
    except Exception as e:
        logger.error(f"[Q{question_idx}/{total_questions}] Error processing question: {e}")
        return {
            "question": question,
            "answer": f"Error: {str(e)}",
            "steps": 0,
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            **{k: v for k, v in question_data.items() if k not in ["question", "answer"]}
        }

def batch_process(
    input_file: Path,
    output_file: Path,
    repo_path: Path,
    model_name: str,
    api_key: str = None,
    api_base: str = None,
    max_steps: int = 10,
    max_workers: int = 1,
    save_trajectory: bool = False,
    trajectory_dir: Optional[Path] = None,
):
    """批量处理问题
    
    Args:
        max_workers: 并行处理的线程数，默认为 1（串行处理）
    """
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 确定轨迹保存目录
    if save_trajectory and trajectory_dir is None:
        trajectory_dir = output_file.parent / "trajectories"
    if save_trajectory and trajectory_dir:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trajectories will be saved to: {trajectory_dir}")
    
    # 读取所有问题
    questions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                questions.append((line_num, data))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num} is not valid JSON: {e}")
                continue
    
    logger.info(f"Loaded {len(questions)} questions from {input_file}")
    logger.info(f"Using {max_workers} worker(s) for parallel processing")
    
    # 创建模型配置
    model_config = ModelConfig(
        name=model_name,
        api_key=api_key,
        api_base=api_base,
    )
    
    # 创建工具配置
    tool_config = ToolConfig()
    
    # 处理每个问题
    results = []
    completed_count = 0
    
    if max_workers == 1:
        # 串行处理
        for idx, (line_num, question_data) in enumerate(questions, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing question {idx}/{len(questions)} (line {line_num})")
            logger.info(f"{'='*80}")
            
            result = process_single_question(
                question_data,
                repo_path,
                model_config,
                tool_config,
                max_steps,
                question_idx=idx,
                total_questions=len(questions),
                save_trajectory=save_trajectory,
                trajectory_dir=trajectory_dir,
            )
            
            if result:
                results.append(result)
                # 实时写入结果（追加模式）
                with file_lock:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                logger.info(f"Result saved to {output_file}")
    else:
        # 并行处理
        def process_with_index(args):
            idx, line_num, question_data = args
            result = process_single_question(
                question_data,
                repo_path,
                model_config,
                tool_config,
                max_steps,
                question_idx=idx,
                total_questions=len(questions),
                save_trajectory=save_trajectory,
                trajectory_dir=trajectory_dir,
            )
            return result
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_question = {
                executor.submit(process_with_index, (idx, line_num, question_data)): (idx, line_num)
                for idx, (line_num, question_data) in enumerate(questions, 1)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_question):
                idx, line_num = future_to_question[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        completed_count += 1
                        # 实时写入结果（追加模式，使用锁保证线程安全）
                        with file_lock:
                            with open(output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        logger.info(f"[Q{idx}] Result saved to {output_file} ({completed_count}/{len(questions)} completed)")
                except Exception as e:
                    logger.error(f"[Q{idx}] Task failed: {e}")
                    completed_count += 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Batch processing completed!")
    logger.info(f"Total: {len(questions)} questions")
    logger.info(f"Success: {len(results)} results")
    logger.info(f"Output saved to: {output_file}")
    if save_trajectory and trajectory_dir:
        logger.info(f"Trajectories saved to: {trajectory_dir}")
    logger.info(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="批量处理 QA 问题")
    parser.add_argument("--config", "-c", type=Path, help="配置文件路径（JSON 格式）")
    parser.add_argument("--input", "-i", type=Path, help="输入 JSONL 文件路径")
    parser.add_argument("--output", "-o", type=Path, help="输出 JSONL 文件路径")
    parser.add_argument("--repo", "-r", type=Path, help="代码仓库路径")
    parser.add_argument("--model", "-m", help="模型名称")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--api-base", help="API base URL")
    parser.add_argument("--max-steps", type=int, help="最大执行步数")
    parser.add_argument("--max-workers", type=int, help="并行处理的线程数，默认为 1（串行）")
    parser.add_argument("--save-trajectory", action="store_true", help="保存执行轨迹到文件")
    parser.add_argument("--trajectory-dir", type=Path, help="轨迹保存目录，默认为输出文件所在目录下的 trajectories 文件夹")
    
    args = parser.parse_args()
    
    # 加载配置文件（如果提供）
    config = None
    if args.config:
        if not args.config.exists():
            logger.error(f"配置文件不存在: {args.config}")
            return
        try:
            config = BatchProcessConfig.from_file(args.config)
            logger.info(f"已加载配置文件: {args.config}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return
    else:
        # 如果没有配置文件，使用默认配置
        config = BatchProcessConfig()
    
    # 合并命令行参数（命令行参数优先级更高）
    args_dict = {
        "input_file": str(args.input) if args.input else None,
        "output_file": str(args.output) if args.output else None,
        "repo_path": str(args.repo) if args.repo else None,
        "model_name": args.model,
        "api_key": args.api_key,
        "api_base": args.api_base,
        "max_steps": args.max_steps,
        "max_workers": args.max_workers,
        "save_trajectory": args.save_trajectory if hasattr(args, 'save_trajectory') and args.save_trajectory else None,
        "trajectory_dir": str(args.trajectory_dir) if args.trajectory_dir else None,
    }
    config = config.merge_with_args(args_dict)
    
    # 验证必需参数
    if not config.input_file:
        parser.error("--input/-i 或配置文件中的 input_file 是必需的")
    if not config.output_file:
        parser.error("--output/-o 或配置文件中的 output_file 是必需的")
    if not config.repo_path:
        parser.error("--repo/-r 或配置文件中的 repo_path 是必需的")
    
    input_file = Path(config.input_file)
    output_file = Path(config.output_file)
    repo_path = Path(config.repo_path)
    
    # 如果输出文件已存在，清空它（重新开始）
    if output_file.exists():
        logger.warning(f"Output file {output_file} already exists. It will be overwritten.")
        output_file.unlink()
    
    trajectory_dir = Path(config.trajectory_dir) if config.trajectory_dir else None
    
    batch_process(
        input_file=input_file,
        output_file=output_file,
        repo_path=repo_path,
        model_name=config.model_name,
        api_key=config.api_key,
        api_base=config.api_base,
        max_steps=config.max_steps,
        max_workers=config.max_workers,
        save_trajectory=config.save_trajectory,
        trajectory_dir=trajectory_dir,
    )

if __name__ == "__main__":
    main()


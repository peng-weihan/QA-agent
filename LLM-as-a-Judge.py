import os
import openai
import json
import concurrent.futures
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("CUSTOM_LLM_API_KEY")

from openai import OpenAI

client = OpenAI(api_key=api_key, base_url="https://aihubmix.com/v1")

def score_answer(question, reference, candidate):
    # ... existing code ...
    prompt = f"""You are a professional evaluator. Please rate the candidate answer against the reference answer based on five criteria.
    Evaluation Criteria and Scoring Guidelines (each scored 1 to 10):
        1. Correctness:
            10 — Completely correct; core points and details are accurate with no ambiguity.
            8-9 — Mostly correct; only minor details are slightly inaccurate or loosely expressed.
            6-7 — Partially correct; some errors or omissions, but main points are generally accurate.
            4-5 — Several errors or ambiguities that affect understanding of the core information.
            2-3 — Many errors; misleading or fails to convey key information.
            1 — Serious errors; completely wrong or misleading.
        2. Completeness:
            10 — Covers all key points from the reference answer without omission.
            8-9 — Covers most key points; only minor non-critical information missing.
            6-7 — Missing several key points; content is somewhat incomplete.
            4-5 — Important information largely missing; content is one-sided.
            2-3 — Covers very little relevant information; seriously incomplete.
            1 — Covers almost no relevant information; completely incomplete.
        3. Relevance:
            10 — Content fully focused on the question topic; no irrelevant information.
            8-9 — Mostly focused; only minor irrelevant or peripheral information.
            6-7 — Generally on topic; some off-topic content but still relevant overall.
            4-5 — Topic not sufficiently focused; contains considerable off-topic content.
            2-3 — Content deviates from topic; includes excessive irrelevant information.
            1 — Majority of content irrelevant to the question.
        4. Clarity:
            10 — Fluent language; clear and precise expression; very easy to understand.
            8-9 — Mostly fluent; clear expression with minor unclear points.
            6-7 — Generally clear; some expressions slightly unclear or not concise.
            4-5 — Expression somewhat awkward; some ambiguity or lack of fluency.
            2-3 — Language obscure; sentences are not smooth; hinders understanding.
            1 — Expression confusing; very difficult to understand.
        5. Reasoning:
            10 — Reasoning is clear, logical, and well-structured; argumentation is excellent.
            8-9 — Reasoning is clear and logical; well-structured with solid argumentation.
            6-7 — Reasoning generally reasonable; mostly clear logic; minor jumps.
            4-5 — Reasoning is average; some logical jumps or organization issues.
            2-3 — Reasoning unclear; lacks logical order; difficult to follow.
            1 — No clear reasoning; logic is chaotic.

INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,10], corresponding
    to the evaluation scores:
        {{
        "correctness": <1-10>,
        "completeness": <1-10>,
        "relevance": <1-10>,
        "clarity": <1-10>,
        "reasoning": <1-10>
        }}

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON"""

    try:
        response = client.chat.completions.create(
            model="claude-sonnet-4-5-20250929",
            messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
            ],
            stream=False
        )
        score_str = response.choices[0].message.content.strip()
        print(f"评分结果：{score_str}")
        try:
            # 清理可能的代码块标记
            if score_str.startswith("```json"):
                score_str = score_str[7:]  # 移除 ```json
            if score_str.endswith("```"):
                score_str = score_str[:-3]  # 移除 ```
            score_str = score_str.strip()
            
            # 解析JSON格式的小分
            scores = json.loads(score_str)
            # 验证所有维度都在1-10范围内
            for key in ["correctness", "completeness", "clarity", "relevance", "reasoning"]:
                if key not in scores or not (1 <= scores[key] <= 10):
                    print(f"评分验证失败: {key} = {scores.get(key)}")
                    return None
            return scores
        except Exception as e:
            print(f"JSON解析失败: {e}")
            return None
    except Exception as e:
        print(f"评分出错: {e}")
        return None

def process_single_record(candidate_record: Dict[str, Any], reference_dict: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """处理单个记录的函数，用于并行执行"""
    try:
        question = candidate_record.get("question", "")
        candidate_answer = candidate_record.get("answer", "")
        
        # 从reference字典中获取对应问题的参考答案
        reference = reference_dict.get(question, "")
        
        if not reference:
            print(f"跳过记录: 缺少参考答案")
            return None
            
        if not candidate_answer or candidate_answer.strip() == "No answer found":
            print(f"跳过记录: 候选答案为空或'No answer found'")
            return None

        # 对候选答案进行评分
        scores = score_answer(question, reference, candidate_answer)
        
        if scores is None:
            print(f"跳过记录: 评分失败")
            return None
        
        # 创建新的记录，格式类似于现有的评分文件
        result_record = {
            "question": question,
            "score": {
                "correctness": scores["correctness"],
                "completeness": scores["completeness"],
                "clarity": scores["clarity"],
                "relevance": scores["relevance"],
                "reasoning": scores["reasoning"]
            }
        }
        
        print(f"已评分问题: {question[:50]}... - 小分: {scores} - 总分: {sum(scores.values())}")
        return result_record
        
    except Exception as e:
        print(f"处理记录时出错: {e}")
        return None

def evaluate_jsonl_parallel(candidate_jsonl_path, reference_jsonl_path, output_jsonl_path, max_workers=16):
    """并行处理 JSONL 文件"""
    # 读取参考答案并构建字典
    reference_dict = {}
    with open(reference_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                question = record.get("question", "")
                answer = record.get("answer", "")
                if question and answer:
                    reference_dict[question] = answer
            except Exception as e:
                print(f"[跳过] 无效参考答案JSON行: {e}")
                continue
    
    print(f"读取到 {len(reference_dict)} 条参考答案")
    
    # 读取候选答案记录
    candidate_records = []
    with open(candidate_jsonl_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                record = json.loads(line)
                candidate_records.append(record)
            except Exception as e:
                print(f"[跳过] 无效候选答案JSON行: {e}")
                continue
    
    print(f"总共读取到 {len(candidate_records)} 条候选答案记录，开始并行处理...")
    
    # 使用线程池并行处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_record = {executor.submit(process_single_record, record, reference_dict): record for record in candidate_records}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"处理记录时出错: {e}")
    
    print(f"评分完成，共处理 {len(results)} 条记录，准备写入结果...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # 写入结果
    with open(output_jsonl_path, 'w', encoding='utf-8') as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"结果已保存到: {output_jsonl_path}")
    
if __name__ == "__main__":
    repos = [
        # 'astropy',
        # 'django',
        # 'flask',
        # 'matplotlib',
        # 'pylint',
        # 'pytest',
        # 'requests',
        # 'scikit-learn',
        # 'sphinx',
        # 'sqlfluff',
        # 'sympy',
        # 'xarray',
        "reflex",
        # "streamlink",
        "conan"
    ]
    MODEL = "gpt-4.1-mini"
    # MODEL = "grok-4"
    # MODEL = "lingma"  
    # MODEL = "gpt-4o-v2"
    # MODEL = "claude-sonnet-3-7"
    METHOD = [ 
                "cli-DeepRepoQA", 
                "DeepRepoQA",
                # "func_chunk",
                # "sliding_window",
                # "swe_qa_agent",
                # "mcts"
                # "mcts_v1"
                # "mcts_v2",
                # "mcts_v3",
                # "mcts_v4"
                # "mcts_v5",
                # "mcts_max_3",
                # "mcts_max_5"
                # "mcts_max_10"
            ]
    # 设置路径
    for repo in repos:
        for method in METHOD:
            candidate_path = f"/data3/pwh/answer/{MODEL}/{method}/{repo}.jsonl"
            reference_path = f"/data3/pwh/reference/{repo}.jsonl"
            output_path = f"/data3/pwh/answer/score/llm-as-a-judge/{MODEL}/{method}/{repo}_score.jsonl"
            print(f"\n开始处理 {repo}...")
            print(f"候选答案路径: {candidate_path}")
            print(f"参考答案路径: {reference_path}")
            print(f"输出路径: {output_path}")
            
            # 检查文件是否存在
            if not os.path.exists(candidate_path):
                print(f"跳过 {repo}: 候选答案文件不存在")
                continue

            if not os.path.exists(reference_path):
                print(f"跳过 {repo}: 参考答案文件不存在")
                continue
                
            # 使用并行处理
            evaluate_jsonl_parallel(candidate_path, reference_path, output_path, max_workers=48)
            print(f"完成处理 {repo}")
            
import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from openai import OpenAI

from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent
from openhands.sdk.event import MessageEvent, ActionEvent, ObservationEvent

# 配置
LLM_CONFIG = {
    "model": "gpt-4.1-mini",
    "api_key": "sk-Xr5Mfi9JCZco8hPg22250d274dDb455bB40dA0Fc3b492dA6",
    "base_url": "https://aihubmix.com/v1",
    "usage_id": "agent"  # 使用 usage_id 替代已弃用的 service_id
}

# 仓库配置：依次处理 reflex、streamlink、conan
REPOS_CONFIG = [
    {
        "name": "reflex",
        "workspace": "/home/ugproj/raymone/swe-repos/reflex",
        "input_file": "/data2/raymone/questions/reflex.jsonl"
    },
    # {
    #     "name": "streamlink",
    #     "workspace": "/home/ugproj/pwh/swebench-repos/streamlink",
    #     "input_file": "/data2/raymone/questions/streamlink.jsonl"
    # },
    # {
    #     "name": "conan",
    #     "workspace": "/home/ugproj/raymone/swe-repos/conan",
    #     "input_file": "/data2/raymone/questions/conan.jsonl"
    # }
    # {
    #     "name": "astropy",
    #     "workspace": "/home/ugproj/pwh/swebench-repos/astropy",
    #     "input_file": "/data2/raymone/questions/astropy.jsonl"
    # },
    # {
    #     "name": "scikit-learn",
    #     "workspace": "/home/ugproj/pwh/swebench-repos/scikit-learn",
    #     "input_file": "/data2/raymone/questions/scikit-learn.jsonl"
    # },
]

OUTPUT_DIR = "/data2/raymone/answer/OpenHands_v1"
MAX_ITERATION_PER_RUN = 10
MAX_TIME_PER_QUESTION = 60  # 每个问题的最大处理时间（秒），默认10分钟

# 从输入文件名提取仓库名
def get_repo_name_from_path(file_path):
    """从文件路径提取仓库名"""
    basename = os.path.basename(file_path)
    # 去掉 .jsonl 后缀
    if basename.endswith('.jsonl'):
        return basename[:-6]
    return basename

def load_questions_from_jsonl(file_path):
    """从 jsonl 文件加载问题"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question = data.get('question', '')
                if question:
                    questions.append(data)
            except json.JSONDecodeError as e:
                print(f"跳过无效的 JSON 行: {e}")
    return questions

def load_answered_questions(output_file):
    """从输出文件中加载已经回答过的问题"""
    answered_questions = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        question = data.get('question', '')
                        if question:
                            answered_questions.add(question)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"读取已回答问题时出错: {e}")
    return answered_questions

def get_message_history(state):
    """从 conversation state 中提取完整的 message history"""
    messages = []
    events = list(state.events)
    
    print(f"[DEBUG] 总共有 {len(events)} 个事件")
    
    for idx, event in enumerate(events):
        event_type = type(event).__name__
        
        # 处理 MessageEvent
        if isinstance(event, MessageEvent):
            source = getattr(event, 'source', 'unknown')
            
            # 提取消息内容
            content = ""
            if hasattr(event, 'extended_content') and event.extended_content:
                text_parts = []
                for item in event.extended_content:
                    if hasattr(item, 'text'):
                        text_parts.append(str(item.text))
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    content = "\n".join(text_parts)
            
            if not content and hasattr(event, 'llm_message') and event.llm_message:
                msg = event.llm_message
                if hasattr(msg, 'content') and msg.content:
                    if isinstance(msg.content, list):
                        text_parts = []
                        for item in msg.content:
                            if hasattr(item, 'text'):
                                text_parts.append(str(item.text))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        if text_parts:
                            content = "\n".join(text_parts)
                    elif isinstance(msg.content, str):
                        content = msg.content
            
            if content:
                messages.append({
                    'role': source,  # 'user', 'agent', 'tool'
                    'content': content,
                    'timestamp': getattr(event, 'timestamp', None),
                    'event_type': event_type
                })
                print(f"[DEBUG Event {idx}] MessageEvent, source={source}, content长度={len(content)}")
        
        # 处理 ActionEvent (agent 执行的动作)
        elif isinstance(event, ActionEvent):
            action_info = ""
            if hasattr(event, 'action') and event.action:
                action = event.action
                action_name = getattr(action, 'name', 'unknown')
                action_info = f"Action: {action_name}"
                
                # 尝试获取动作的参数
                if hasattr(action, 'model_dump'):
                    try:
                        action_dict = action.model_dump()
                        action_info += f"\nParameters: {json.dumps(action_dict, ensure_ascii=False, indent=2)}"
                    except:
                        action_info += f"\nAction object: {str(action)}"
            
            if action_info:
                messages.append({
                    'role': 'agent',
                    'content': action_info,
                    'timestamp': getattr(event, 'timestamp', None),
                    'event_type': event_type
                })
                print(f"[DEBUG Event {idx}] ActionEvent, content长度={len(action_info)}")
        
        # 处理 ObservationEvent (工具执行的结果)
        elif isinstance(event, ObservationEvent):
            observation_info = ""
            if hasattr(event, 'observation') and event.observation:
                observation = event.observation
                # 尝试获取观察结果
                if hasattr(observation, 'message'):
                    observation_info = f"Observation: {observation.message}"
                elif hasattr(observation, 'content'):
                    observation_info = f"Observation: {observation.content}"
                elif hasattr(observation, 'model_dump'):
                    try:
                        obs_dict = observation.model_dump()
                        observation_info = f"Observation: {json.dumps(obs_dict, ensure_ascii=False, indent=2)}"
                    except:
                        observation_info = f"Observation: {str(observation)}"
                else:
                    observation_info = f"Observation: {str(observation)}"
            
            if observation_info:
                messages.append({
                    'role': 'tool',
                    'content': observation_info,
                    'timestamp': getattr(event, 'timestamp', None),
                    'event_type': event_type
                })
                print(f"[DEBUG Event {idx}] ObservationEvent, content长度={len(observation_info)}")
        else:
            # 其他类型的事件，也尝试提取信息
            print(f"[DEBUG Event {idx}] 其他事件类型: {event_type}")
    
    print(f"[DEBUG] 提取到 {len(messages)} 条消息")
    return messages

def generate_answer_from_history(llm_config, question, message_history):
    """基于 message_history 使用 LLM 生成答案"""
    try:
        client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"]
        )
        
        # 将 message_history 格式化为文本格式，避免 tool role 的问题
        # 将所有消息合并为一个文本，而不是使用 tool role
        conversation_text = []
        
        for msg in message_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if content:
                # 将不同角色的消息格式化为文本
                if role == 'user':
                    conversation_text.append(f"User: {content}")
                elif role == 'agent':
                    conversation_text.append(f"Assistant: {content}")
                elif role == 'tool':
                    conversation_text.append(f"Tool Output: {content}")
                else:
                    conversation_text.append(f"{role}: {content}")
        
        # 构建完整的提示
        full_conversation = "\n\n".join(conversation_text)
        
        system_prompt = """You are a code repository question answering assistant. 
Based on the conversation history provided, synthesize all the information gathered and provide a comprehensive answer to the user's question.
Even if the information is incomplete, provide the best answer you can based on what was discovered during the exploration."""
        
        user_prompt = f"""Original Question: {question}

Conversation History:
{full_conversation}

Based on the conversation history above, please provide a comprehensive answer to the original question."""
        
        formatted_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[DEBUG] 调用 LLM 生成答案，对话历史长度: {len(full_conversation)} 字符")
        
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=formatted_messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content.strip()
        
        # 返回答案和 token 使用量（分别返回 prompt 和 completion）
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens
        
        print(f"[DEBUG] LLM 生成答案成功，长度: {len(answer)} 字符")
        print(f"[DEBUG] Token使用: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        
        return answer, (prompt_tokens, completion_tokens)
    except Exception as e:
        print(f"[DEBUG] 生成答案时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def safe_close_conversation(conversation):
    """安全地关闭 conversation，避免 agent 未初始化的错误"""
    if conversation is None:
        return
    try:
        conversation.close()
    except RuntimeError as e:
        # 捕获 "Agent not initialized" 错误并忽略
        if "not initialized" in str(e) or "Agent not initialized" in str(e):
            pass
        else:
            # 其他 RuntimeError 重新抛出
            raise
    except Exception:
        # 其他所有错误都忽略，避免影响主流程
        pass

def process_single_question(qa_data, workspace):
    """处理单个问题"""
    question = qa_data.get('question', '')
    if not question:
        return None
    
    # 为每个任务创建独立的 agent 和 conversation
    llm = LLM(**LLM_CONFIG)
    agent = get_default_agent(llm=llm, cli_mode=True)
    
    # 用于存储答案的变量
    answer_data = {
        "question": question,
        "answer": "",
        "timestamp": datetime.now().isoformat(),
        "time_cost": 0.0,
        "token_cost": 0,  # 总 token 数（向后兼容）
        "prompt_tokens": 0,  # input tokens
        "completion_tokens": 0  # output tokens
    }
    
    # 回调函数来捕获答案
    def on_event(event):
        nonlocal answer_data
        if isinstance(event, MessageEvent):
            if hasattr(event, 'source') and event.source == 'agent':
                if hasattr(event, 'extended_content') and event.extended_content:
                    text_content = []
                    for item in event.extended_content:
                        if hasattr(item, 'text'):
                            text_content.append(item.text)
                        elif isinstance(item, str):
                            text_content.append(item)
                    if text_content:
                        answer_data["answer"] = "\n".join(text_content)
                elif hasattr(event, 'llm_message') and event.llm_message:
                    msg = event.llm_message
                    if hasattr(msg, 'content') and msg.content:
                        if isinstance(msg.content, list):
                            text_content = []
                            for item in msg.content:
                                if hasattr(item, 'text'):
                                    text_content.append(item.text)
                                elif isinstance(item, str):
                                    text_content.append(item)
                            if text_content:
                                answer_data["answer"] = "\n".join(text_content)
                        elif isinstance(msg.content, str):
                            answer_data["answer"] = msg.content
    
    conversation = None
    try:
        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            max_iteration_per_run=MAX_ITERATION_PER_RUN,
            callbacks=[on_event]
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 添加探索提示
        enhanced_question = f"""Please first explore the codebase structure to find the relevant files.
Use the terminal tool to search for files related to the question.
Then answer: {question}"""
        
        conversation.send_message(enhanced_question)
        
        # 使用线程池执行 conversation.run() 并设置超时
        timeout_occurred = False
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(conversation.run)
                future.result(timeout=MAX_TIME_PER_QUESTION)
        except FutureTimeoutError:
            timeout_occurred = True
            print(f"[TIMEOUT] 问题处理超时（>{MAX_TIME_PER_QUESTION}秒），放弃当前问题")
            answer_data["answer"] = f"Timeout: Question processing exceeded {MAX_TIME_PER_QUESTION} seconds and was aborted."
            answer_data["time_cost"] = MAX_TIME_PER_QUESTION
            return answer_data
        
        # 计算时间成本
        end_time = time.time()
        answer_data["time_cost"] = round(end_time - start_time, 2)
        
        # 从 conversation state 中获取最后的答案和 token 使用量
        state = conversation.state
        
        # 获取完整的 message history（运行10轮之后）
        message_history = get_message_history(state)
        print(f"[DEBUG] 获取到 {len(message_history)} 条消息历史")
        
        # 获取 token 使用量（分别统计 input 和 output）
        if hasattr(state, 'stats') and state.stats:
            stats = state.stats
            if hasattr(stats, 'usage_to_metrics'):
                total_prompt_tokens = 0
                total_completion_tokens = 0
                for usage_id, metrics in stats.usage_to_metrics.items():
                    if hasattr(metrics, 'accumulated_token_usage') and metrics.accumulated_token_usage:
                        token_usage = metrics.accumulated_token_usage
                        if hasattr(token_usage, 'prompt_tokens') and hasattr(token_usage, 'completion_tokens'):
                            total_prompt_tokens += token_usage.prompt_tokens
                            total_completion_tokens += token_usage.completion_tokens
                    if hasattr(metrics, 'token_usages') and metrics.token_usages:
                        for token_usage in metrics.token_usages:
                            if hasattr(token_usage, 'prompt_tokens') and hasattr(token_usage, 'completion_tokens'):
                                total_prompt_tokens += token_usage.prompt_tokens
                                total_completion_tokens += token_usage.completion_tokens
                answer_data["prompt_tokens"] = total_prompt_tokens
                answer_data["completion_tokens"] = total_completion_tokens
                answer_data["token_cost"] = total_prompt_tokens + total_completion_tokens  # 总 token 数（向后兼容）
        
        # 如果还没有获取到答案，从 events 中查找
        if not answer_data["answer"]:
            events = list(state.events)
            for event in reversed(events):
                if isinstance(event, MessageEvent) and hasattr(event, 'source') and event.source == 'agent':
                    if hasattr(event, 'extended_content') and event.extended_content:
                        text_content = []
                        for item in event.extended_content:
                            if hasattr(item, 'text'):
                                text_content.append(item.text)
                            elif isinstance(item, str):
                                text_content.append(item)
                        if text_content:
                            answer_data["answer"] = "\n".join(text_content)
                            break
                    elif hasattr(event, 'llm_message') and event.llm_message:
                        msg = event.llm_message
                        if hasattr(msg, 'content') and msg.content:
                            if isinstance(msg.content, list):
                                text_content = []
                                for item in msg.content:
                                    if hasattr(item, 'text'):
                                        text_content.append(item.text)
                                    elif isinstance(item, str):
                                        text_content.append(item)
                                if text_content:
                                    answer_data["answer"] = "\n".join(text_content)
                                    break
                            elif isinstance(msg.content, str):
                                answer_data["answer"] = msg.content
                                break
        
        # 如果仍然没有答案，检查是否达到 MAX_ITERATION_PER_RUN 限制
        if not answer_data["answer"]:
            # 统计 ActionEvent 的数量来判断是否达到限制
            events = list(state.events)
            action_count = sum(1 for event in events if isinstance(event, ActionEvent))
            
            print(f"[DEBUG] 未找到答案，ActionEvent 数量: {action_count}, 最大限制: {MAX_ITERATION_PER_RUN}")
            
            # 如果达到或超过限制，使用 message_history 生成答案
            if action_count >= MAX_ITERATION_PER_RUN:
                print(f"[DEBUG] 达到最大迭代次数限制 ({MAX_ITERATION_PER_RUN})，使用 message_history 生成最终答案...")
                
                if message_history and len(message_history) > 0:
                    forced_answer, (forced_prompt_tokens, forced_completion_tokens) = generate_answer_from_history(
                        LLM_CONFIG, question, message_history
                    )
                    
                    if forced_answer:
                        answer_data["answer"] = forced_answer
                        # 更新 token 成本（累加强制生成的 tokens）
                        previous_prompt = answer_data.get("prompt_tokens", 0)
                        previous_completion = answer_data.get("completion_tokens", 0)
                        answer_data["prompt_tokens"] = previous_prompt + forced_prompt_tokens
                        answer_data["completion_tokens"] = previous_completion + forced_completion_tokens
                        answer_data["token_cost"] = answer_data["prompt_tokens"] + answer_data["completion_tokens"]
                        print(f"[DEBUG] 已使用 message_history 生成答案，答案长度: {len(answer_data['answer'])} 字符")
                        print(f"[DEBUG] Token统计: prompt={previous_prompt}+{forced_prompt_tokens}={answer_data['prompt_tokens']}, completion={previous_completion}+{forced_completion_tokens}={answer_data['completion_tokens']}, total={answer_data['token_cost']}")
                    else:
                        print(f"[DEBUG] 警告: 使用 message_history 生成答案失败")
                        answer_data["answer"] = "Unable to generate answer based on conversation history."
                else:
                    print(f"[DEBUG] 警告: message_history 为空，无法生成答案")
                    answer_data["answer"] = "No conversation history available to generate answer."
        
        return answer_data
        
    except Exception as e:
        print(f"处理问题失败: {question[:50]}... 错误: {e}")
        answer_data["answer"] = f"Error: {str(e)}"
        return answer_data
    finally:
        # 确保无论什么情况都关闭 conversation，避免 __del__ 时的错误
        safe_close_conversation(conversation)

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 全局统计
    all_processed_count = 0
    all_error_count = 0
    all_time_costs = []
    all_token_costs = []
    all_prompt_tokens_list = []
    all_completion_tokens_list = []
    
    # 依次处理每个仓库
    for repo_idx, repo_config in enumerate(REPOS_CONFIG, 1):
        repo_name = repo_config["name"]
        workspace = repo_config["workspace"]
        input_file = repo_config["input_file"]
        
        print(f"\n{'='*60}")
        print(f"开始处理仓库 {repo_idx}/{len(REPOS_CONFIG)}: {repo_name}")
        print(f"{'='*60}")
        print(f"工作空间: {workspace}")
        print(f"问题文件: {input_file}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告: 输入文件不存在，跳过: {input_file}")
            continue
        
        output_file = os.path.join(OUTPUT_DIR, f"{repo_name}_answers.jsonl")
        
        # 加载已回答的问题
        answered_questions = load_answered_questions(output_file)
        if answered_questions:
            print(f"已找到 {len(answered_questions)} 个已回答的问题")
        
        # 加载所有问题
        print(f"从 {input_file} 加载问题...")
        all_questions = load_questions_from_jsonl(input_file)
        print(f"共加载 {len(all_questions)} 个问题")
        
        # 过滤掉已回答的问题
        questions = [
            qa_data for qa_data in all_questions 
            if qa_data.get('question', '') not in answered_questions
        ]
        
        if len(questions) < len(all_questions):
            print(f"过滤后剩余 {len(questions)} 个未回答的问题")
        else:
            print(f"所有问题都未回答，将处理全部 {len(questions)} 个问题")
        
        if len(questions) == 0:
            print(f"仓库 {repo_name} 没有需要处理的问题，跳过")
            continue
        
        # 串行处理
        processed_count = 0
        error_count = 0
        time_costs = []
        token_costs = []
        prompt_tokens_list = []
        completion_tokens_list = []
        
        for idx, qa_data in enumerate(questions, 1):
            try:
                result = process_single_question(qa_data, workspace)
                if result:
                    # 写入文件
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')
                    
                    # 收集统计数据
                    time_cost = result.get('time_cost', 0)
                    token_cost = result.get('token_cost', 0)
                    prompt_tokens = result.get('prompt_tokens', 0)
                    completion_tokens = result.get('completion_tokens', 0)
                    
                    if time_cost is not None:
                        time_costs.append(time_cost)
                        all_time_costs.append(time_cost)
                    if token_cost is not None:
                        token_costs.append(token_cost)
                        all_token_costs.append(token_cost)
                    if prompt_tokens is not None:
                        prompt_tokens_list.append(prompt_tokens)
                        all_prompt_tokens_list.append(prompt_tokens)
                    if completion_tokens is not None:
                        completion_tokens_list.append(completion_tokens)
                        all_completion_tokens_list.append(completion_tokens)
                    
                    processed_count += 1
                    all_processed_count += 1
                    print(f"[{repo_name}][{idx}/{len(questions)}] 完成: {result['question'][:50]}...")
                else:
                    error_count += 1
                    all_error_count += 1
            except Exception as e:
                error_count += 1
                all_error_count += 1
                print(f"[{repo_name}][{idx}/{len(questions)}] 处理失败: {qa_data.get('question', '')[:50]}... 错误: {e}")
        
        # 计算当前仓库的统计信息
        avg_time_cost = sum(time_costs) / len(time_costs) if time_costs else 0
        avg_token_cost = sum(token_costs) / len(token_costs) if token_costs else 0
        total_time_cost = sum(time_costs)
        total_token_cost = sum(token_costs)
        
        avg_prompt_tokens = sum(prompt_tokens_list) / len(prompt_tokens_list) if prompt_tokens_list else 0
        avg_completion_tokens = sum(completion_tokens_list) / len(completion_tokens_list) if completion_tokens_list else 0
        total_prompt_tokens = sum(prompt_tokens_list)
        total_completion_tokens = sum(completion_tokens_list)
        
        print(f"\n仓库 {repo_name} 处理完成:")
        print(f"  成功: {processed_count}, 失败: {error_count}, 总计: {len(questions)}")
        print(f"  平均 time_cost: {avg_time_cost:.2f} 秒")
        print(f"  总 time_cost: {total_time_cost:.2f} 秒")
        print(f"  平均 token_cost: {avg_token_cost:.0f} tokens")
        print(f"  总 token_cost: {total_token_cost:.0f} tokens")
        print(f"  平均 prompt_tokens: {avg_prompt_tokens:.0f}, completion_tokens: {avg_completion_tokens:.0f}")
        print(f"  总 prompt_tokens: {total_prompt_tokens:.0f}, completion_tokens: {total_completion_tokens:.0f}")
        print(f"  结果已保存到: {output_file}")
    
    # 计算全局统计信息
    avg_time_cost = sum(all_time_costs) / len(all_time_costs) if all_time_costs else 0
    avg_token_cost = sum(all_token_costs) / len(all_token_costs) if all_token_costs else 0
    total_time_cost = sum(all_time_costs)
    total_token_cost = sum(all_token_costs)
    
    avg_prompt_tokens = sum(all_prompt_tokens_list) / len(all_prompt_tokens_list) if all_prompt_tokens_list else 0
    avg_completion_tokens = sum(all_completion_tokens_list) / len(all_completion_tokens_list) if all_completion_tokens_list else 0
    total_prompt_tokens = sum(all_prompt_tokens_list)
    total_completion_tokens = sum(all_completion_tokens_list)
    
    print(f"\n{'='*60}")
    print(f"所有仓库处理完成!")
    print(f"{'='*60}")
    print(f"成功: {all_processed_count}, 失败: {all_error_count}")
    print(f"\n全局统计信息:")
    print(f"  平均 time_cost: {avg_time_cost:.2f} 秒")
    print(f"  总 time_cost: {total_time_cost:.2f} 秒")
    print(f"\nToken 统计 (总计):")
    print(f"  平均 token_cost: {avg_token_cost:.0f} tokens")
    print(f"  总 token_cost: {total_token_cost:.0f} tokens")
    print(f"\nToken 统计 (分开):")
    print(f"  平均 prompt_tokens (input): {avg_prompt_tokens:.0f} tokens")
    print(f"  平均 completion_tokens (output): {avg_completion_tokens:.0f} tokens")
    print(f"  总 prompt_tokens (input): {total_prompt_tokens:.0f} tokens")
    print(f"  总 completion_tokens (output): {total_completion_tokens:.0f} tokens")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

# CUSTOM_LLM_MODEL= "deepseek/gpt-4.1-mini"
# CUSTOM_LLM_API_BASE="https://aihubmix.com/v1"
# CUSTOM_LLM_API_KEY="sk-Xr5Mfi9JCZco8hPg22250d274dDb455bB40dA0Fc3b492dA6"
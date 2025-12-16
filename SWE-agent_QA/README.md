# SWE-Agent QA: 轻量化的代码问答 Agent

这是从 SWE-agent 提取的轻量化 QA 系统，**不需要 Docker**，直接使用本地文件系统执行命令。专门用于回答关于代码仓库的问题。

## 特性

- ✅ **无 Docker 依赖**：直接使用本地文件系统
- ✅ **轻量级**：只包含核心的 Agent 框架和工具
- ✅ **易于使用**：简单的命令行接口
- ✅ **批量处理**：支持从 JSONL 文件批量处理问题
- ✅ **轨迹保存**：可保存完整的执行轨迹用于分析
- ✅ **基于 SWE-agent**：使用相同的核心架构

## 安装

```bash
cd SWE-agent_QA
pip install -r requirements.txt
```

依赖项：
- `litellm>=1.0.0` - 模型接口
- `pydantic>=2.0.0` - 数据验证

## 使用方法

### 1. 单问题处理（main.py）

#### 基本用法

```bash
python main.py \
    --question "这个项目的主要功能是什么？" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key your-api-key
```

#### 使用自定义 API

```bash
python main.py \
    --question "这个项目的主要功能是什么？" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key your-api-key \
    --api-base https://api.openai.com/v1
```

#### 完整参数

```bash
python main.py \
    --question "问题" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key your-api-key \
    --api-base https://api.openai.com/v1 \
    --max-steps 20
```

**参数说明：**
- `--question, -q`: 要回答的问题（必需）
- `--repo, -r`: 代码仓库路径（必需）
- `--model, -m`: 模型名称（默认: gpt-4o）
- `--api-key`: API key
- `--api-base`: API base URL
- `--max-steps`: 最大执行步数（默认: 10）

### 2. 批量处理（batch_process.py）

#### 使用配置文件

1. 复制示例配置文件：
```bash
cp batch_config.example.json config.json
```

2. 编辑 `config.json`：
```json
{
  "input_file": "path/to/input.jsonl",
  "output_file": "path/to/output.jsonl",
  "repo_path": "path/to/repo",
  "model_name": "gpt-4o",
  "api_key": "your-api-key-here",
  "api_base": "https://api.openai.com/v1",
  "max_steps": 20,
  "max_workers": 8,
  "save_trajectory": true,
  "trajectory_dir": "path/to/trajectories"
}
```

3. 运行批量处理：
```bash
python batch_process.py --config config.json
```

#### 使用命令行参数

```bash
python batch_process.py \
    --input input.jsonl \
    --output output.jsonl \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key your-api-key \
    --api-base https://api.openai.com/v1 \
    --max-steps 20 \
    --max-workers 8 \
    --save-trajectory \
    --trajectory-dir ./trajectory
```

**输入文件格式（JSONL）：**
每行一个 JSON 对象，包含 `question` 字段：
```json
{"question": "这个函数的作用是什么？"}
{"question": "如何配置这个组件？"}
```

**输出文件格式（JSONL）：**
每行一个 JSON 对象，包含：
```json
{
  "question": "这个函数的作用是什么？",
  "answer": "...",
  "steps": 5,
  "latency": 12.34,
  "input_tokens": 1234,
  "output_tokens": 567,
  "trajectory_path": "path/to/trajectory/trajectory_0001.json"
}
```

## 项目结构

```
SWE-agent_QA/
├── main.py                 # 单问题处理入口
├── batch_process.py        # 批量处理入口
├── batch_config.py        # 批量处理配置类
├── batch_config.example.json  # 配置文件示例
├── qa_agent.py            # QA Agent 主类
├── agent_models.py        # 模型接口（基于 LiteLLM）
├── agent_tools.py         # 工具处理系统
├── local_env.py           # 本地执行环境（替代 Docker）
├── problem_statement.py   # 问题陈述处理
├── requirements.txt       # 依赖项
└── trajectory/            # 轨迹保存目录（可选）
```

## 核心组件

### QAAgent
主要的 Agent 类，负责执行问答任务。支持多步骤推理，可以：
- 搜索代码库
- 读取文件
- 执行命令
- 分析代码结构

### LocalEnv
本地执行环境，提供：
- 文件系统操作
- 命令执行
- 代码搜索

### ToolHandler
工具处理系统，管理 Agent 可用的工具：
- `grep`: 搜索代码
- `read_file`: 读取文件
- `list_directory`: 列出目录
- `run_command`: 执行命令

## 与 SWE-agent 的区别

1. **无 Docker**：直接使用本地文件系统执行命令
2. **简化工具**：只保留基本的文件操作和命令执行
3. **轻量级**：移除了复杂的部署、hooks、reviewer 等组件
4. **专注 QA**：专门为问答任务优化，不适合代码修复

## 限制

- 不支持复杂的工具 bundle
- 不支持多步骤的代码修改
- 专注于问答任务，不适合代码修复
- 需要提供有效的 API key 和模型访问权限

## 示例

### 示例 1：询问项目功能

```bash
python main.py \
    --question "这个项目的主要功能是什么？" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key $OPENAI_API_KEY
```

### 示例 2：询问特定函数

```bash
python main.py \
    --question "process_data 函数是如何工作的？" \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key $OPENAI_API_KEY \
    --max-steps 15
```

### 示例 3：批量处理

```bash
# 准备输入文件 questions.jsonl
echo '{"question": "这个项目的主要功能是什么？"}' > questions.jsonl
echo '{"question": "如何配置这个组件？"}' >> questions.jsonl

# 运行批量处理
python batch_process.py \
    --input questions.jsonl \
    --output answers.jsonl \
    --repo /path/to/repo \
    --model gpt-4o \
    --api-key $OPENAI_API_KEY \
    --max-steps 20 \
    --max-workers 4 \
    --save-trajectory \
    --trajectory-dir ./trajectory
```

## 日志和调试

程序使用 Python 的 `logging` 模块输出日志。默认级别为 `INFO`，可以通过环境变量或代码修改日志级别。

## 许可证

基于 SWE-agent 项目，请参考原始项目的许可证。

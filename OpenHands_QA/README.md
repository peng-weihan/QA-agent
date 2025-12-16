# OpenHands QA Agent

基于 OpenHands SDK 的代码仓库问答系统，用于自动回答关于代码仓库的问题。

## 功能特性

- 使用 OpenHands SDK 和工具集进行代码分析和问答
- 支持批量处理 JSONL 格式的问题文件
- 支持多个代码仓库的配置和处理
- 自动跳过已回答的问题，支持断点续传
- 可配置的超时和迭代限制
- 完整的对话历史记录和答案生成

## 环境要求

- Python 3.12.11（推荐）
- Conda 环境

## 安装步骤

### 1. 创建 Conda 环境

```bash
conda create -n openhands_qa python=3.12.11 -y
conda activate openhands_qa
```

### 2. 安装依赖

```bash
cd <项目目录>
pip install -r requirements.txt
```

## 配置说明

### LLM 配置

在 `main_.py` 中配置 LLM 参数：

```python
LLM_CONFIG = {
    "model": "gpt-4.1-mini",
    "api_key": "your-api-key",
    "base_url": "https://aihubmix.com/v1",
    "usage_id": "agent"
}
```

### 仓库配置

在 `REPOS_CONFIG` 中配置要处理的代码仓库：

```python
REPOS_CONFIG = [
    {
        "name": "reflex",
        "workspace": "/path/to/repo",
        "input_file": "/path/to/questions.jsonl"
    },
    # 添加更多仓库...
]
```

### 输出配置

- `OUTPUT_DIR`: 答案输出目录
- `MAX_ITERATION_PER_RUN`: 每个问题的最大迭代次数（默认：10）
- `MAX_TIME_PER_QUESTION`: 每个问题的最大处理时间，单位：秒（默认：120）

## 使用方法

### 运行主程序

```bash
python main_.py
```

程序会：
1. 读取配置的仓库和问题文件
2. 检查已回答的问题，跳过重复处理
3. 使用 OpenHands Agent 处理每个问题
4. 生成答案并保存到输出目录

### 输入文件格式

问题文件应为 JSONL 格式，每行一个 JSON 对象：

```json
{"question": "How does the authentication work in this codebase?"}
{"question": "What is the purpose of the main function?"}
```

### 输出文件格式

答案文件同样为 JSONL 格式，包含原始问题和生成的答案：

```json
{"question": "...", "answer": "...", "message_history": [...], ...}
```

## 项目结构

```
OpenHands_QA/
├── main_.py           # 主程序文件
├── requirements.txt   # Python 依赖包
└── README.md         # 本文件
```

## 依赖包

- `openai==2.8.1` - OpenAI API 客户端
- `openhands-sdk==1.1.0` - OpenHands SDK
- `openhands-tools==1.1.0` - OpenHands 工具集

## 注意事项

- 确保代码仓库路径和问题文件路径正确
- 确保有足够的 API 配额和权限
- 输出目录需要有写入权限
- 程序支持断点续传，可以安全地中断和重启

## 故障排除

如果遇到问题：

1. 检查 Python 版本是否符合要求
2. 确认所有依赖包已正确安装
3. 验证 API 密钥和配置是否正确
4. 检查文件路径和权限设置


"""批处理配置文件管理"""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class BatchProcessConfig(BaseModel):
    """批处理配置"""
    input_file: Optional[str] = Field(None, description="输入 JSONL 文件路径")
    output_file: Optional[str] = Field(None, description="输出 JSONL 文件路径")
    repo_path: Optional[str] = Field(None, description="代码仓库路径")
    model_name: str = Field("gpt-4o", description="模型名称")
    api_key: Optional[str] = Field(None, description="API key")
    api_base: Optional[str] = Field(None, description="API base URL")
    max_steps: int = Field(10, description="最大执行步数")
    max_workers: int = Field(1, description="并行处理的线程数")
    save_trajectory: bool = Field(False, description="是否保存执行轨迹")
    trajectory_dir: Optional[str] = Field(None, description="轨迹保存目录，如果为 None 则使用输出文件所在目录")
    
    @classmethod
    def from_file(cls, config_path: Path) -> "BatchProcessConfig":
        """从 JSON 文件加载配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_file(self, config_path: Path) -> None:
        """保存配置到 JSON 文件"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(exclude_none=True), f, indent=2, ensure_ascii=False)
    
    def merge_with_args(self, args: Dict[str, Any]) -> "BatchProcessConfig":
        """合并命令行参数（命令行参数优先级更高）"""
        merged_data = self.model_dump(exclude_none=True)
        for key, value in args.items():
            # 合并所有非 None 的值（包括 False 等布尔值）
            if value is not None:
                merged_data[key] = value
        return BatchProcessConfig(**merged_data)


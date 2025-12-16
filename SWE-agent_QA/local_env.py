"""轻量化的本地执行环境，替代 Docker 环境"""
from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import Optional, Union, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LocalEnv:
    """本地执行环境，直接执行命令而不需要 Docker"""
    
    def __init__(self, repo_path: Path, working_dir: Optional[Path] = None):
        self.repo_path = Path(repo_path).resolve()
        self.working_dir = working_dir or self.repo_path
        self.env_variables = {}
        
    def communicate(
        self,
        command: str,
        timeout: int = 25,
        check: str = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """在本地执行命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间（秒）
            check: "ignore", "warn", "raise"
            error_msg: 错误消息
            
        Returns:
            命令输出
        """
        logger.debug(f"Executing: {command}")
        try:
            env = {**self.env_variables}
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            output = result.stdout + result.stderr
            
            if check != "ignore" and result.returncode != 0:
                logger.error(f"{error_msg}:\n{output}")
                if check == "raise":
                    raise RuntimeError(f"Command {command!r} failed ({result.returncode=}): {error_msg}")
            
            return output
        except subprocess.TimeoutExpired:
            msg = f"Command {command!r} timed out after {timeout}s"
            logger.error(msg)
            if check == "raise":
                raise RuntimeError(msg)
            return f"Command timed out after {timeout}s"
    
    def read_file(self, path: Union[str, Path], encoding: Optional[str] = None, errors: Optional[str] = None) -> str:
        """读取文件内容
        
        Args:
            path: 文件路径（相对于 repo_path 或绝对路径）
            encoding: 编码
            errors: 错误处理
            
        Returns:
            文件内容
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.repo_path / file_path
        
        if encoding:
            return file_path.read_text(encoding=encoding, errors=errors or "strict")
        return file_path.read_text(errors=errors or "strict")
    
    def write_file(self, path: Union[str, Path], content: str) -> None:
        """写入文件
        
        Args:
            path: 文件路径（相对于 repo_path 或绝对路径）
            content: 文件内容
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.repo_path / file_path
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    
    def set_env_variables(self, env_variables: Dict[str, str]) -> None:
        """设置环境变量"""
        self.env_variables.update(env_variables)
        logger.debug(f"Set environment variables: {list(env_variables.keys())}")
    
    def execute_command(
        self,
        command: str,
        shell: bool = True,
        check: bool = False,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        """执行命令（独立进程）"""
        env_vars = {**self.env_variables}
        if env:
            env_vars.update(env)
        
        subprocess.run(
            command,
            shell=shell,
            cwd=cwd or str(self.working_dir),
            env=env_vars,
            check=check,
        )


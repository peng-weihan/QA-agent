"""问题陈述处理，基于 SWE-agent 的 problem_statement"""
import hashlib
import logging
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProblemStatement(Protocol):
    """问题陈述协议"""
    id: str
    
    def get_problem_statement(self) -> str: ...


class TextProblemStatement(BaseModel):
    """文本问题陈述"""
    text: str
    id: str = None  # type: ignore
    
    def model_post_init(self, __context) -> None:
        if self.id is None:
            self.id = hashlib.sha256(self.text.encode()).hexdigest()[:8]
    
    def get_problem_statement(self) -> str:
        return self.text


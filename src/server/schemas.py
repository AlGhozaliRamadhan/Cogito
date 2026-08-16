"""
OpenAI-Compatible Pydantic Request & Response Schemas
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field
from src.core.prompt import ChatMessage

class ChatCompletionRequest(BaseModel):
    model: str = "Cogito-0.9.1-15B"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=0)
    repeat_penalty: Optional[float] = Field(default=1.1, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = Field(default=1, ge=1, le=1)

class CompletionRequest(BaseModel):
    model: str = "Cogito-0.9.1-15B"
    prompt: str
    max_tokens: Optional[int] = Field(default=512, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=0)
    repeat_penalty: Optional[float] = Field(default=1.1, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class CreateKeyRequest(BaseModel):
    name: str
    role: str = "user"
    rpm: int = Field(default=30, ge=1)

class RevokeKeyRequest(BaseModel):
    key: str

class EmbeddingRequest(BaseModel):
    model: str = "Cogito-0.9.1-15B"
    input: Union[str, List[str]]

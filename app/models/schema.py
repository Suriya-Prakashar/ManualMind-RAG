from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    reply: str
    sources: Optional[List[Dict[str, Any]]] = None

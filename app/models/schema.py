from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ChatRequest(BaseModel):
    question: str
    stream: bool = True


class ChatResponse(BaseModel):
    reply: str
    sources: Optional[List[Dict[str, Any]]] = None

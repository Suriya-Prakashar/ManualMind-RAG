from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    stream: bool = False


class ChatResponse(BaseModel):
    reply: str
    provider: str
    model: str
    fallback: bool
    stream: bool

from pydantic import BaseModel


class ChatRequest(BaseModel):

    question: str


class ChatResponse(BaseModel):

    replay: str
    provider: str
    model: str
    fallback: bool
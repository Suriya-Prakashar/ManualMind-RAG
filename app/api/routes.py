from fastapi import APIRouter
from app.services.fallback import FallbackService

fallback = FallbackService()
from app.models.schema import (
    ChatRequest,
    ChatResponse
)

from app.services.llm import LLMService

router = APIRouter()

llm = LLMService()


@router.get("/")
def health():

    return {
        "status": "Running",
        "message": "ManualMind API is Live"
    }


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    answer = fallback.generate(
    request.question
    )

    return ChatResponse(
        answer=answer
    )
from fastapi import APIRouter

from app.models.schema import ChatRequest
from app.models.schema import ChatResponse

router = APIRouter()


@router.get("/")

def health():

    return {
        "status": "Running",
        "message": "ManualMind API is Live"
    }


@router.post("/chat", response_model=ChatResponse)

def chat(request: ChatRequest):

    return ChatResponse(
        answer=f"You asked: {request.question}"
    )
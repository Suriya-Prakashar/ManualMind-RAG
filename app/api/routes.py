from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schema import ChatRequest, ChatResponse
from app.services.fallback import FallbackService
from app.services.streaming import StreamingService

router = APIRouter()
streaming = StreamingService()
fallback = FallbackService()


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
    # -------------------------
    # Streaming Response
    # -------------------------
    if request.stream:
        print("[DEBUG] Streaming function is working properly. Starting stream...")
        return StreamingResponse(
            streaming.stream(request.question),
            media_type="application/x-ndjson"
        )

    # -------------------------
    # Normal Response
    # -------------------------
    print("[DEBUG] Normal response function is working properly. Generating response...")
    answer_dict = fallback.generate(request.question)
    answer_dict["stream"] = False
    print(f"[DEBUG] Generated response successfully using provider: {answer_dict.get('provider')}, model: {answer_dict.get('model')}")
    return ChatResponse(**answer_dict)
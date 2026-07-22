import json
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from app.models.schema import ChatRequest, ChatResponse
from app.services.fallback import FallbackService
from app.services.streaming import StreamingService
from app.rag.pipeline import RAGPipeline

router = APIRouter()
streaming = StreamingService()
fallback = FallbackService()
rag_pipeline = RAGPipeline()


@router.get("/")
def health():
    return {
        "status": "Running",
        "message": "ManualMind API is Live"
    }


@router.post(
    "/chat"
)
def chat(request: ChatRequest, response: Response):
    # Retrieve context and sources using RAG
    print(f"[DEBUG] Retrieving context for question: {request.question}")
    chunks = rag_pipeline.query(request.question)
    
    context = "\n\n".join([f"Page {c['page']}:\n{c['text']}" for c in chunks])
    sources = [
        {
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "chunk_id": c.get("chunk_id"),
            "text": c["text"],
            "score": c.get("score", 0.0)
        }
        for c in chunks
    ]
    print(f"[DEBUG] Retrieved {len(sources)} sources from vector store.")

    should_stream = False

    # -------------------------
    # Normal Response (Default)
    # -------------------------
    if not should_stream:
        print("[DEBUG] Generating normal JSON response...")
        try:
            answer_dict = fallback.generate(request.question, context=context)
            reply = answer_dict["reply"]
            provider = answer_dict["provider"]
            model = answer_dict["model"]
            fallback_used = str(answer_dict["fallback"]).lower()
            
            response.headers["X-RAG-Provider"] = provider
            response.headers["X-RAG-Model"] = model
            response.headers["X-RAG-Fallback"] = fallback_used
            response.headers["X-RAG-Stream"] = "false"
            
            print(f"[DEBUG] Generated normal response using provider: {provider}, model: {model}")
            return ChatResponse(reply=reply, sources=sources)
        except Exception as e:
            print(f"[WARNING] Normal response failed ({e}). Automatically activating streaming fallback...")
            should_stream = True

    # -------------------------
    # Streaming Response (Fallback or Explicit)
    # -------------------------
    if should_stream:
        print("[DEBUG] Establishing streaming response...")
        try:
            provider_info, stream_iter, first_chunk = streaming.get_stream_iterator(request.question, context=context)
            
            headers = {
                "X-RAG-Provider": provider_info["provider"],
                "X-RAG-Model": provider_info["model"],
                "X-RAG-Fallback": str(provider_info["fallback"]).lower(),
                "X-RAG-Stream": "true"
            }
            
            return StreamingResponse(
                streaming.generate_chunks(stream_iter, first_chunk, sources),
                media_type="application/x-ndjson",
                headers=headers
            )
        except Exception as e:
            error_reply = f"Error generating answer: {e}"
            headers = {
                "X-RAG-Provider": "None",
                "X-RAG-Model": "None",
                "X-RAG-Fallback": "false",
                "X-RAG-Stream": "true"
            }
            def error_generator():
                yield json.dumps({"sources": sources}) + "\n"
                yield json.dumps({"reply": error_reply}) + "\n"
            
            return StreamingResponse(
                error_generator(),
                media_type="application/x-ndjson",
                headers=headers
            )
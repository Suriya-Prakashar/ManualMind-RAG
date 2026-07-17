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
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest, response: Response, stream: bool = False):
    # Retrieve context and sources using RAG
    print(f"[DEBUG] Retrieving context for question: {request.question}")
    chunks = rag_pipeline.query(request.question)
    
    context = "\n\n".join([f"Page {c['page']}:\n{c['text']}" for c in chunks])
    sources = [
        {
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "score": c.get("score", 0.0)
        }
        for c in chunks
    ]
    print(f"[DEBUG] Retrieved {len(sources)} sources from vector store.")

    # -------------------------
    # Streaming Response
    # -------------------------
    if stream:
        print("[DEBUG] Streaming function is working properly. Establishing stream...")
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

    # -------------------------
    # Normal Response
    # -------------------------
    print("[DEBUG] Normal response function is working properly. Generating response...")
    answer_dict = fallback.generate(request.question, context=context)
    
    # Inject metadata into response headers instead of response body
    response.headers["X-RAG-Provider"] = answer_dict["provider"]
    response.headers["X-RAG-Model"] = answer_dict["model"]
    response.headers["X-RAG-Fallback"] = str(answer_dict["fallback"]).lower()
    response.headers["X-RAG-Stream"] = "false"
    
    print(f"[DEBUG] Generated response successfully using provider: {answer_dict.get('provider')}, model: {answer_dict.get('model')}")
    return ChatResponse(reply=answer_dict["reply"], sources=sources)
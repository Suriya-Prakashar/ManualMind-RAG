from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import (
    GROQ_MODEL,
    GROQ_API_KEY,
)

from app.core.prompt import SYSTEM_PROMPT


def extract_text_from_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif hasattr(part, "text"):
                texts.append(part.text)
            elif hasattr(part, "get") and part.get("text"):
                texts.append(part.get("text"))
        return "".join(texts)
    return str(content)


class LLMService:

    def __init__(self):
        # Strip provider prefix if present, e.g. "groq/llama-3.1-8b-instant" -> "llama-3.1-8b-instant"
        model_name = GROQ_MODEL.split("/")[-1] if "/" in GROQ_MODEL else GROQ_MODEL
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=GROQ_API_KEY,
            temperature=0.3
        )

    def generate(self, question: str) -> dict:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question)
        ]

        response = self.llm.invoke(messages)

        return {
            'reply': extract_text_from_content(response.content),
            'provider': 'Groq',
            'model': GROQ_MODEL,
            'fallback': False
        }

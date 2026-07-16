from litellm import completion

from app.core.config import (
    GROQ_MODEL,
    GROQ_API_KEY,
)

from app.core.prompt import SYSTEM_PROMPT


class LLMService:

    def generate(self, question: str) -> dict:

        response = completion(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3
        )


        return {
            'reply': response.choices[0].message.content,
            'provider': 'Groq',
            'model': GROQ_MODEL,
            'fallback': False
        }
import time
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import FALLBACK_CHAIN, MAX_RETRIES, RETRY_DELAY
from app.core.prompt import SYSTEM_PROMPT


def get_langchain_model(provider_name: str, model_name: str, api_key: str, temperature: float = 0.3):
    model = model_name.split("/")[-1] if "/" in model_name else model_name

    if provider_name.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature
        )
    elif provider_name.lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


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


class FallbackService:

    def generate(self, question: str, context: str = None):

        last_error = None

        user_content = question
        if context:
            user_content = f"Use the following context to answer the question:\n{context}\n\nQuestion: {question}"

        # Loop through providers
        for idx, provider in enumerate(FALLBACK_CHAIN):

            # Retry same provider
            for attempt in range(1, MAX_RETRIES + 1):

                try:

                    print(
                        f"[{provider['model']}] Attempt {attempt}/{MAX_RETRIES}"
                    )

                    llm = get_langchain_model(
                        provider_name=provider["provider"],
                        model_name=provider["model"],
                        api_key=provider["api_key"],
                        temperature=0.3
                    )

                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=user_content)
                    ]

                    response = llm.invoke(messages)

                    print(
                        f"{provider['model']} Success"
                    )

                    return {
                        "reply": extract_text_from_content(response.content),
                        "provider": provider["provider"],
                        "model": provider["model"],
                        "fallback": idx > 0
                    }

                except Exception as e:

                    last_error = e

                    print(
                        f"{provider['model']} Failed (Attempt {attempt}): {e}"
                    )

                    if attempt < MAX_RETRIES:

                        print(
                            f"Retrying in {RETRY_DELAY} seconds..."
                        )

                        time.sleep(RETRY_DELAY)

            print(
                f"Switching to next provider..."
            )

        raise Exception(
            f"All providers failed.\n{last_error}"
        )


      
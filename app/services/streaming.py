import json
import time
from litellm import completion

from app.core.config import FALLBACK_CHAIN, MAX_RETRIES, RETRY_DELAY
from app.core.prompt import SYSTEM_PROMPT


class StreamingService:

    def get_stream_iterator(self, question: str, context: str = None):
        """
        Loops through providers to establish a streaming connection, validating the
        connection by fetching the first chunk to ensure rate limits and errors
        are caught and fallbacks are handled before headers are sent.
        """
        last_error = None

        user_content = question
        if context:
            user_content = f"Use the following context to answer the question:\n{context}\n\nQuestion: {question}"

        for idx, provider in enumerate(FALLBACK_CHAIN):

            for attempt in range(1, MAX_RETRIES + 1):

                try:

                    print(f"[{provider['model']}] Streaming Attempt {attempt}/{MAX_RETRIES}")

                    response = completion(
                        model=provider["model"],
                        api_key=provider["api_key"],
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": user_content
                            }
                        ],
                        stream=True,
                        temperature=0.3
                    )
                    
                    # Validate the stream by fetching the first chunk
                    response_iter = iter(response)
                    try:
                        first_chunk = next(response_iter)
                    except StopIteration:
                        first_chunk = None
                    
                    provider_info = {
                        "provider": provider["provider"],
                        "model": provider["model"],
                        "fallback": idx > 0
                    }
                    return provider_info, response_iter, first_chunk

                except Exception as e:

                    print(f"Streaming error on {provider['model']} (Attempt {attempt}): {e}")

                    last_error = e

                    if attempt < MAX_RETRIES:

                        print(f"Retrying in {RETRY_DELAY} seconds...")

                        time.sleep(RETRY_DELAY)

            print(f"Switching to next provider...")

        raise Exception(f"All Providers Failed: {last_error}")

    def generate_chunks(self, response_iter, first_chunk, sources: list = None):
        """
        Generates streaming output chunks containing metadata followed by replies.
        """
        # Yield metadata first (sources list only, provider info is in headers)
        yield json.dumps({
            "sources": sources
        }) + "\n"

        # Yield the first chunk's content if present
        if first_chunk and first_chunk.choices and first_chunk.choices[0].delta.content:
            yield json.dumps({
                "reply": first_chunk.choices[0].delta.content
            }) + "\n"

        # Yield the remaining chunks
        for chunk in response_iter:
            if (
                chunk.choices
                and chunk.choices[0].delta.content
            ):
                yield json.dumps({
                    "reply": chunk.choices[0].delta.content
                }) + "\n"
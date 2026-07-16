import json
import time
from litellm import completion

from app.core.config import FALLBACK_CHAIN, MAX_RETRIES, RETRY_DELAY
from app.core.prompt import SYSTEM_PROMPT


class StreamingService:

    def stream(self, question: str):

        last_error = None

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
                                "content": question
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

                    # Yield metadata first
                    yield json.dumps({
                        "provider": provider["provider"],
                        "model": provider["model"],
                        "fallback": idx > 0,
                        "stream": True
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

                    return

                except Exception as e:

                    print(f"Streaming error on {provider['model']} (Attempt {attempt}): {e}")

                    last_error = e

                    if attempt < MAX_RETRIES:

                        print(f"Retrying in {RETRY_DELAY} seconds...")

                        time.sleep(RETRY_DELAY)

            print(f"Switching to next provider...")

        yield json.dumps({"error": f"All Providers Failed: {last_error}"}) + "\n"
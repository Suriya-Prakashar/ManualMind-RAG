import time
from litellm import completion
from app.core.config import FALLBACK_CHAIN, MAX_RETRIES, RETRY_DELAY
from app.core.prompt import SYSTEM_PROMPT


class FallbackService:

    def generate(self, question: str):

        last_error = None

        # Loop through providers
        for idx, provider in enumerate(FALLBACK_CHAIN):

            # Retry same provider
            for attempt in range(1, MAX_RETRIES + 1):

                try:

                    print(
                        f"[{provider['model']}] Attempt {attempt}/{MAX_RETRIES}"
                    )

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
                        temperature=0.3
                    )

                    print(
                        f"{provider['model']} Success"
                    )

                    return {
                        "reply": response.choices[0].message.content,
                        "provider": provider["provider"],
                        "model": provider["model"],
                        "fallback": idx > 0
                    }

                except Exception as e:

                    last_error = e

                    print(
                        f"{provider['model']} Failed (Attempt {attempt})"
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

      
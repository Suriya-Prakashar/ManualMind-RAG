
from litellm import completion

from app.core.config import FALLBACK_CHAIN
from app.core.prompt import SYSTEM_PROMPT


class FallbackService:

    def generate(self, question: str) -> dict:

        last_error = None

        for idx, provider in enumerate(FALLBACK_CHAIN):

            try:

                print(
                    f"Trying {provider['provider']}..."
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
                    f"{provider['provider']} Success"
                )

                return {
                    'replay': response.choices[0].message.content,
                    'provider': provider['provider'],
                    'model': provider['model'],
                    'fallback': idx > 0
                }

            except Exception as e:

                print(
                    f"{provider['provider']} Failed"
                )

                last_error = e

        raise Exception(last_error)

      
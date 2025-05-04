import sys

from typing import Optional
from openai import OpenAI, OpenAIError, ChatCompletion

from ...constant.constant import large_language_models


open_ai_client: OpenAI = OpenAI(api_key=large_language_models.OPEN_AI_API_KEY)


def query_chatgpt(prompt: str, model: str) -> Optional[str]:
    """
    Queries the answer to a prompt from ChatGPT.
    """
    try:
        chat_gpt_response: ChatCompletion = open_ai_client.chat.completions.create(
            model=model,
            store=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return chat_gpt_response.choices[0].message.content

    except OpenAIError as e:
        sys.stderr.write(f"Error requesting answer from ChatGPT: {e}")

    return None

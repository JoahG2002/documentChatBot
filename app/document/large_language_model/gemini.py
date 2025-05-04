import sys

from google import genai
from typing import Optional
from google.genai.client import Client
from google.genai.types import GenerateContentResponse

from ...constant.constant import large_language_models


client: Client = genai.Client(api_key=large_language_models.GEMINI_API_KEY)


def query_gemini(prompt: str, model: str = "gemini-2.0-flash") -> Optional[str]:
    """
    Queries the answer to a prompt from Gemini.
    """
    try:
        response: GenerateContentResponse = client.models.generate_content(
            model=model,
            contents=prompt
        )

        return response.text

    except Exception as e:
        sys.stderr.write(f"Error requesting answer from Gemini: {e}")

    return None

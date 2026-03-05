import google.generativeai as genai
from .config import GEMINI_API_KEY, EMBED_MODEL

genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text: str) -> list:
    """Generate embedding for a single text string."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']
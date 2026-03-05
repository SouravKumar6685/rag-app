import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "my_docs"
VECTOR_SIZE = 768  # embedding-001l
EMBED_MODEL = "models/gemini-embedding-001"
GEN_MODEL = "gemini-1.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
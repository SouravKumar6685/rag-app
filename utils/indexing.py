import time
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models
from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .embeddings import get_embedding
from .qdrant_ops import QdrantManager

def chunk_text(text: str, source: str):
    """Split text into chunks and return list of dicts with text and source."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [{"text": chunk, "source": source} for chunk in chunks]

def index_documents(docs, progress_callback=None):
    """
    Index a list of document dicts (each with 'text' and 'source').
    progress_callback: optional function to call with (current, total).
    """
    qdrant = QdrantManager()
    qdrant.recreate_collection()

    points = []
    total = len(docs)
    for i, doc in enumerate(docs):
        embedding = get_embedding(doc["text"])
        point_id = str(uuid4())
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": doc["text"], "source": doc["source"]}
            )
        )
        time.sleep(0.5)  # be gentle with free tier

        # Upsert in batches of 50
        if len(points) >= 50:
            qdrant.upsert_points(points)
            points = []

        if progress_callback:
            progress_callback(i + 1, total)

    if points:
        qdrant.upsert_points(points)
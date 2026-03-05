import google.generativeai as genai
from .config import GEMINI_API_KEY, GEN_MODEL
from .embeddings import get_embedding
from .qdrant_ops import QdrantManager

genai.configure(api_key=GEMINI_API_KEY)

def search_qdrant(query, top_k=5):
    qdrant = QdrantManager()
    query_vector = get_embedding(query)
    hits = qdrant.search(query_vector, top_k)
    return [hit.payload["text"] for hit in hits]

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {query}

Answer:"""
    model = genai.GenerativeModel(GEN_MODEL)
    response = model.generate_content(prompt)
    return response.text

def rag_answer(query):
    chunks = search_qdrant(query)
    if not chunks:
        return "No relevant documents found. Please upload and index documents first."
    return generate_answer(query, chunks)
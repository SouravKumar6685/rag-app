from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, VECTOR_SIZE

class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = COLLECTION_NAME

    def recreate_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )

    def upsert_points(self, points):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector, top_k=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return results

    def get_collection_info(self):
        """Return collection info (e.g., number of points)."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return collection
        except Exception:
            return None
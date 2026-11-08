import numpy as np
from embeddings.embedding_interfaces.query_embedding import QueryEmbedder
from retrieval.retrieval_interfaces.retriever import Retriever
from data.query import RetrievalQuery
from storage.storage_interfaces.memory_store import MemoryStore
from data.retrieved_result import RetrievalResult
from data.item import MemoryItem

class LocalCosineRetriever(Retriever):

    def __init__(
        self,
        memory_store: MemoryStore,
        query_embedder: QueryEmbedder,
    ):
        self._memory_store = memory_store
        self._query_embedder = query_embedder

    def retrieve(
    self,
    query: RetrievalQuery,
    k: int,
) -> list[RetrievalResult]:

      if k <= 0:
        raise ValueError("k must be greater than 0.")

      items = self._memory_store.load_all()

      if not items:
        return []

      query_embedding = self._query_embedder.embed(query)

      similarities = []

      for item in items:
        if (
            item.embedding.embedding_id != query_embedding.embedding_id
            or item.embedding.dimension != query_embedding.dimension
        ):
            raise ValueError("Embedding mismatch.")

        similarity = self._cosine_similarity(
            query_embedding.vector,
            item.embedding.vector,
        )

        result = RetrievalResult(
            item=item,
            score=similarity,
        )

        similarities.append(result)
      # sort the similarities in descending order
      # for tie breaking do deterministic ordering by item id
      similarities.sort(
    key=lambda x: (-x.score, x.item.segment.segment_id)
)

      return similarities[:k]

    def _cosine_similarity(
        self,
        query_vector: list[float],
        memory_vector: list[float],
    ) -> float:
        query_vector = np.array(query_vector)
        memory_vector = np.array(memory_vector)
        if np.linalg.norm(query_vector) == 0 or np.linalg.norm(memory_vector) == 0:
            return 0.0
        return float(np.dot(query_vector, memory_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(memory_vector)))
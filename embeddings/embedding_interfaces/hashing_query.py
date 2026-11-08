from sklearn.feature_extraction.text import HashingVectorizer
from data.query import RetrievalQuery, QueryEmbedding
from embeddings.embedding_interfaces.query_embedding import QueryEmbedder


class HashingInstructionQueryEmbedder(QueryEmbedder):

    def __init__(
        self,
        dimension: int = 256,
    ):
      self._dimension = dimension if dimension > 0 else 256
      self._vectorizer = HashingVectorizer(
              n_features=dimension,
              alternate_sign=False,
              norm="l2",
              )

    @property
    def embedding_id(self) -> str:
        return f"hashing-instruction-v1-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        query: RetrievalQuery,
    ) -> QueryEmbedding:
        eps_memory_query = query.instruction
        if not eps_memory_query:
            raise ValueError("The instruction in the retrieval query cannot be empty.")
        vector = self._vectorizer.transform([eps_memory_query])
        dense_1d_vector = vector.toarray().ravel()
        float_vector = [float(x) for x in dense_1d_vector]
        if len(float_vector) != self._dimension:
            raise ValueError(f"The generated vector length {len(float_vector)} does not match the expected dimension {self._dimension}.")
        return QueryEmbedding(
            vector=float_vector,
            embedding_id=self.embedding_id,
            dimension=self.dimension,
        )

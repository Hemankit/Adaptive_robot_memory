from abc import ABC, abstractmethod
from data.query import RetrievalQuery, QueryEmbedding

class QueryEmbedder(ABC):

    @property
    @abstractmethod
    def embedding_id(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def embed(
        self,
        query: RetrievalQuery,
    ) -> QueryEmbedding:
        ...
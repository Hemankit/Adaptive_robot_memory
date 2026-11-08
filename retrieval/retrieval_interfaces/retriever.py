from abc import ABC, abstractmethod
from data.query import RetrievalQuery
from data.retrieved_result import RetrievalResult

class Retriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: RetrievalQuery,
        k: int,
    ) -> list[RetrievalResult]:
        ...
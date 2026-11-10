from abc import ABC, abstractmethod
from data.segment import SegmentData, SegmentEmbedding


class SegmentEmbedder(ABC):

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
        segment: SegmentData,
    ) -> SegmentEmbedding:
        ...
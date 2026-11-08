from abc import ABC, abstractmethod
from data.segment_data import SegmentData
from data.embedding import SegmentEmbedding


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
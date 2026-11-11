from abc import ABC, abstractmethod

from data.event import RetrievalEventRecord


class RetrievalEventStore(ABC):

    @abstractmethod
    def save(
        self,
        episode_id: str,
        events: list[RetrievalEventRecord],
    ) -> None:
        ...

    @abstractmethod
    def load(
        self,
        episode_id: str,
    ) -> list[RetrievalEventRecord]:
        ...
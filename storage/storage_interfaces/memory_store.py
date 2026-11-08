from abc import ABC, abstractmethod

from data.item import MemoryItem


class MemoryStore(ABC):

    @abstractmethod
    def save(
        self,
        items: list[MemoryItem],
    ) -> None:
        ...

    @abstractmethod
    def load_all(
        self,
    ) -> list[MemoryItem]:
        ...

    @abstractmethod
    def load(
        self,
        segment_id: str,
    ) -> MemoryItem:
        ...
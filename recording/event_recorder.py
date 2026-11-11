from abc import ABC, abstractmethod

from data.event import RetrievalEvent
from data.event import RetrievalEvent

class RetrievalEventRecorder(ABC):

    @abstractmethod
    def record(
        self,
        event: RetrievalEvent,
    ) -> None:
        ...

    @abstractmethod
    def get_events(
        self,
    ) -> list[RetrievalEvent]:
        ...

    @abstractmethod
    def clear(
        self,
    ) -> None:
        ...
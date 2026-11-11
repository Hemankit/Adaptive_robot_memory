from recording.event_recorder import RetrievalEventRecorder
from data.event import RetrievalEvent

class InMemoryRetrievalEventRecorder(RetrievalEventRecorder):

    def __init__(self):
        self._events: list[RetrievalEvent] = []

    def record(
        self,
        event: RetrievalEvent,
    ) -> None:
        return self._events.append(
            event.model_copy(deep=True)
        )

    def get_events(
        self,
    ) -> list[RetrievalEvent]:
        return [
            event.model_copy(deep=True)
            for event in self._events
        ]

    def clear(
        self,
    ) -> None:
        self._events.clear()
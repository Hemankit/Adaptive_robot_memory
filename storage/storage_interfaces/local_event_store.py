from pathlib import Path
from storage.storage_interfaces.event_store import RetrievalEventStore
from data.event import RetrievalEventRecord
import json


class LocalRetrievalEventStore(RetrievalEventStore):

    def __init__(
        self,
        root_dir: str | Path,
    ):
        self.root_dir = Path(root_dir)

    def save(
    self,
    episode_id: str,
    events: list[RetrievalEventRecord],
) -> None:
     if not episode_id:
        raise ValueError("Episode ID cannot be empty.")

     for step_index, event in enumerate(events):
        if event.episode_id != episode_id:
            raise ValueError("Event episode_id does not match episode_id.")

        if event.step_index != step_index:
            raise ValueError(
                f"Expected step_index {step_index}, got {event.step_index}."
            )

     episode_dir = self.root_dir / episode_id
     episode_dir.mkdir(parents=True, exist_ok=True)

     episode_file = episode_dir / "retrieval_events.json"
     with open(episode_file, "w", encoding="utf-8") as f:
        json.dump(
            [event.model_dump(mode="json") for event in events],
            f,
            indent=2,
        )

    def load(
        self,
        episode_id: str,
    ) -> list[RetrievalEventRecord]:
        """Load retrieval events for the given episode ID."""
        episode_file = self.root_dir / episode_id / "retrieval_events.json"
        if not episode_file.exists():
            raise FileNotFoundError(f"No retrieval events found for episode {episode_id}.")

        with open(episode_file, "r", encoding="utf-8") as f:
            events_data = json.load(f)

        return [RetrievalEventRecord.model_validate(event) for event in events_data]
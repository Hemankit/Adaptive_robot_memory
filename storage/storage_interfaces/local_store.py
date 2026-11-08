from pathlib import Path
from data.item import MemoryItem
from storage.storage_interfaces.memory_store import MemoryStore
import json

class LocalMemoryStore(MemoryStore):

    def __init__(
        self,
        root_dir: str | Path,
    ):
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        # define memory location
        self._location = self._root_dir / "memory"

    def save(
    self,
    items: list[MemoryItem],
) -> None:
      seen_segment_ids = set()
      embedding_ids = set()
      dimensions = set()

      for item in items:
        segment_id = item.segment.segment_id
        embedding = item.embedding

        # 1. Unique segment IDs
        if segment_id in seen_segment_ids:
            raise ValueError(
                f"Duplicate segment_id: {segment_id}"
            )
        seen_segment_ids.add(segment_id)

        # 2. Matching segment and embedding IDs
        if segment_id != embedding.segment_id:
            raise ValueError(
                f"Segment ID mismatch: {segment_id} != {embedding.segment_id}"
            )

        # 3. Vector length matches dimension
        if len(embedding.vector) != embedding.dimension:
            raise ValueError(
                f"Vector length mismatch for segment {segment_id}: "
                f"expected {embedding.dimension}, "
                f"got {len(embedding.vector)}"
            )

        embedding_ids.add(embedding.embedding_id)
        dimensions.add(embedding.dimension)

    # 4. Consistent embedding ID
      if len(embedding_ids) > 1:
        raise ValueError(
            f"Multiple embedding_ids found: {embedding_ids}"
        )

    # 5. Consistent embedding dimension
      if len(dimensions) > 1:
        raise ValueError(
            f"Multiple embedding dimensions found: {dimensions}"
        )

      payload = [
        item.model_dump(mode="json")
        for item in items
    ]

      with self._location.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    def load_all(
        self,
    ) -> list[MemoryItem]:
        with self._location.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        items = [
            MemoryItem.model_validate(item)
            for item in payload
        ]
        return items

    def load(
        self,
        segment_id: str,
    ) -> MemoryItem:
        all_items = self.load_all()
        for item in all_items:
            if item.segment.segment_id == segment_id:
                return item
        raise ValueError(f"Segment ID {segment_id} not found")
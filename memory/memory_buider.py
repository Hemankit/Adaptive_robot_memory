



from data.item import MemoryItem
from trajectory_segmentation.segment_loader import SegmentLoader
from embeddings.embedding_interfaces.segment_embedder import SegmentEmbedder
from storage.storage_interfaces.memory_store import MemoryStore
from trajectory_segmentation.segmenter_interfaces.trajectory_segmenter import TrajectorySegmenter
from trajectory_segmentation.storage_interfaces.trajectory_store import TrajectoryStore


class MemoryBuilder:

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        segmenter: TrajectorySegmenter,
        segment_loader: SegmentLoader,
        embedder: SegmentEmbedder,
        memory_store: MemoryStore,
    ):
        self._trajectory_store = trajectory_store
        self._segmenter = segmenter
        self._segment_loader = segment_loader
        self._embedder = embedder
        self._memory_store = memory_store

    def build(
        self,
        episode_ids: list[str],
    ) -> list[MemoryItem]:
        # Reject duplicate episode_ids
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("Duplicate episode_ids found")
        memory_items: list[MemoryItem] = []
        seen_segment_ids: set[str] = set()
        # load the episoderecord and steprecords
        for episode_id in episode_ids:
            episode =self._trajectory_store.load_episode(episode_id)
            steps = self._trajectory_store.load_steps(episode_id)
            # skip episodes with success not true
            if not episode.success:
                continue
            segments = self._segmenter.segment(episode=episode, steps=steps)
            for segment in segments:
                segment_data = self._segment_loader.load(segment)
                embedding = self._embedder.embed(segment_data)
                memory_item = MemoryItem(segment=segment, embedding=embedding)
                # validate segment_ids are unique across the entire memory build.
                if segment.segment_id in seen_segment_ids:
                    raise ValueError("Duplicate segment IDs found")
                seen_segment_ids.add(segment.segment_id)
                memory_items.append(memory_item)

        self._memory_store.save(memory_items)
        return memory_items

        

import pytest

from data.episode import EpisodeRecord
from data.query import RetrievalQuery
from data.record import ObservationRecord
from data.step import StepRecord
from embeddings.embedding_interfaces.hashing_embedder import HashingInstructionEmbedder
from embeddings.embedding_interfaces.hashing_query import HashingInstructionQueryEmbedder
from memory.memory_buider import MemoryBuilder
from retrieval.retrieval_interfaces.local_cosine import LocalCosineRetriever

from storage.storage_interfaces.local_store import LocalMemoryStore
from trajectory_segmentation.fixed_window import FixedWindowSegmenter
from trajectory_segmentation.local_trajectory_store import LocalTrajectoryStore
from trajectory_segmentation.segment_loader import SegmentLoader

NUM_STEPS = 5


def persist_episode(
    trajectory_store,
    episode_id: str,
    instruction: str,
    success: bool,
):
    trajectory_store.save_episode(
        EpisodeRecord(
            episode_id=episode_id,
            task_id=f"task_{episode_id}",
            instruction=instruction,
            num_steps=NUM_STEPS,
            success=success,
            total_reward=1.0 if success else 0.0,
            terminated=success,
            truncated=not success,
        )
    )

    trajectory_store.save_steps(
        [
            StepRecord(
                episode_id=episode_id,
                step_index=i,
                action_type="eef_delta_pose",
                action_values={"dx": 0.0},
                reward=1.0 if success and i == NUM_STEPS - 1 else 0.0,
                terminated=success and i == NUM_STEPS - 1,
                truncated=(not success) and i == NUM_STEPS - 1,
                success=success if i == NUM_STEPS - 1 else None,
            )
            for i in range(NUM_STEPS)
        ]
    )

    trajectory_store.save_observations(
        [
            ObservationRecord(
                episode_id=episode_id,
                observation_index=i,
                rgb_artifacts={
                    "agentview": f"{episode_id}/{i}"
                },
            )
            for i in range(NUM_STEPS + 1)
        ]
    )


def test_memory_can_be_built_and_retrieved(tmp_path):
  
  trajectory_store = LocalTrajectoryStore(
    root_dir=tmp_path / "trajectories",
)

  successful_episode_a = "episode_a"
  successful_episode_b = "episode_b"
  failed_episode_c = "episode_c"
  persist_episode(
    trajectory_store, successful_episode_a, "Place the mug inside the drawer.", True
  )
  persist_episode(
    trajectory_store, successful_episode_b, "Open the top drawer of the cabinet.", True
  )
  persist_episode(
    trajectory_store, failed_episode_c, "Turn on the stove.", False
  )

  segment_loader = SegmentLoader(
    trajectory_store=trajectory_store,
)

  segmenter = FixedWindowSegmenter(
    window_size=4,
    stride=2,
)

  segment_embedder = HashingInstructionEmbedder(
    dimension=256,
)

  memory_store = LocalMemoryStore(
    root_dir=tmp_path / "memory",
)

  builder = MemoryBuilder(
    trajectory_store=trajectory_store,
    segmenter=segmenter,
    segment_loader=segment_loader,
    embedder=segment_embedder,
    memory_store=memory_store,
)
  # build the memory
  items = builder.build(
    episode_ids=[
      successful_episode_a,
      successful_episode_b,
      failed_episode_c,
    ],
)

  # assert
  assert len(items) > 0
  persisted_items = memory_store.load_all()

  assert len(persisted_items) == len(items)
  for item in persisted_items:
    assert item.segment.segment_id == item.embedding.segment_id
    assert item.embedding.dimension == 256
    assert len(item.embedding.vector) == 256
  assert all(
  item.segment.episode_id != failed_episode_c
  for item in persisted_items
)
  for item in persisted_items:
    assert item.segment.start_step >= 0
    assert item.segment.end_step >= item.segment.start_step

  segment_ids = [
    item.segment.segment_id
    for item in persisted_items
]

  assert len(segment_ids) == len(set(segment_ids))

  query_embedder = HashingInstructionQueryEmbedder(
    dimension=256,
)

  retriever = LocalCosineRetriever(
    memory_store=memory_store,
    query_embedder=query_embedder,
)
  query = RetrievalQuery(
    instruction="Place the mug inside the drawer."
)
  results = retriever.retrieve(
    query=query,
    k=3,
)
  assert len(results) > 0
  assert len(results) <= 3

  scores = [
    result.score
    for result in results
]

  assert scores == sorted(
    scores,
    reverse=True,
)
  
  assert results[0].item.segment.instruction == (
    "Place the mug inside the drawer."
)

  assert (
    query_embedder.embedding_id
    == segment_embedder.embedding_id
)
  assert (
    query_embedder.dimension
    == segment_embedder.dimension
)
  large_k_results = retriever.retrieve(
    query=query,
    k=10_000,
)

  assert len(large_k_results) == len(persisted_items)

  with pytest.raises(ValueError):
    retriever.retrieve(
        query=query,
        k=0,
    )


  # Round up test
  top_item = results[0].item
  segment_data = segment_loader.load(
    top_item.segment
)
  assert (
    segment_data.segment.segment_id
    == top_item.segment.segment_id
)
  assert (
    len(segment_data.observations)
    == len(segment_data.steps) + 1
)


  

  

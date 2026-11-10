from action_selection.action_selection_interfaces.retrieval_augmented_selection import RetrievalAugmentedActionSelector
from adapter.libero_env import LiberoEnvironment
from data.action import Action
from data.obs import Observation
from data.query import RetrievalQuery
from data.rollout import RolloutResult
from data.task import TaskSpec
from data.transition import Transition
import numpy as np
from embeddings.embedding_interfaces.hashing_embedder import HashingInstructionEmbedder
from embeddings.embedding_interfaces.hashing_query import HashingInstructionQueryEmbedder
from memory.memory_buider import MemoryBuilder
from policy.dummy import RecordingActionPolicy
from retrieval.retrieval_interfaces.always_retrieve_controller import AlwaysRetrieveController
from retrieval.retrieval_interfaces.local_cosine import LocalCosineRetriever
from retrieval.retrieval_interfaces.never_retrieve_controller import NeverRetrieveController
from retrieval.retrieval_interfaces.periodic_retrieve_controller import PeriodicRetrieveController
from retrieval.retrieval_interfaces.retrieval_controller import RetrievalController
from sim_evaluation.rollout_runner import RolloutRunner
from storage.storage_interfaces.local_store import LocalMemoryStore
from storage.storage_interfaces.memory_store import MemoryStore
from test_memory_retrieval_integration import persist_episode
from trajectory_segmentation.fixed_window import FixedWindowSegmenter
from trajectory_segmentation.local_trajectory_store import LocalTrajectoryStore
from trajectory_segmentation.segment_loader import SegmentLoader
from trajectory_segmentation.storage_interfaces.trajectory_store import TrajectoryStore

EMBEDDING_DIMENSION = 256
MAX_STEPS = 3
RETRIEVAL_K = 2
ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")


class SegmentCapturingPolicy(RecordingActionPolicy):
    """Records the retrieved segments themselves, not just their count."""

    def __init__(self):
        super().__init__()
        self.retrieved_segment_ids: list[list[str]] = []

    def act(self, task, observation, retrieved_segments):
        self.retrieved_segment_ids.append(
            [segment.segment.segment_id for segment in retrieved_segments]
        )
        return super().act(task, observation, retrieved_segments)


def build_action_selector(
    controller: RetrievalController,
    memory_store: MemoryStore,
    trajectory_store: TrajectoryStore,
    k: int = RETRIEVAL_K,
    policy: RecordingActionPolicy | None = None,
) -> tuple[
    RetrievalAugmentedActionSelector,
    RecordingActionPolicy,
]:
    query_embedder = HashingInstructionQueryEmbedder(
        dimension=EMBEDDING_DIMENSION,
    )
    retriever = LocalCosineRetriever(
        memory_store=memory_store,
        query_embedder=query_embedder,
    )
    segment_loader = SegmentLoader(
        trajectory_store=trajectory_store,
    )
    policy = policy if policy is not None else RecordingActionPolicy()
    selector = RetrievalAugmentedActionSelector(
        controller=controller,
        retriever=retriever,
        segment_loader=segment_loader,
        policy=policy,
        k=k,
    )
    return selector, policy


def build_populated_stores(tmp_path) -> tuple[TrajectoryStore, MemoryStore]:
    """Persist two successful episodes and build a memory store from them."""
    trajectory_store = LocalTrajectoryStore(
        root_dir=tmp_path / "trajectories",
    )
    episode_ids = ["episode_a", "episode_b"]
    persist_episode(
        trajectory_store, episode_ids[0], "Place the mug inside the drawer.", True
    )
    persist_episode(
        trajectory_store, episode_ids[1], "Open the top drawer of the cabinet.", True
    )

    segment_loader = SegmentLoader(
        trajectory_store=trajectory_store,
    )
    memory_store = LocalMemoryStore(
        root_dir=tmp_path / "memory",
    )
    builder = MemoryBuilder(
        trajectory_store=trajectory_store,
        segmenter=FixedWindowSegmenter(
            window_size=4,
            stride=2,
        ),
        segment_loader=segment_loader,
        embedder=HashingInstructionEmbedder(
            dimension=EMBEDDING_DIMENSION,
        ),
        memory_store=memory_store,
    )
    builder.build(episode_ids=episode_ids)
    return trajectory_store, memory_store


def run_short_rollout(
    selector: RetrievalAugmentedActionSelector,
):
    task = TaskSpec(
        task_id="pick_up_black_bowl_and_place_on_plate",
        instruction=(
            "Pick up the black bowl between the plate and the "
            "ramekin and place it on the plate."
        ),
        metadata={
            "libero_suite": "libero_spatial",
            "libero_task_id": 0,
        },
    )
    environment = LiberoEnvironment(suite_name="libero_spatial")
    try:
        runner = RolloutRunner(environment=environment, max_steps=MAX_STEPS)
        return runner.run(
            task=task,
            action_fn=selector.select_action,
            seed=42,
        )
    finally:
        environment.close()


def assert_canonical_rollout(result: RolloutResult) -> None:
    """Every step carries the canonical fixed Action and a canonical Transition."""
    assert len(result.steps) > 0
    for step_index, step in enumerate(result.steps):
        assert step.step_index == step_index

        assert isinstance(step.action, Action)
        assert step.action.action_type == "eef_delta_pose"
        assert set(step.action.values) == set(ACTION_KEYS)
        assert all(value == 0.0 for value in step.action.values.values())

        assert isinstance(step.transition, Transition)
        assert isinstance(step.transition.observation, Observation)
        assert isinstance(step.transition.reward, (float, int, np.floating, np.integer))
        assert np.isfinite(float(step.transition.reward))
        assert isinstance(step.transition.terminated, bool)
        assert isinstance(step.transition.truncated, bool)
        assert step.transition.success is None or isinstance(step.transition.success, bool)


def test_rollout_with_never_retrieve(tmp_path):
    trajectory_store, memory_store = build_populated_stores(tmp_path)
    selector, policy = build_action_selector(
        controller=NeverRetrieveController(),
        memory_store=memory_store,
        trajectory_store=trajectory_store,
    )

    result = run_short_rollout(selector)

    assert_canonical_rollout(result)
    assert len(result.steps) > 0
    assert len(policy.retrieval_counts) == len(result.steps)
    assert all(
        count == 0
        for count in policy.retrieval_counts
    )


def test_rollout_with_always_retrieve(tmp_path):
    trajectory_store, memory_store = build_populated_stores(tmp_path)
    # The memory must hold at least k items for every retrieval to return k.
    assert len(memory_store.load_all()) >= RETRIEVAL_K

    selector, policy = build_action_selector(
        controller=AlwaysRetrieveController(),
        memory_store=memory_store,
        trajectory_store=trajectory_store,
        k=RETRIEVAL_K,
    )

    result = run_short_rollout(selector)

    assert_canonical_rollout(result)
    assert len(result.steps) > 0
    assert len(policy.retrieval_counts) == len(result.steps)
    assert all(
        count > 0
        for count in policy.retrieval_counts
    )
    assert all(
        count == RETRIEVAL_K
        for count in policy.retrieval_counts
    )


def test_rollout_with_periodic_retrieve(tmp_path):
    trajectory_store, memory_store = build_populated_stores(tmp_path)
    assert len(memory_store.load_all()) >= RETRIEVAL_K

    selector, policy = build_action_selector(
        controller=PeriodicRetrieveController(
            interval=2,
        ),
        memory_store=memory_store,
        trajectory_store=trajectory_store,
        k=RETRIEVAL_K,
    )

    result = run_short_rollout(selector)

    assert_canonical_rollout(result)
    assert len(result.steps) > 0
    assert len(policy.retrieval_counts) == len(result.steps)
    for step_index, count in enumerate(policy.retrieval_counts):
        if step_index % 2 == 0:
            assert count > 0
        else:
            assert count == 0


def test_selector_preserves_retrieval_ranking(tmp_path):
    trajectory_store, memory_store = build_populated_stores(tmp_path)
    k = 3
    assert len(memory_store.load_all()) >= k

    instruction = "Place the mug inside the drawer."
    task = TaskSpec(task_id="place_mug", instruction=instruction)
    observation = Observation(
        rgb={"agentview": np.zeros((2, 2, 3), dtype=np.uint8)},
    )

    policy = SegmentCapturingPolicy()
    selector, _ = build_action_selector(
        controller=AlwaysRetrieveController(),
        memory_store=memory_store,
        trajectory_store=trajectory_store,
        k=k,
        policy=policy,
    )
    expected = LocalCosineRetriever(
        memory_store=memory_store,
        query_embedder=HashingInstructionQueryEmbedder(
            dimension=EMBEDDING_DIMENSION,
        ),
    ).retrieve(
        query=RetrievalQuery(instruction=instruction),
        k=k,
    )

    selector.select_action(task=task, observation=observation, step_index=0)

    # The policy sees exactly the retriever's top-k, in the retriever's order.
    assert policy.retrieval_counts == [k]
    assert policy.retrieved_segment_ids == [
        [result.item.segment.segment_id for result in expected]
    ]
    scores = [result.score for result in expected]
    assert scores == sorted(scores, reverse=True)
    assert expected[0].item.segment.instruction == instruction

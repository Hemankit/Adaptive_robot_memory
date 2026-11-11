
import pytest

from action_selection.action_selection_interfaces.retrieval_augmented_selection import (
    RetrievalAugmentedActionSelector,
)
from adapter.libero_env import LiberoEnvironment
from data.action import Action
from data.experiment import EpisodeRunSpec, ExperimentSpec
from data.task import TaskSpec
from embeddings.embedding_interfaces.hashing_embedder import (
    HashingInstructionEmbedder,
)
from embeddings.embedding_interfaces.hashing_query import (
    HashingInstructionQueryEmbedder,
)
from evaluation.eps_metrics_calc import EpisodeMetricsCalculator
from evaluation.experiment_metrics_calc import ExperimentMetricsCalculator
from memory.memory_buider import MemoryBuilder
from orchestration.runner import ExperimentRunner
from policy.dummy import FixedActionPolicy
from recording.in_memory_recorder import InMemoryRetrievalEventRecorder
from retrieval.retrieval_interfaces.always_retrieve_controller import (
    AlwaysRetrieveController,
)
from retrieval.retrieval_interfaces.local_cosine import LocalCosineRetriever
from sim_evaluation.rollout_runner import RolloutRunner
from storage.storage_interfaces.local_event_store import LocalRetrievalEventStore
from storage.storage_interfaces.local_memory_store import LocalMemoryStore
from test_memory_retrieval_integration import persist_episode
from trajectory_segmentation.fixed_window import FixedWindowSegmenter
from trajectory_segmentation.local_artifact_store import LocalArtifactStore
from trajectory_segmentation.local_trajectory_store import LocalTrajectoryStore
from trajectory_segmentation.observation_converter import ObservationConverter
from trajectory_segmentation.segment_loader import SegmentLoader
from trajectory_segmentation.trajectory_converter import TrajectoryConverter
from trajectory_segmentation.trajectory_writer import TrajectoryWriter


EMBEDDING_DIMENSION = 256
RETRIEVAL_K = 2
MAX_STEPS = 2

ACTION_KEYS = (
    "dx",
    "dy",
    "dz",
    "dax",
    "day",
    "daz",
    "gripper",
)


def test_experiment_runner_executes_and_persists_multiple_episodes(tmp_path):

    trajectory_root = tmp_path / "trajectories"
    artifact_root = tmp_path / "artifacts"
    retrieval_root = tmp_path / "retrieval_events"
    memory_root = tmp_path / "memory"

    # ---------------------------------------------------------
    # 1. Structured trajectory storage
    # ---------------------------------------------------------

    trajectory_store = LocalTrajectoryStore(
        root_dir=trajectory_root,
    )

    # ---------------------------------------------------------
    # 2. Seed successful episodes used to construct memory
    # ---------------------------------------------------------

    seed_episode_ids = [
        "seed_episode_a",
        "seed_episode_b",
    ]

    persist_episode(
        trajectory_store,
        seed_episode_ids[0],
        "Place the mug inside the drawer.",
        True,
    )

    persist_episode(
        trajectory_store,
        seed_episode_ids[1],
        "Open the top drawer of the cabinet.",
        True,
    )

    # ---------------------------------------------------------
    # 3. Build episodic memory
    # ---------------------------------------------------------

    segment_loader = SegmentLoader(
        trajectory_store=trajectory_store,
    )

    memory_store = LocalMemoryStore(
        root_dir=memory_root,
    )

    memory_builder = MemoryBuilder(
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

    memory_builder.build(
        episode_ids=seed_episode_ids,
    )

    memory_items = memory_store.load_all()

    assert len(memory_items) >= RETRIEVAL_K

    # ---------------------------------------------------------
    # 4. Retriever
    # ---------------------------------------------------------

    retriever = LocalCosineRetriever(
        memory_store=memory_store,
        query_embedder=HashingInstructionQueryEmbedder(
            dimension=EMBEDDING_DIMENSION,
        ),
    )

    # ---------------------------------------------------------
    # 5. Retrieval controller + policy
    # ---------------------------------------------------------

    controller = AlwaysRetrieveController()

    policy = FixedActionPolicy(
        action=Action(
            action_type="eef_delta_pose",
            values={
                key: 0.0
                for key in ACTION_KEYS
            },
        )
    )

    # ---------------------------------------------------------
    # 6. Trajectory persistence pipeline
    # ---------------------------------------------------------

    artifact_store = LocalArtifactStore(
        root_dir=artifact_root,
    )

    observation_converter = ObservationConverter(
        artifact_store=artifact_store,
    )

    trajectory_converter = TrajectoryConverter()

    trajectory_writer = TrajectoryWriter(
        trajectory_store=trajectory_store,
        trajectory_converter=trajectory_converter,
        observation_converter=observation_converter,
    )

    # ---------------------------------------------------------
    # 7. Retrieval-event pipeline
    # ---------------------------------------------------------

    event_recorder = InMemoryRetrievalEventRecorder()

    event_store = LocalRetrievalEventStore(
        root_dir=retrieval_root,
    )

    selector = RetrievalAugmentedActionSelector(
        controller=controller,
        retriever=retriever,
        segment_loader=segment_loader,
        policy=policy,
        k=RETRIEVAL_K,
        event_recorder=event_recorder,
    )

    # ---------------------------------------------------------
    # 8. LIBERO environment + rollout runner
    # ---------------------------------------------------------

    env = LiberoEnvironment(suite_name="libero_spatial")

    rollout_runner = RolloutRunner(
        environment=env,
        max_steps=MAX_STEPS,
    )

    # ---------------------------------------------------------
    # 9. Experiment runner
    # ---------------------------------------------------------

    runner = ExperimentRunner(
        rollout_runner=rollout_runner,
        action_selector=selector,
        trajectory_writer=trajectory_writer,
        trajectory_store=trajectory_store,
        event_recorder=event_recorder,
        event_store=event_store,
        episode_metrics_calculator=EpisodeMetricsCalculator(),
        experiment_metrics_calculator=ExperimentMetricsCalculator(),
    )

    # ---------------------------------------------------------
    # 10. Use ONE already-known-good LIBERO task
    #
    # This is the exact TaskSpec that already passed the
    # LiberoEnvironment integration test.
    # ---------------------------------------------------------

    libero_task = TaskSpec(
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

    # Same task + different seeds is intentional.
    # We are testing experiment orchestration, not task diversity.

    experiment = ExperimentSpec(
        experiment_id="integration-experiment",
        episodes=[
            EpisodeRunSpec(
                episode_id="episode-1",
                task=libero_task,
                seed=42,
            ),
            EpisodeRunSpec(
                episode_id="episode-2",
                task=libero_task,
                seed=43,
            ),
        ],
    )

    # ---------------------------------------------------------
    # 11. Execute
    # ---------------------------------------------------------

    try:
        result = runner.run(
            experiment=experiment,
        )
    finally:
        env.close()

    # ---------------------------------------------------------
    # 12. Experiment result
    # ---------------------------------------------------------

    assert result.experiment_id == "integration-experiment"

    assert len(result.episode_metrics) == 2

    assert result.metrics.num_episodes == 2

    assert [
        metrics.episode_id
        for metrics in result.episode_metrics
    ] == [
        "episode-1",
        "episode-2",
    ]

    # ---------------------------------------------------------
    # 13. Persisted episodes
    # ---------------------------------------------------------

    episode_1 = trajectory_store.load_episode(
        "episode-1"
    )

    episode_2 = trajectory_store.load_episode(
        "episode-2"
    )

    assert episode_1.episode_id == "episode-1"
    assert episode_2.episode_id == "episode-2"

    # ---------------------------------------------------------
    # 14. Persisted steps
    # ---------------------------------------------------------

    steps_1 = trajectory_store.load_steps(
        "episode-1"
    )

    steps_2 = trajectory_store.load_steps(
        "episode-2"
    )

    assert len(steps_1) > 0
    assert len(steps_2) > 0

    assert len(steps_1) <= MAX_STEPS
    assert len(steps_2) <= MAX_STEPS

    # ---------------------------------------------------------
    # 15. Persisted observations
    # ---------------------------------------------------------

    observations_1 = (
        trajectory_store.load_observations(
            "episode-1"
        )
    )

    observations_2 = (
        trajectory_store.load_observations(
            "episode-2"
        )
    )

    assert len(observations_1) == len(steps_1) + 1

    assert len(observations_2) == len(steps_2) + 1

    # ---------------------------------------------------------
    # 16. Persisted retrieval events
    # ---------------------------------------------------------

    events_1 = event_store.load(
        "episode-1"
    )

    events_2 = event_store.load(
        "episode-2"
    )

    # Selector should emit exactly one retrieval decision
    # for every action-selection step.
    assert len(events_1) == len(steps_1)
    assert len(events_2) == len(steps_2)

    # AlwaysRetrieveController means every decision retrieves.
    assert all(
        event.retrieval_performed
        for event in events_1
    )

    assert all(
        event.retrieval_performed
        for event in events_2
    )

    # Recorder must have been cleared between episodes.
    assert [
        event.step_index
        for event in events_1
    ] == list(range(len(events_1)))

    assert [
        event.step_index
        for event in events_2
    ] == list(range(len(events_2)))

    # Memory contains at least k items, so every retrieval
    # should return k segment IDs and k scores.
    assert all(
        len(event.retrieved_segment_ids)
        == RETRIEVAL_K
        for event in events_1
    )

    assert all(
        len(event.retrieved_segment_ids)
        == RETRIEVAL_K
        for event in events_2
    )

    assert all(
        len(event.scores) == RETRIEVAL_K
        for event in events_1
    )

    assert all(
        len(event.scores) == RETRIEVAL_K
        for event in events_2
    )

    # ---------------------------------------------------------
    # 17. Episode metrics
    # ---------------------------------------------------------

    metrics_by_episode = {
        metrics.episode_id: metrics
        for metrics in result.episode_metrics
    }

    metrics_1 = metrics_by_episode["episode-1"]
    metrics_2 = metrics_by_episode["episode-2"]

    assert metrics_1.num_steps == episode_1.num_steps
    assert metrics_2.num_steps == episode_2.num_steps

    assert metrics_1.total_reward == pytest.approx(
        episode_1.total_reward
    )

    assert metrics_2.total_reward == pytest.approx(
        episode_2.total_reward
    )

    assert metrics_1.retrieval_count == len(events_1)
    assert metrics_2.retrieval_count == len(events_2)

    # Since every step retrieved.
    assert metrics_1.retrieval_frequency == pytest.approx(
        1.0
    )

    assert metrics_2.retrieval_frequency == pytest.approx(
        1.0
    )

    # ---------------------------------------------------------
    # 18. Experiment aggregation
    # ---------------------------------------------------------

    expected_average_reward = (
        metrics_1.total_reward
        + metrics_2.total_reward
    ) / 2

    expected_average_steps = (
        metrics_1.num_steps
        + metrics_2.num_steps
    ) / 2

    expected_average_retrieval_count = (
        metrics_1.retrieval_count
        + metrics_2.retrieval_count
    ) / 2

    assert result.metrics.average_total_reward == pytest.approx(
        expected_average_reward
    )

    assert result.metrics.average_num_steps == pytest.approx(
        expected_average_steps
    )

    assert result.metrics.average_retrieval_count == pytest.approx(
        expected_average_retrieval_count
    )

    assert result.metrics.average_retrieval_frequency == pytest.approx(
        1.0
    )

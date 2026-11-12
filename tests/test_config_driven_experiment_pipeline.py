


from config.loader import load_experiment_spec
from orchestration.experiment_pipeline_builder import ExperimentPipelineBuilder
from storage.storage_interfaces.local_event_store import LocalRetrievalEventStore
from storage.storage_interfaces.local_event_store import LocalRetrievalEventStore
from tests.test_memory_retrieval_integration import persist_episode
from trajectory_segmentation.local_trajectory_store import LocalTrajectoryStore
import pytest

def test_config_driven_experiment_pipeline(tmp_path):
    trajectory_root = tmp_path / "trajectories"
    artifact_root = tmp_path / "artifacts"
    memory_root = tmp_path / "memory"
    retrieval_root = tmp_path / "retrieval_events"

    trajectory_store = LocalTrajectoryStore(
        root_dir=trajectory_root,
    )

    persist_episode(
        trajectory_store=trajectory_store,
        episode_id="seed_episode_a",
        instruction="Place the mug inside the drawer.",
        success=True,
    )

    persist_episode(
        trajectory_store=trajectory_store,
        episode_id="seed_episode_b",
        instruction="Open the top drawer of the cabinet.",
        success=True,
    )

    config_path = tmp_path / "experiment.yaml"

    config_path.write_text(
        f"""
experiment_id: config-driven-integration

episodes:
  - episode_id: episode-001
    seed: 42
    task:
      task_id: pick_up_black_bowl_and_place_on_plate
      instruction: Pick up the black bowl between the plate and the ramekin and place it on the plate.
      metadata:
        libero_suite: libero_spatial
        libero_task_id: 0

  - episode_id: episode-002
    seed: 43
    task:
      task_id: pick_up_black_bowl_and_place_on_plate
      instruction: Pick up the black bowl between the plate and the ramekin and place it on the plate.
      metadata:
        libero_suite: libero_spatial
        libero_task_id: 0

pipeline:

  rollout:
    max_steps: 2

  segmentation:
    type: fixed_window
    window_size: 4
    stride: 2

  embedding:
    type: hashing_instruction
    dimension: 256

  memory:
    source_episode_ids:
      - seed_episode_a
      - seed_episode_b

  retrieval:
    k: 2

  controller:
    type: always
    interval: null

  policy:
    type: fixed
    fallback_action_type: eef_delta_pose
    fallback_action_values:
      dx: 0.0
      dy: 0.0
      dz: 0.0
      dax: 0.0
      day: 0.0
      daz: 0.0
      gripper: 0.0

  storage:
    trajectory_root: "{trajectory_root.as_posix()}"
    artifact_root: "{artifact_root.as_posix()}"
    memory_root: "{memory_root.as_posix()}"
    retrieval_event_root: "{retrieval_root.as_posix()}"
"""
    )

    spec = load_experiment_spec(
        config_path
    )

    builder = ExperimentPipelineBuilder()

    runtime = builder.build(
        spec
    )

    try:
        result = runtime.run(
            spec
        )
    finally:
        runtime.close()

    assert result.experiment_id == (
        "config-driven-integration"
    )

    assert result.metrics.num_episodes == 2

    assert len(result.episode_metrics) == 2

    assert [
        metrics.episode_id
        for metrics in result.episode_metrics
    ] == [
        "episode-001",
        "episode-002",
    ]

    assert result.metrics.average_retrieval_frequency == pytest.approx(
        1.0
    )
    trajectory_store = LocalTrajectoryStore(
    root_dir=trajectory_root
    )

    episode_1 = trajectory_store.load_episode(
        "episode-001"
    )

    episode_2 = trajectory_store.load_episode(
        "episode-002"
    )

    assert episode_1.episode_id == "episode-001"
    assert episode_2.episode_id == "episode-002"
    event_store = LocalRetrievalEventStore(
        root_dir=retrieval_root
    )

    events_1 = event_store.load(
        "episode-001"
    )

    events_2 = event_store.load(
        "episode-002"
    )

    assert len(events_1) > 0
    assert len(events_2) > 0

    assert all(
        event.retrieval_performed
        for event in events_1
    )

    assert all(
        event.retrieval_performed
        for event in events_2
    )


import numpy as np
from data.action import Action

from adapter.libero_env import LiberoEnvironment
from data import episode, rollout
from data import task
from data.obs import Observation
from data.task import TaskSpec
from sim_evaluation.rollout_runner import RolloutRunner
from trajectory_segmentation.local_trajectory_store import LocalTrajectoryStore
from trajectory_segmentation.local_artifact_store import LocalArtifactStore
from trajectory_segmentation.observation_converter import ObservationConverter
from trajectory_segmentation.storage_interfaces import artifact_store
from trajectory_segmentation.trajectory_converter import TrajectoryConverter
from trajectory_segmentation.trajectory_writer import TrajectoryWriter

ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")

def zero_action() -> Action:
    """A valid no-op end-effector delta action for LIBERO."""
    return Action(
        action_type="eef_delta_pose",
        values={key: 0.0 for key in ACTION_KEYS},
    )


def zero_action_fn(task: TaskSpec, observation: Observation, step_index: int) -> Action:
    return zero_action()

def test_rollout_can_be_persisted_and_reloaded(tmp_path):
    # create seperate storage roots
    trajectory_root = tmp_path / "trajectories"
    artifact_root = tmp_path / "artifacts"
    # construct the storage components
    artifact_store = LocalArtifactStore(
    root_dir=artifact_root,
)

    trajectory_store = LocalTrajectoryStore(
    root_dir=trajectory_root,
)
    # construct converters
    trajectory_converter = TrajectoryConverter()

    observation_converter = ObservationConverter(
    artifact_store=artifact_store,
)
    # construct the trajectory writer
    writer = TrajectoryWriter(
    trajectory_store=trajectory_store,
    trajectory_converter=trajectory_converter,
    observation_converter=observation_converter,
)
    # construct the environment
    environment = LiberoEnvironment(suite_name="libero_spatial")
    # construct the rollout runner
    runner = RolloutRunner(
    environment=environment,
    max_steps=3,
)
    # Reuse the same task definition for the rollout
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
    # run the rollout on defined task and the zero action function
    rollout = runner.run(
    task=task,
    action_fn=zero_action_fn,
    seed=42,
)
    assert len(rollout.steps) > 0
    assert len(rollout.steps) <= 3

    # persist the rollout using the trajectory writer
    episode_id = "integration-test-episode"
    # write the rollout to the trajectory store
    writer.write(
    rollout=rollout,
    episode_id=episode_id,
)
    # Reload the episode
    episode = trajectory_store.load_episode(
    episode_id=episode_id,
)
    assert episode.episode_id == episode_id
    assert episode.task_id == task.task_id
    assert episode.instruction == task.instruction
    assert episode.seed == 42
    assert episode.num_steps == len(rollout.steps)

    expected_reward = sum(
    step.transition.reward
    for step in rollout.steps
)

    assert np.isclose(
    episode.total_reward,
    expected_reward,
)

    # reload steps
    steps = trajectory_store.load_steps(
    episode_id=episode_id,
)
    assert len(steps) == len(rollout.steps)
    assert [
    step.step_index
    for step in steps
] == list(range(len(steps)))
    for persisted, original in zip(
    steps,
    rollout.steps,
):
      assert persisted.step_index == original.step_index
      assert persisted.action_type == original.action.action_type
      assert persisted.action_values == original.action.values
      assert np.isclose(
        persisted.reward,
        original.transition.reward,
    )
      assert persisted.terminated == original.transition.terminated
      assert persisted.truncated == original.transition.truncated
      assert persisted.success == original.transition.success
    # reload observations
      observations = trajectory_store.load_observations(
    episode_id=episode_id,
)
      assert len(observations) == len(steps) + 1
      assert [
    obs.observation_index

    for obs in observations
] == list(range(len(observations)))
    # validate the observation indices are sequential
    persisted_initial = observations[0]
    runtime_initial = rollout.initial_observation
    assert persisted_initial.timestamp == runtime_initial.timestamp

    for camera_name, runtime_rgb in runtime_initial.rgb.items():
      artifact_key = persisted_initial.rgb_artifacts[camera_name]

      loaded_rgb = artifact_store.load_array(
        artifact_key
    )

      np.testing.assert_array_equal(
        loaded_rgb,
        runtime_rgb,
    )

      if runtime_initial.depth is not None:
        assert persisted_initial.depth_artifacts is not None

        for camera_name, runtime_depth in runtime_initial.depth.items():
          artifact_key = persisted_initial.depth_artifacts[camera_name]

          loaded_depth = artifact_store.load_array(
            artifact_key
        )

          np.testing.assert_array_equal(
            loaded_depth,
            runtime_depth,
        )

          # Validate every observation's artifact references

          for observation in observations:
            for artifact_key in observation.rgb_artifacts.values():
              array = artifact_store.load_array(artifact_key)

            assert isinstance(array, np.ndarray)
          if not observation.depth_artifacts:
              continue
          for artifact_key in observation.depth_artifacts.values():
              array = artifact_store.load_array(artifact_key)

              assert isinstance(array, np.ndarray)
          # close environment
    environment.close()
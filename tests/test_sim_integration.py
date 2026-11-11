from adapter.libero_env import LiberoEnvironment
from sim_evaluation.rollout_runner import RolloutRunner
from data.task import TaskSpec
from data.obs import Observation
from data.action import Action
from data.transition import Transition
from data.rollout import RolloutResult, RolloutStep
import numpy as np

ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")


def zero_action() -> Action:
    """A valid no-op end-effector delta action for LIBERO."""
    return Action(
        action_type="eef_delta_pose",
        values={key: 0.0 for key in ACTION_KEYS},
    )


def zero_action_fn(task: TaskSpec, observation: Observation, step_index: int) -> Action:
    return zero_action()

def test_environment_reset():
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

    env = LiberoEnvironment(suite_name="libero_spatial")

    try:
        # 1. Reset using the canonical task.
        obs = env.reset(task, seed=42)

        # 2. Verify canonical Observation type.
        assert isinstance(obs, Observation)

        # 3. Verify required RGB observations.
        assert isinstance(obs.rgb, dict)

        required_cameras = {
            "agentview",
            "robot0_eye_in_hand",
        }

        assert required_cameras.issubset(obs.rgb)

        for camera in required_cameras:
            image = obs.rgb[camera]

            assert isinstance(image, np.ndarray)
            assert image.ndim == 3
            assert image.shape[-1] == 3
            assert image.size > 0

        # 4. Verify raw LIBERO image keys were converted.
        assert "agentview_image" not in obs.rgb
        assert "robot0_eye_in_hand_image" not in obs.rgb

        # 5. Verify canonical optional fields.
        assert obs.depth is None or isinstance(obs.depth, dict)
        assert obs.robot_state is None or isinstance(
            obs.robot_state, (dict, np.ndarray)
        )
        assert obs.object_state is None or isinstance(
            obs.object_state, dict
        )

        # 6. No privileged object state for an image-only agent.
        assert obs.object_state is None

        # 7. Verify raw simulator structures are not exposed.
        assert obs.model_extra is None or not obs.model_extra

        if obs.metadata is not None:
            assert isinstance(obs.metadata, dict)
            assert "raw_obs" not in obs.metadata
            assert "raw_observation" not in obs.metadata
            assert "sim" not in obs.metadata
            assert "env" not in obs.metadata

        # 8. Verify timestamp when available.
        assert obs.timestamp is None or np.isfinite(obs.timestamp)

    finally:
        # 9. Always release simulator resources.
        env.close()

def test_environment_step():
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

    env = LiberoEnvironment(suite_name="libero_spatial")

    try:
        obs = env.reset(task, seed=42)

        action = zero_action()
        transition = env.step(action)

        assert isinstance(transition, Transition)
        assert isinstance(transition.observation, Observation)

        assert isinstance(transition.reward, (float, int, np.floating, np.integer))
        assert np.isfinite(float(transition.reward))
        assert isinstance(transition.terminated, bool)
        assert isinstance(transition.truncated, bool)
        assert transition.success is None or isinstance(transition.success, bool)

    finally:
        env.close()

def test_rollout_runner():
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

    # Record every observation the runner hands to the policy.
    received: list[Observation] = []

    def recording_action_fn(task: TaskSpec, observation: Observation, step_index: int) -> Action:
        received.append(observation)
        return zero_action()

    max_steps = 5
    try:
        runner = RolloutRunner(environment=environment, max_steps=max_steps)
        result = runner.run(
            task=task,
            action_fn=recording_action_fn,
            seed=42,
        )

        assert isinstance(result, RolloutResult)
        assert result.task == task
        assert result.seed == 42
        assert isinstance(result.initial_observation, Observation)

        assert 1 <= len(result.steps) <= max_steps
        assert len(received) == len(result.steps)

        for i, step in enumerate(result.steps):
            assert isinstance(step, RolloutStep)
            assert step.step_index == i
            assert isinstance(step.transition, Transition)
            assert isinstance(step.transition.observation, Observation)
            # The observation recorded for the step is the one the policy saw.
            assert step.observation is received[i]

        # The first step starts from the reset observation.
        assert result.steps[0].observation is result.initial_observation

        # Each later step must start from the previous transition's output,
        # not the initial observation or any other stale observation.
        for i in range(1, len(result.steps)):
            previous = result.steps[i - 1]
            current = result.steps[i]
            assert current.observation is previous.transition.observation
            assert received[i] is previous.transition.observation
            assert current.observation is not result.initial_observation
            assert current.observation is not previous.observation

        # The rollout stops early only on termination or truncation.
        last = result.steps[-1]
        if len(result.steps) < max_steps:
            assert last.transition.terminated or last.transition.truncated
    finally:
        environment.close()

def test_rollout_respects_max_steps():
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

    max_steps = 3
    try:
        runner = RolloutRunner(environment=environment, max_steps=max_steps)
        result = runner.run(
            task=task,
            action_fn=zero_action_fn,
            seed=42,
        )
        assert len(result.steps) <= max_steps
    finally:
        environment.close()

def test_environment_close_is_safe():
    environment = LiberoEnvironment(suite_name="libero_spatial")
    try:
        environment.close()
        environment.close()  # Closing again should be safe and not raise an error.
    except Exception as e:
        assert False, f"Environment close raised an exception: {e}"
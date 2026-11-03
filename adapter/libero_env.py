import numpy as np
from simulation_interface.embodied_environment import EmbodiedEnvironment
from data.action import Action
from data.obs import Observation
from data.task import TaskSpec
from data.transition import Transition
import os
# LIBERO simulator
from libero.libero.envs import OffScreenRenderEnv

# LIBERO task suites and configuration
from libero.libero import benchmark, get_libero_path

class LiberoEnvironment(EmbodiedEnvironment):

    def __init__(self, suite_name: str = "libero_10", render: bool = False):
        # Default suite, used when a task's metadata does not name one.
        self.suite_name = suite_name
        self._render = render
        self.env: OffScreenRenderEnv | None = None
        self.current_task: TaskSpec | None = None

        # Empty means no ground-truth object information is exposed.
        self.agent_object_keys: set[str] = set()

        # Preserve LIBERO's native image orientation by default.
        self.flip_images_vertically = False

    def reset(
        self,
        task: TaskSpec,
        seed: int | None = None,
    ) -> Observation:
        # Resolve the canonical task using _resolve_task()
        canonical_task = self._resolve_task(task)
        # Tear down/reconfigure the previous environment if necessary.
        if self.env is not None:
            self.env.close()
            self.env = None

        # Create the new LIBERO environment
        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            canonical_task.problem_folder,
            canonical_task.bddl_file,
        )

        self.env = OffScreenRenderEnv(bddl_file_name=bddl_file)
        # Seed the environment if a seed is provided.
        if seed is not None:
            self.env.seed(seed)
        # get the initial observation from the environment
        raw_observation = self.env.reset()

        observation = self._convert_observation(raw_observation)
        # Store the current task
        self.current_task = task

        # Return the canonical observation
        return observation

    def step(
        self,
        action: Action,
    ) -> Transition:
        # 1. Ensure environment is initialized
        if self.env is None:
            raise RuntimeError(
                "Environment has not been initialized. Call reset() first."
            )

        # 2. Convert canonical action to LIBERO action
        env_action = self._convert_action(action)

        # 3. Execute action
        raw_observation, reward, done, info = self.env.step(env_action)

        # 4. Convert raw observation
        observation = self._convert_observation(raw_observation)

        # 5. Extract reward
        reward = float(reward)

        # 6. Determine termination
        terminated = bool(done)

        # 7. Determine truncation
        # NOTE: In standard LIBERO, BDDLBaseDomain.step() overrides the underlying
        # simulator's done signal with the task-success check. A time-limit signal
        # is therefore not reliably propagated through done, and LIBERO does not
        # guarantee an info["truncated"] entry.
        truncated = bool(info.get("truncated", False))

        # 8. Extract task success
        success = bool(done)

        # 9. Preserve environment diagnostics
        metadata = dict(info) if info else None

        # 10. Return canonical transition
        return Transition(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            metadata=metadata,
        )

    def close(self) -> None:
        # Release the LIBERO environment and its resources
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None

        # Clear the current task
        self.current_task = None

    def _resolve_task(self, task: TaskSpec):
        meta = task.metadata or {}
        suite_name = meta.get("libero_suite", self.suite_name)

        benchmark_dict = benchmark.get_benchmark_dict()
        if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown suite name: {suite_name}")
        task_suite = benchmark_dict[suite_name]()

        # Prefer the explicit LIBERO index; fall back to the LIBERO task name.
        task_index = meta.get("libero_task_id")
        if task_index is None:
            task_names = task_suite.get_task_names()
            if task.task_id not in task_names:
                raise ValueError(f"Unknown task id: {task.task_id}")
            task_index = task_names.index(task.task_id)
        elif not 0 <= task_index < task_suite.get_num_tasks():
            raise ValueError(
                f"libero_task_id {task_index} is out of range for suite {suite_name}"
            )
        return task_suite.get_task(task_index)


    def _convert_observation(self, raw_observation) -> Observation:
        # 1. Extract RGB observations by camera.
        rgb: dict[str, np.ndarray] = {}
        for k, v in raw_observation.items():
            if not k.endswith("_image"):
                continue
            image = np.asarray(v) # Convert to numpy array
            # Only consider images with 3 channels (RGB)
            if image.ndim == 3 and image.shape[-1] == 3:
                cameraname = k.removesuffix("_image")
                rgb[cameraname] = image.copy()
        depth: dict[str, np.ndarray] = {}
        for k, v in raw_observation.items():
            if not k.endswith("_depth"):
                continue
            depth_image = np.asarray(v) # Convert to numpy array
            if depth_image.ndim == 3 and depth_image.shape[-1] == 1:
                depth_image = depth_image[..., 0] # Squeeze the last dimension if it's 1
            if depth_image.ndim == 2:
                cameraname = k.removesuffix("_depth")
                depth[cameraname] = depth_image.copy()
                 
        if self.flip_images_vertically:
            rgb = {name: np.ascontiguousarray(np.flipud(image)) for name, image in rgb.items()}
            depth = {name: np.ascontiguousarray(np.flipud(image)) for name, image in depth.items()}
        robot_keys = {
            "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_eef_vel_lin",
        "robot0_eef_vel_ang",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_vel",
        }
        robot_state = {key: np.asarray(raw_observation[key]) for key in robot_keys if key in raw_observation}

        object_state = {key: np.asarray(raw_observation[key]) for key in self.agent_object_keys if key in raw_observation}

        timestamp = None
        if self.env is not None:
            sim = getattr(self.env, "sim", None)
            if sim is not None:
                sim_time = getattr(sim.data, "time", None)
                if sim_time is not None:
                    timestamp = float(sim_time)

        metadata = None
        return Observation(
        rgb=rgb,
        depth=depth or None,
        robot_state=robot_state or None,
        object_state=object_state or None,
        timestamp=timestamp,
        metadata=metadata,
    )

            

    def _convert_action(self, action: Action):
        if action.action_type != "eef_delta_pose":
          raise ValueError(
            f"Unsupported LIBERO action type: {action.action_type!r}. "
            "Expected 'eef_delta_pose'."
        )
        required_keys = (
        "dx",
        "dy",
        "dz",
        "dax",
        "day",
        "daz",
        "gripper"
        )
        missing = set(required_keys) - action.values.keys()
        if missing:
            raise ValueError(f"Missing required action values: {sorted(missing)}")
        unexpected = action.values.keys() - set(required_keys)
        if unexpected:
            raise ValueError(
                f"Unexpected action values: {sorted(unexpected)}"
            )
        libero_action = np.asarray([action.values[key] for key in required_keys], dtype=np.float32)
        if not np.all(np.isfinite(libero_action)):
            raise ValueError("Action contains Nan or infinite values")

         # Validate normalized action bounds.
        if np.any((libero_action < -1.0) | (libero_action > 1.0)):
          raise ValueError(
            "LIBERO action values must be within [-1, 1]."
        )

        return libero_action
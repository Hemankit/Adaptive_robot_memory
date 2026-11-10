from collections.abc import Callable
from random import seed

from data import action
from data.action import Action
from data.obs import Observation
from data.rollout import RolloutResult, RolloutStep
from data.task import TaskSpec
from simulation_interface.embodied_environment import EmbodiedEnvironment


class RolloutRunner:
    def __init__(
        self,
        environment: EmbodiedEnvironment,
        max_steps: int,
    ):
        self.environment = environment
        self.max_steps = max_steps if max_steps >= 0 else 0

    def run(
    self,
    task: TaskSpec,
    action_fn: Callable[
        [TaskSpec, Observation, int],
        Action,
    ],
    seed: int | None = None,
) -> RolloutResult:
        # save initial observation
        initial_observation = self.environment.reset(task, seed=seed)
        observation = initial_observation
        steps = []

        # The loop itself enforces max_steps: if the environment has not
        # terminated or truncated by then, the runner simply stops.
        for step_index in range(self.max_steps):
            action = action_fn(task, observation, step_index)
            transition = self.environment.step(action)
            steps.append(
                RolloutStep(
                    step_index=step_index,
                    observation=observation,
                    action=action,
                    transition=transition,
                )
            )
            # update the observation for the next step
            observation = transition.observation
            if transition.terminated or transition.truncated:
                break

        return RolloutResult(
            task=task,
            seed=seed,
            initial_observation=initial_observation,
            steps=steps,
        )
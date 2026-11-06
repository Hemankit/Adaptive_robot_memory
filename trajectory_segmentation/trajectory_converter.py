

from data import rollout
from data.episode import EpisodeRecord
from data.rollout import RolloutResult
from data.step import StepRecord


class TrajectoryConverter:

    def convert_episode(
        self,
        rollout: RolloutResult,
        episode_id: str,
    ) -> EpisodeRecord:
      task_info = rollout.task
      num_steps = len(rollout.steps)

      total_reward = sum(
        step.transition.reward
        for step in rollout.steps
    )

      terminated = (
        rollout.steps[-1].transition.terminated
        if rollout.steps else False
    )

      truncated = (
        rollout.steps[-1].transition.truncated
        if rollout.steps else False
    )
      signals = [
          step.transition.success
          for step in rollout.steps
          if step.transition.success is not None
      ]
      success = any(signals) if signals else None
      # NOTE: the success method below uses the last non-None success signal in the rollout steps
      # success = next(
      #     (
      #         step.transition.success
      #         for step in reversed(rollout.steps)
      #         if step.transition.success is not None
      #     ),
      #     None,
      # )
      
      return EpisodeRecord(
        episode_id=episode_id,
        task_id=task_info.task_id,
        instruction=task_info.instruction,
        seed=rollout.seed,
        num_steps=num_steps,
        success=success,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        metadata=None,
    )

    def convert_steps(
        self,
        rollout: RolloutResult,
        episode_id: str,
    ) -> list[StepRecord]:
        step_records = []

        for step in rollout.steps:
            step_records.append(
                StepRecord(
                    episode_id=episode_id,
                    step_index=step.step_index,
                    action_type=step.action.action_type,
                    action_values=step.action.values,
                    reward=step.transition.reward,
                    terminated=step.transition.terminated,
                    truncated=step.transition.truncated,
                    success=step.transition.success,
                    timestamp=step.observation.timestamp,
                    metadata=None,
                )
            )

        return step_records

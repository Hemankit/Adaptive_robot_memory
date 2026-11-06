
from data.obs import Observation
from data.rollout import RolloutResult
from trajectory_segmentation.observation_converter import ObservationConverter
from trajectory_segmentation.storage_interfaces.trajectory_store import TrajectoryStore
from trajectory_segmentation.trajectory_converter import TrajectoryConverter


class TrajectoryWriter:
    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        trajectory_converter: TrajectoryConverter,
        observation_converter: ObservationConverter,
    ):
        self._trajectory_store = trajectory_store
        self._trajectory_converter = trajectory_converter
        self._observation_converter = observation_converter

    def write(
        self,
        rollout: RolloutResult,
        episode_id: str,
    ) -> None:
        if not episode_id:
            raise ValueError("Episode ID is empty")

        # convert the episode
        episode = self._trajectory_converter.convert_episode(rollout, episode_id)
        # convert steps
        steps = self._trajectory_converter.convert_steps(rollout, episode_id)

        observations = self._collect_observations(rollout)

        # convert observations using the observation converter
        observations = [
            self._observation_converter.convert(obs, episode_id, index)
            for index, obs in enumerate(observations)
        ]

        if not len(observations) == len(steps) + 1:
            raise ValueError("Number of observations does not match number of steps + 1")
        

        # save to trajectory store
        self._trajectory_store.save_episode(episode)
        self._trajectory_store.save_steps(steps)
        self._trajectory_store.save_observations(observations)

      

    def _collect_observations(
        self,
        rollout: RolloutResult,
    ) -> list[Observation]:
        observations = [rollout.initial_observation]
        for step in rollout.steps:
            observations.append(step.transition.observation)
        return observations
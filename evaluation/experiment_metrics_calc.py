

from data.metric import EpisodeMetrics, ExperimentMetrics


class ExperimentMetricsCalculator:

    def calculate(
        self,
        episodes: list[EpisodeMetrics],
    ) -> ExperimentMetrics:
        if not episodes:
            raise ValueError("The list of episodes cannot be empty.")
        episode_ids = [
        episode.episode_id
        for episode in episodes
        ]

        if len(episode_ids) != len(set(episode_ids)):
          raise ValueError("Duplicate episode IDs found in the list of episodes.")
        # calculate success rate for episodes where episode.success is not None
        success_evaluable_episodes = [episode for episode in episodes if episode.success is not None]
        num_success_evaluable_episodes = len(success_evaluable_episodes)
        success_rate = (
            sum(1 for episode in success_evaluable_episodes if episode.success)
            / num_success_evaluable_episodes
            if num_success_evaluable_episodes > 0
            else None
        )

        # calculate average total reward
        average_total_reward = sum(
    episode.total_reward
    for episode in episodes
) / len(episodes)
        # calculate average number of steps
        average_num_steps = sum(
            episode.num_steps
            for episode in episodes
        ) / len(episodes)

        # calculate average retrieval count
        average_retrieval_count = sum(
            episode.retrieval_count
            for episode in episodes
        ) / len(episodes)

        # calculate average retrieval frequency
        average_retrieval_frequency = sum(
            episode.retrieval_frequency
            for episode in episodes
        ) / len(episodes)

        # calculate average retrieval latency
        retrieval_latency_episodes = [episode for episode in episodes if episode.average_retrieval_latency_ms is not None]
        average_retrieval_latency_ms = (
            sum(episode.average_retrieval_latency_ms for episode in retrieval_latency_episodes)
            / len(retrieval_latency_episodes)
            if retrieval_latency_episodes
            else None
        )

        # calculate average retrieval score
        retrieval_score_episodes = [episode for episode in episodes if episode.average_retrieval_score is not None]
        average_retrieval_score = (
            sum(episode.average_retrieval_score for episode in retrieval_score_episodes)
            / len(retrieval_score_episodes)
            if retrieval_score_episodes
            else None
        )

        return ExperimentMetrics(
            num_episodes=len(episodes),
            num_success_evaluable_episodes=num_success_evaluable_episodes,
            success_rate=success_rate,
            average_total_reward=average_total_reward,
            average_num_steps=average_num_steps,
            average_retrieval_count=average_retrieval_count,
            average_retrieval_frequency=average_retrieval_frequency,
            average_retrieval_latency_ms=average_retrieval_latency_ms,
            average_retrieval_score=average_retrieval_score,
        )

from data.episode import EpisodeRecord
from data.event import RetrievalEventRecord
from data.metric import EpisodeMetrics


class EpisodeMetricsCalculator:

    def calculate(
        self,
        episode: EpisodeRecord,
        retrieval_events: list[RetrievalEventRecord],
    ) -> EpisodeMetrics:
        episode_events = [
            event
            for event in retrieval_events
            if event.episode_id == episode.episode_id
        ]

        retrieval_count = sum(
            1 for event in episode_events if event.retrieval_performed
        )
        retrieval_frequency = (
            retrieval_count / episode.num_steps
            if episode.num_steps > 0
            else 0.0
        )

        # Latency is only meaningful when retrieval was performed.
        latencies = [
            event.retrieval_latency_ms
            for event in episode_events
            if event.retrieval_performed
            and event.retrieval_latency_ms is not None
        ]
        average_retrieval_latency_ms = (
            sum(latencies) / len(latencies) if latencies else None
        )

        scores = [
            score
            for event in episode_events
            for score in event.scores
        ]
        average_retrieval_score = (
            sum(scores) / len(scores) if scores else None
        )

        return EpisodeMetrics(
            episode_id=episode.episode_id,
            success=episode.success,
            total_reward=episode.total_reward,
            num_steps=episode.num_steps,
            retrieval_count=retrieval_count,
            retrieval_frequency=retrieval_frequency,
            average_retrieval_latency_ms=average_retrieval_latency_ms,
            average_retrieval_score=average_retrieval_score,
        )
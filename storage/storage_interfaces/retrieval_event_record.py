from data.event import RetrievalEvent, RetrievalEventRecord


def to_retrieval_event_record(
    event: RetrievalEvent,
    episode_id: str,
) -> RetrievalEventRecord:
    return RetrievalEventRecord(
    episode_id=episode_id,
    step_index=event.step_index,
    retrieval_performed=event.retrieval_performed,
    query_instruction=(
        event.query.instruction
        if event.query is not None
        else None
    ),
    retrieved_segment_ids=list(
        event.retrieved_segment_ids
    ),
    scores=list(event.scores),
    retrieval_latency_ms=event.retrieval_latency_ms,
)
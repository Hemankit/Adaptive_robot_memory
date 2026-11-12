from pydantic import BaseModel, Field

class StorageConfig(BaseModel):
    trajectory_root: str = Field(
        ...,
        description="The root directory used for persistent structured trajectory records.",
    )

    artifact_root: str = Field(
        ...,
        description="The root directory used for large observation artifacts.",
    )

    memory_root: str = Field(
        ...,
        description="The root directory used for persisted episodic memory.",
    )

    retrieval_event_root: str = Field(
        ...,
        description="The root directory used for persisted retrieval-event records.",
    )
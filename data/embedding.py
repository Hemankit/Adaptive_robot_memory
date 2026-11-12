from pydantic import BaseModel, Field

class EmbeddingConfig(BaseModel):
    type: str = Field(
        default="hashing_instruction",
        description="The embedding implementation used for trajectory segments and retrieval queries.",
    )

    dimension: int = Field(
        ...,
        gt=0,
        description="The dimensionality of the embedding vectors.",
    )
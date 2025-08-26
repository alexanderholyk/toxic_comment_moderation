from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    comment_text: str = Field(min_length=1, max_length=5000)

class PredictResponse(BaseModel):
    labels: list[str]
    scores: dict[str, float]
    model_version: str
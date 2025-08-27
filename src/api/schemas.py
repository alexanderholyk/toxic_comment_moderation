from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    comment_text: str

class PredictResponse(BaseModel):
    labels: list[str]
    scores: dict[str, float]
    model_version: str = Field(..., description="W&B artifact version")
    request_id: str
    # Fix the 'model_version' protected namespace warning:
    model_config = {"protected_namespaces": ()}
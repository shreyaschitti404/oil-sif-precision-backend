from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    report_id: str | None = None


class AnalysisResult(BaseModel):
    sif_potential: bool
    confidence: float = Field(ge=0.0, le=1.0)
    life_saving_rule: str | None = None
    activity: str | None = None
    hazard: str | None = None
    barrier_failure: str | None = None
    precursor_pattern: str | None = None
    model_version: str
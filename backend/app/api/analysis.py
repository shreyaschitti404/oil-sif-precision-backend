from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.ml_service import analyze_report


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


class AnalyzeRequest(BaseModel):
    text: str
    report_id: str | None = None


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        result = analyze_report(request.text)

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

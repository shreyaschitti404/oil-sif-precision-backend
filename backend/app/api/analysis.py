from fastapi import APIRouter, HTTPException

from backend.app.schemas import AnalyzeRequest, AnalysisResult
from backend.app.services.ml_service import analyze_report


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.post("/analyze", response_model=AnalysisResult)
def analyze(request: AnalyzeRequest):
    try:
        result = analyze_report(request.text)

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

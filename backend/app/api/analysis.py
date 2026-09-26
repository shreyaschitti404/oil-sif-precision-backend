from fastapi import APIRouter, HTTPException

from backend.app.schemas import AnalyzeRequest, AnalysisResult
from backend.app.services.ml_service import analyze_report
from backend.app.services.prediction_service import save_prediction


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.post("/analyze", response_model=AnalysisResult)
def analyze(request: AnalyzeRequest):
    try:
        result = analyze_report(request.text)

        # Save the prediction only when a real model is connected.
        if (
            request.report_id
            and result.get("model_version") != "not-connected"
        ):
            save_prediction(
                report_id=request.report_id,
                sif_potential=result["sif_potential"],
                confidence=result["confidence"],
                life_saving_rule=result["life_saving_rule"],
                hazard=result["hazard"],
                barrier_failure=result["barrier_failure"],
                precursor_pattern=result["precursor_pattern"],
                model_version=result["model_version"],
            )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

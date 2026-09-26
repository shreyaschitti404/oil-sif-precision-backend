from sqlalchemy import text

from backend.app.db.database import engine


def save_prediction(
    report_id: str,
    sif_potential: bool,
    confidence: float | None = None,
    life_saving_rule: str | None = None,
    hazard: str | None = None,
    barrier_failure: str | None = None,
    precursor_pattern: str | None = None,
    model_version: str | None = None,
):
    query = text(
        """
        INSERT INTO predictions (
            report_id,
            sif_potential,
            confidence,
            life_saving_rule,
            hazard,
            barrier_failure,
            precursor_pattern,
            model_version
        )
        VALUES (
            :report_id,
            :sif_potential,
            :confidence,
            :life_saving_rule,
            :hazard,
            :barrier_failure,
            :precursor_pattern,
            :model_version
        )
        RETURNING id
        """
    )

    with engine.begin() as connection:
        result = connection.execute(
            query,
            {
                "report_id": report_id,
                "sif_potential": sif_potential,
                "confidence": confidence,
                "life_saving_rule": life_saving_rule,
                "hazard": hazard,
                "barrier_failure": barrier_failure,
                "precursor_pattern": precursor_pattern,
                "model_version": model_version,
            },
        )

        prediction_id = result.scalar_one()

    return str(prediction_id)
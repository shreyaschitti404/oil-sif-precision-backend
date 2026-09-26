from sqlalchemy import text

from backend.app.db.database import engine


def _to_db_text(value):
    """
    Convert model outputs into text suitable for the PostgreSQL
    prediction columns.

    The ML engine may return lists such as:
    ["Confined Space", "Work Authorisation"]

    PostgreSQL text columns should receive:
    "Confined Space | Work Authorisation"
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        cleaned = [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]
        return " | ".join(cleaned) if cleaned else None

    return str(value)


def save_prediction(
    report_id: str,
    sif_potential: bool,
    confidence: float | None = None,
    life_saving_rule=None,
    hazard=None,
    barrier_failure=None,
    precursor_pattern=None,
    model_version: str | None = None,
):
    """
    Save a prediction against an existing report.

    report_id here is the human-readable reports.report_id value.
    The function resolves it to the UUID stored in reports.id before
    inserting into predictions.report_id.
    """

    if not report_id or not str(report_id).strip():
        raise ValueError("report_id is required.")

    find_report_query = text(
        """
        SELECT id
        FROM reports
        WHERE report_id = :report_id
        LIMIT 1
        """
    )

    insert_query = text(
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
            :report_uuid,
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
        report_uuid = connection.execute(
            find_report_query,
            {"report_id": str(report_id).strip()},
        ).scalar_one_or_none()

        if report_uuid is None:
            raise ValueError(
                f"Report '{report_id}' was not found in the reports table."
            )

        result = connection.execute(
            insert_query,
            {
                "report_uuid": report_uuid,
                "sif_potential": bool(sif_potential),
                "confidence": (
                    float(confidence)
                    if confidence is not None
                    else None
                ),
                "life_saving_rule": _to_db_text(
                    life_saving_rule
                ),
                "hazard": _to_db_text(hazard),
                "barrier_failure": _to_db_text(
                    barrier_failure
                ),
                "precursor_pattern": _to_db_text(
                    precursor_pattern
                ),
                "model_version": model_version,
            },
        )

        prediction_id = result.scalar_one()

    return str(prediction_id)

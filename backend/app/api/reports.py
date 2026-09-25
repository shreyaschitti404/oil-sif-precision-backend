from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from backend.app.db.database import engine

router = APIRouter(
    prefix="/api",
    tags=["Reports"],
)


class ReportCreate(BaseModel):
    report_id: str
    report_text: str
    report_type: str | None = None
    severity: str | None = None
    site: str | None = None
    activity: str | None = None
    facility: str | None = None


@router.get("/reports")
def get_reports():
    query = text("""
        SELECT
            r.report_id,
            r.report_text,
            r.report_date,
            r.report_type,
            r.severity,
            r.facility,
            s.name AS site,
            a.name AS activity
        FROM reports r
        LEFT JOIN sites s
            ON r.site_id = s.id
        LEFT JOIN activities a
            ON r.activity_id = a.id
        ORDER BY r.created_at DESC
    """)

    with engine.connect() as connection:
        result = connection.execute(query)
        reports = result.mappings().all()

    return reports


@router.post("/reports")
def create_report(report: ReportCreate):
    if not report.report_id.strip():
        raise HTTPException(status_code=400, detail="report_id is required")

    if not report.report_text.strip():
        raise HTTPException(status_code=400, detail="report_text is required")

    insert_query = text("""
        INSERT INTO reports (
            report_id,
            report_text,
            report_date,
            report_type,
            severity,
            site_id,
            activity_id,
            facility
        )
        VALUES (
            :report_id,
            :report_text,
            now(),
            :report_type,
            :severity,
            (SELECT id FROM sites WHERE name = :site),
            (SELECT id FROM activities WHERE name = :activity),
            :facility
        )
        RETURNING report_id
    """)

    try:
        with engine.begin() as connection:
            result = connection.execute(
                insert_query,
                {
                    "report_id": report.report_id,
                    "report_text": report.report_text,
                    "report_type": report.report_type,
                    "severity": report.severity,
                    "site": report.site,
                    "activity": report.activity,
                    "facility": report.facility,
                },
            )

            new_report_id = result.scalar_one()

        return {
            "status": "created",
            "report_id": new_report_id,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
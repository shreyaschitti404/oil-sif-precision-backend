from fastapi import APIRouter, HTTPException, Query
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
def get_reports(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str | None = None,
    site: str | None = None,
):
    filters = []

    params = {
        "limit": page_size,
        "offset": (page - 1) * page_size,
    }

    # Search filter
    if search:
        filters.append(
            """
            (
                r.report_id ILIKE :search
                OR r.report_text ILIKE :search
                OR COALESCE(s.name, '') ILIKE :search
                OR COALESCE(r.facility, '') ILIKE :search
            )
            """
        )
        params["search"] = f"%{search}%"

    # Site filter
    if site and site != "all":
        filters.append("s.name = :site")
        params["site"] = site

    where_clause = ""

    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    # Count matching reports
    count_query = text(
        f"""
        SELECT COUNT(*)
        FROM reports r
        LEFT JOIN sites s
            ON r.site_id = s.id
        LEFT JOIN activities a
            ON r.activity_id = a.id
        {where_clause}
        """
    )

    # Get report data + latest ML prediction, if available
    data_query = text(
        f"""
        SELECT
            r.report_id,
            r.report_text,
            r.report_date,
            r.report_type,
            r.severity,
            r.facility,

            s.name AS site,
            a.name AS activity,

            p.sif_potential,
            p.confidence,
            p.life_saving_rule,
            p.hazard,
            p.barrier_failure,
            p.precursor_pattern,
            p.model_version

        FROM reports r

        LEFT JOIN sites s
            ON r.site_id = s.id

        LEFT JOIN activities a
            ON r.activity_id = a.id

        LEFT JOIN LATERAL (
            SELECT
                predictions.sif_potential,
                predictions.confidence,
                predictions.life_saving_rule,
                predictions.hazard,
                predictions.barrier_failure,
                predictions.precursor_pattern,
                predictions.model_version

            FROM predictions

            WHERE predictions.report_id = r.id

            ORDER BY predictions.created_at DESC

            LIMIT 1
        ) p
            ON TRUE

        {where_clause}

        ORDER BY r.created_at DESC

        LIMIT :limit
        OFFSET :offset
        """
    )

    with engine.connect() as connection:
        total = connection.execute(
            count_query,
            params,
        ).scalar_one()

        result = connection.execute(
            data_query,
            params,
        )

        reports = result.mappings().all()

    return {
        "data": reports,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("/reports")
def create_report(report: ReportCreate):
    if not report.report_id.strip():
        raise HTTPException(
            status_code=400,
            detail="report_id is required",
        )

    if not report.report_text.strip():
        raise HTTPException(
            status_code=400,
            detail="report_text is required",
        )

    insert_query = text(
        """
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
        """
    )

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

from fastapi import APIRouter
from sqlalchemy import text

from backend.app.db.database import engine

router = APIRouter(
    prefix="/api",
    tags=["Dashboard"],
)


@router.get("/dashboard")
def get_dashboard():
    with engine.connect() as connection:

        total_reports = connection.execute(
            text("SELECT COUNT(*) FROM reports")
        ).scalar_one()

        critical_reports = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM reports
                WHERE LOWER(COALESCE(severity, '')) = 'critical'
            """)
        ).scalar_one()

        high_reports = connection.execute(
            text("""
                SELECT COUNT(*)
                FROM reports
                WHERE LOWER(COALESCE(severity, '')) = 'high'
            """)
        ).scalar_one()

        reports_by_site = connection.execute(
            text("""
                SELECT
                    COALESCE(s.name, 'Unknown') AS site,
                    COUNT(*) AS count
                FROM reports r
                LEFT JOIN sites s
                    ON r.site_id = s.id
                GROUP BY s.name
                ORDER BY count DESC
            """)
        ).mappings().all()

        reports_by_activity = connection.execute(
            text("""
                SELECT
                    COALESCE(a.name, 'Unknown') AS activity,
                    COUNT(*) AS count
                FROM reports r
                LEFT JOIN activities a
                    ON r.activity_id = a.id
                GROUP BY a.name
                ORDER BY count DESC
            """)
        ).mappings().all()

        reports_by_severity = connection.execute(
            text("""
                SELECT
                    COALESCE(severity, 'Unknown') AS severity,
                    COUNT(*) AS count
                FROM reports
                GROUP BY severity
                ORDER BY count DESC
            """)
        ).mappings().all()

    return {
        "total_reports": total_reports,
        "critical_reports": critical_reports,
        "high_reports": high_reports,
        "reports_by_site": reports_by_site,
        "reports_by_activity": reports_by_activity,
        "reports_by_severity": reports_by_severity,
    }
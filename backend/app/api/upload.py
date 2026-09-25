from fastapi import APIRouter, UploadFile, File, HTTPException
from sqlalchemy import text
import pandas as pd
from io import BytesIO

from backend.app.db.database import engine

router = APIRouter(
    prefix="/api",
    tags=["Upload"],
)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    filename = file.filename.lower()

    if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and XLSX files are supported",
        )

    contents = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        else:
            df = pd.read_excel(BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read file: {exc}",
        ) from exc

    required_columns = ["report_id", "report_text"]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}",
        )

    inserted = 0

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
            :report_date,
            :report_type,
            :severity,
            (SELECT id FROM sites WHERE name = :site),
            (SELECT id FROM activities WHERE name = :activity),
            :facility
        )
        ON CONFLICT (report_id) DO NOTHING
    """)

    with engine.begin() as connection:
        for _, row in df.iterrows():
            connection.execute(
                insert_query,
                {
                    "report_id": str(row["report_id"]),
                    "report_text": str(row["report_text"]),
                    "report_date": (
                        row["report_date"]
                        if "report_date" in df.columns
                        and pd.notna(row["report_date"])
                        else None
                    ),
                    "report_type": (
                        str(row["report_type"])
                        if "report_type" in df.columns
                        and pd.notna(row["report_type"])
                        else None
                    ),
                    "severity": (
                        str(row["severity"])
                        if "severity" in df.columns
                        and pd.notna(row["severity"])
                        else None
                    ),
                    "site": (
                        str(row["site"])
                        if "site" in df.columns
                        and pd.notna(row["site"])
                        else None
                    ),
                    "activity": (
                        str(row["activity"])
                        if "activity" in df.columns
                        and pd.notna(row["activity"])
                        else None
                    ),
                    "facility": (
                        str(row["facility"])
                        if "facility" in df.columns
                        and pd.notna(row["facility"])
                        else None
                    ),
                },
            )
            inserted += 1

    return {
        "status": "success",
        "filename": file.filename,
        "rows_received": len(df),
        "rows_processed": inserted,
    }
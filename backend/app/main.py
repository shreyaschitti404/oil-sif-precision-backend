from fastapi import FastAPI

from backend.app.api.reports import router as reports_router
from backend.app.api.upload import router as upload_router
from backend.app.api.dashboard import router as dashboard_router


app = FastAPI(
    title="OIL SIF Precision API",
    version="1.0.0",
)

app.include_router(reports_router)
app.include_router(upload_router)
app.include_router(dashboard_router)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "OIL SIF Precision API",
    }
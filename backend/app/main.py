from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.reports import router as reports_router
from backend.app.api.upload import router as upload_router
from backend.app.api.dashboard import router as dashboard_router
from backend.app.api.analysis import router as analysis_router


app = FastAPI(
    title="OIL SIF Precision API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://oil-sif-precision-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reports_router)
app.include_router(upload_router)
app.include_router(dashboard_router)
app.include_router(analysis_router)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "OIL SIF Precision API",
    }

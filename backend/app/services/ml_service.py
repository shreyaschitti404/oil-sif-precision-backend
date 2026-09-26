import os
from pathlib import Path

from huggingface_hub import snapshot_download

from sih26165_sif_engine_full import SIFEngine, analyze_report


MODEL_REPO_ID = os.getenv(
    "HF_MODEL_ID",
    "salphayas/oil-sif-precision-model",
)

MODEL_DIR = Path(
    os.getenv("SIF_MODEL_DIR", "./sif_model")
)

_engine: SIFEngine | None = None


def _get_engine() -> SIFEngine:
    global _engine

    # Reuse the already-loaded engine.
    if _engine is not None:
        return _engine

    # Download the private Hugging Face model if it is not already
    # available locally.
    if not MODEL_DIR.exists() or not (
        MODEL_DIR / "model_metadata.json"
    ).exists():
        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN is not configured. "
                "A Hugging Face token is required to download "
                "the private model repository."
            )

        downloaded_path = snapshot_download(
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            token=token,
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        downloaded = Path(downloaded_path)

        for source in downloaded.iterdir():
            target = MODEL_DIR / source.name

            if source.is_file() and not target.exists():
                target.write_bytes(source.read_bytes())

    # Load the trained SIF engine.
    _engine = SIFEngine(str(MODEL_DIR))

    return _engine


def analyze_report(text: str):
    if not text or not text.strip():
        raise ValueError("Report text is required.")

    engine = _get_engine()

    # The friend's ML engine exposes analyze_report(engine, text).
    result = analyze_report(engine, text)

    sif_prediction = result.get("sif_prediction")

    return {
        "sif_potential": sif_prediction == "SIF_PRECURSOR",
        "confidence": float(result.get("sif_score", 0.0)),
        "life_saving_rule": result.get("life_saving_rule"),
        "activity": result.get("activity"),
        "hazard": None,
        "barrier_failure": result.get("barrier_failure"),
        "precursor_pattern": None,
        "model_version": MODEL_REPO_ID,
    }

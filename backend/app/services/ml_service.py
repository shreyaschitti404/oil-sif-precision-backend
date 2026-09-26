def analyze_report(text: str):
    if not text or not text.strip():
        raise ValueError("Report text is required.")

    return {
        "sif_potential": False,
        "confidence": 0.0,
        "life_saving_rule": None,
        "activity": None,
        "hazard": None,
        "barrier_failure": None,
        "precursor_pattern": None,
        "model_version": "not-connected",
    }

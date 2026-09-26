def analyze_report(text: str):
    if not text or not text.strip():
        raise ValueError("Report text is required.")

    return {
        "status": "ml_not_connected",
        "message": "ML model will be connected here.",
        "text": text,
    }
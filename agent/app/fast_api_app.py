"""FastAPI application for the cross-border data router agent.

Stripped of Google Cloud dependencies (Cloud Logging, GCS artifacts,
google.auth) so the agent runs fully self-hosted on RHOAI.
"""

import logging
import os

from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

from app.app_utils.typing import Feedback

allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",")
    if os.getenv("ALLOW_ORIGINS")
    else None
)

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    allow_origins=allow_origins,
    session_service_uri=None,
)
app.title = "cross-border-data-router"
app.description = (
    "Cross-border data-policy router powered by Laya (System 1) "
    "and Qwen 2.5 (System 2) on Red Hat OpenShift AI"
)


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logging.info(feedback.model_dump())
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104

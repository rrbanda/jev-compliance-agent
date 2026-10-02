"""Shared LLM model configuration for all agents.

Uses Gemini via the Google AI API as the System 2 reasoning LLM.
Laya handles the System 1 classification (on-cluster, zero output
tokens) and Gemini reasons about the decisions.

Configure via environment variables (see .env.example):
  GOOGLE_API_KEY   — Gemini API key (required)
  MODEL_NAME       — model name (default: gemini-2.5-flash)
"""

from __future__ import annotations

import os

from google.adk.models import Gemini


def get_model() -> Gemini:
    """Returns a Gemini model instance for the agent.

    The API key is read from the GOOGLE_API_KEY environment variable.
    This is the System 2 (deliberative) half of the pipeline — it
    reasons about routing decisions after Laya's System 1 classification
    has provided calibrated probabilities.
    """
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    return Gemini(model=model_name)

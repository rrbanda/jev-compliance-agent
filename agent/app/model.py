"""Shared LLM model configuration for all agents.

Uses Gemini via the Google AI API as the System 2 reasoning LLM.  This
frees all on-cluster GPUs for inference workloads (Laya, DiffusionGemma)
and gives the agent a stronger reasoning model than a local 7B.

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
    This is the System 2 (slow, deliberative) half of the pipeline —
    it reasons about the routing decision after the System 1 engines
    (Laya / DiffusionGemma) have provided their classifications.
    """
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    return Gemini(model=model_name)

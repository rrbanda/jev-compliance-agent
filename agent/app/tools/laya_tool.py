"""Dual-backend Jev System 1 decision tool.

Both Laya and DiffusionGemma speak the same Jev ``/v1/systemone`` wire
protocol, so the agent gets a single tool that can target either backend:

  - **fast** → Laya (421M encoder, CPU, ~145ms): purpose-built decision
    model that cannot hallucinate — outputs calibrated probability
    distributions directly from a bidirectional encoder.
  - **deep** → DiffusionGemma 26B-A4B (H200 MIG, ~10-26s): diffusion
    transformer with structured-read mode.  Denoises a fixed token canvas
    in parallel; vLLM's structured-read pins an answer template and reads
    the distribution at the answer slots.
  - **auto** → calls Laya first; if the top-choice confidence is below a
    threshold (default 0.80), escalates to DiffusionGemma for a deeper
    read.  Returns both results so the agent (and the audit log) can see
    the fast triage and the deep confirmation side by side.

The Jev request/response shapes are identical for both backends.  The
``questions`` schema uses three Jev types:
  - ``noul`` (yes/no with probability): "Is this PII?", "Needs human review?"
  - ``choice`` (categorical with distribution): "What data type?"
  - ``score`` (ordered scale): "Risk tier?"
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

_log = logging.getLogger(__name__)

LAYA_URL = os.getenv(
    "LAYA_API_URL",
    "https://laya-api-gpu-laya-demo.apps.ocp.qn6c5.sandbox1388.opentlc.com",
)
DIFFUSIONGEMMA_URL = os.getenv(
    "DIFFUSIONGEMMA_API_URL",
    "https://dgemma-jev-user-rbanda.apps.ocp.cloud.rhai-tmm.dev",
)
SYSTEM1_MODE = os.getenv("SYSTEM1_MODE", "auto")
CONFIDENCE_THRESHOLD = float(os.getenv("SYSTEM1_CONFIDENCE_THRESHOLD", "0.80"))

# ── Jev question schemas ────────────────────────────────────────────
# Shared between both backends — same wire format.

_LAYA_QUESTIONS: dict[str, Any] = {
    "data_type": {
        "type": "choice",
        "instructions": (
            "What type of sensitive data is present in this record?"
        ),
        "criteria": {
            "PII": {
                "description": (
                    "Personal identifiable information such as "
                    "names, addresses, dates of birth, or ID numbers"
                ),
            },
            "financial": {
                "description": (
                    "Financial records, transactions, account "
                    "numbers, or monetary data"
                ),
            },
            "health": {
                "description": (
                    "Medical or health records, diagnoses, "
                    "treatment plans, or insurance claims"
                ),
            },
            "public": {
                "description": (
                    "Non-sensitive public information with no "
                    "regulatory restrictions"
                ),
            },
        },
    },
    "has_pii": {
        "type": "noul",
        "instructions": (
            "Does this record contain personally identifiable information?"
        ),
    },
}

_DIFFUSIONGEMMA_QUESTIONS: dict[str, Any] = {
    "data_type": {
        "type": "choice",
        "instructions": (
            "What type of sensitive data is present in this record?"
        ),
        "options": [
            {"name": "PII", "description": "Personal identifiable information"},
            {"name": "financial", "description": "Financial records or monetary data"},
            {"name": "health", "description": "Medical or health records"},
            {"name": "public", "description": "Non-sensitive public information"},
        ],
    },
    "has_pii": {
        "type": "noul",
        "instructions": (
            "Does this record contain personally identifiable information "
            "such as names, addresses, or ID numbers?"
        ),
    },
    "needs_human_review": {
        "type": "noul",
        "instructions": (
            "Is this record ambiguous enough that a human compliance "
            "officer should review the classification before routing?"
        ),
    },
}


def _call_jev(
    url: str, text: str, questions: dict, timeout: float = 30.0
) -> dict[str, Any]:
    """Fire a Jev /v1/systemone request and return the raw response."""
    resp = httpx.post(
        f"{url}/v1/systemone",
        json={
            "model": "jev-latest",
            "state": text,
            "questions": questions,
        },
        timeout=timeout,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_laya_result(raw: dict) -> dict[str, Any]:
    """Normalise a Laya Jev response into the tool's return shape."""
    answers = raw.get("answers", {})
    data_type = answers.get("data_type", {})
    has_pii = answers.get("has_pii", {})

    classification = data_type.get("choice", "unknown")
    probabilities = data_type.get("probabilities", {})
    confidence = max(probabilities.values()) if probabilities else 0.0
    pii_prob = has_pii.get("noul", has_pii.get("choice", 0.0))
    if isinstance(pii_prob, str):
        pii_prob = 1.0 if pii_prob == "yes" else 0.0

    return {
        "data_classification": classification,
        "has_pii": pii_prob > 0.5 if isinstance(pii_prob, (int, float)) else pii_prob == "yes",
        "pii_probability": float(pii_prob) if isinstance(pii_prob, (int, float)) else None,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }


def _parse_diffusiongemma_result(raw: dict) -> dict[str, Any]:
    """Normalise a DiffusionGemma Jev response into the tool's return shape."""
    answers = raw.get("answers", {})
    data_type = answers.get("data_type", {})
    has_pii = answers.get("has_pii", {})
    needs_review = answers.get("needs_human_review", {})

    classification = data_type.get("choice", "unknown")
    probabilities = data_type.get("probabilities", {})
    confidence_val = data_type.get("confidence", 0.0)
    if not confidence_val and probabilities:
        confidence_val = max(probabilities.values())
    pii_noul = has_pii.get("noul", 0.0)
    review_noul = needs_review.get("noul", None)

    return {
        "data_classification": classification,
        "has_pii": pii_noul > 0.5,
        "pii_probability": round(pii_noul, 4),
        "confidence": round(confidence_val, 4),
        "probabilities": probabilities,
        "needs_human_review": round(review_noul, 4) if review_noul is not None else None,
    }


def classify_with_laya(text: str, backend: str = "auto") -> dict:
    """Classify a data record using the Jev System 1 decision engine(s).

    Both backends speak the same Jev ``/v1/systemone`` protocol.  The
    ``backend`` parameter selects which to call:

    - ``"fast"`` — Laya only (421M encoder, ~145ms on CPU).
    - ``"deep"`` — DiffusionGemma only (26B diffusion model, ~10-26s on
      H200 MIG).
    - ``"auto"`` (default) — calls Laya first; if the top-choice
      confidence is below 0.80, also calls DiffusionGemma and returns
      both results for the agent to compare.

    Args:
        text: The data record or description to classify.
        backend: Which System 1 engine to use: ``"fast"``, ``"deep"``,
            or ``"auto"``.

    Returns:
        A dict with ``data_classification``, ``has_pii``, ``confidence``,
        ``probabilities``, the ``backend`` used, ``latency_ms``, and
        (when auto-escalated) a ``deep_result`` with the DiffusionGemma
        confirmation.
    """
    mode = backend if backend in ("fast", "deep") else SYSTEM1_MODE
    result: dict[str, Any] = {}

    if mode in ("fast", "auto"):
        t0 = time.monotonic()
        try:
            raw = _call_jev(LAYA_URL, text, _LAYA_QUESTIONS, timeout=10.0)
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            result = _parse_laya_result(raw)
            result["backend"] = "laya"
            result["engine"] = "laya-system1"
            result["latency_ms"] = elapsed
            _log.info("Laya classified %r → %s (%.1f%% in %.0fms)",
                       text[:60], result["data_classification"],
                       result["confidence"] * 100, elapsed)
        except Exception as exc:
            _log.warning("Laya call failed (%s), falling back to deep", exc)
            mode = "deep"

    if mode == "deep" or (
        mode == "auto"
        and result.get("confidence", 0) < CONFIDENCE_THRESHOLD
    ):
        escalation_reason = None
        if mode == "auto" and result:
            escalation_reason = (
                f"Laya confidence {result['confidence']:.2f} < "
                f"threshold {CONFIDENCE_THRESHOLD:.2f}"
            )
            _log.info("Auto-escalating to DiffusionGemma: %s", escalation_reason)

        t0 = time.monotonic()
        try:
            raw = _call_jev(
                DIFFUSIONGEMMA_URL, text, _DIFFUSIONGEMMA_QUESTIONS,
                timeout=120.0,
            )
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            deep = _parse_diffusiongemma_result(raw)
            deep["backend"] = "diffusiongemma"
            deep["engine"] = "diffusiongemma-26b-h200"
            deep["latency_ms"] = elapsed
            _log.info("DiffusionGemma classified %r → %s (%.1f%% in %.0fms)",
                       text[:60], deep["data_classification"],
                       deep["confidence"] * 100, elapsed)

            if result:
                result["deep_result"] = deep
                result["escalation_reason"] = escalation_reason
            else:
                result = deep
        except Exception as exc:
            _log.warning("DiffusionGemma call failed: %s", exc)
            if not result:
                return {
                    "error": f"Both backends failed. DiffusionGemma: {exc}",
                    "backend": "none",
                }

    return result

"""Jev System 1 compliance classification with confidence gating.

Two Jev decision models, same ``/v1/systemone`` protocol, same questions:

  Laya (421M encoder, ~150ms)  — fast triage, runs on every request.
  DiffusionGemma (26B, ~10-26s) — deep confirmation, called only when
                                   Laya's confidence is below threshold.

Both return calibrated probability distributions, not generated text.
Zero output tokens from either model.  The confidence gate follows the
pattern from ``examples/18_confidence_gating.py`` in the Laya repo:
gate on ``answer_confidence``, not ``confidence`` — the former is the
calibrated probability on the reported answer; the latter is normalised
entropy and means different things on different question types.

Question design follows Laya's own preset patterns (``laya.presets``):
  - Names the state field in backticks so the encoder reads the right key.
  - Mixes ``choice``, ``noul`` and ``score`` in one pass.
  - Criteria worded to be mutually exclusive and to share vocabulary with
    the data records the tool will see.
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

CONFIDENCE_GATE = float(os.getenv("CONFIDENCE_GATE", "0.55"))

# ── Jev question schema ─────────────────────────────────────────────
# Five questions, mixed types, answered in one forward pass.
# The backtick convention (`record`) tells the encoder which state key
# to read — same pattern as laya.triage_questions() uses `message`.

COMPLIANCE_QUESTIONS = {
    "data_type": {
        "type": "choice",
        "instructions": "What type of sensitive data is present in `record`?",
        "criteria": {
            "PII": "personal identifiable information: names, addresses, "
                   "dates of birth, national ID numbers, tax IDs",
            "financial": "financial records: account numbers, transactions, "
                         "invoices, monetary amounts, payment data",
            "health": "medical or health records: diagnoses, treatment plans, "
                      "prescriptions, insurance claims, patient IDs",
            "public": "non-sensitive public information with no regulatory "
                      "restrictions on storage or transfer",
        },
    },
    "pii_detected": {
        "type": "noul",
        "instructions": "Does `record` contain personally identifiable "
                        "information such as names, addresses, national IDs, "
                        "or dates of birth?",
        "criteria": {
            "true": "contains one or more PII elements that identify a "
                    "natural person",
            "false": "no information that identifies a natural person",
        },
    },
    "data_subject_location": {
        "type": "choice",
        "instructions": "Where is the data subject in `record` located?",
        "criteria": {
            "EU": "European Union or EEA country — Germany, France, "
                  "Netherlands, Italy, Spain, Austria, Belgium, etc.",
            "UK": "United Kingdom — England, Scotland, Wales, "
                  "Northern Ireland, London, GBP, NHS",
            "US": "United States — any US state or territory, "
                  "USD, SSN, ZIP code",
            "other": "none of the above, or location is not stated",
        },
    },
    "sensitivity": {
        "type": "score",
        "instructions": "How sensitive is the data in `record`?",
        "criteria": [
            "public: no restrictions on storage or transfer",
            "internal: not for public release but no regulatory burden",
            "confidential: subject to regulatory controls, breach notification required",
            "restricted: highest sensitivity, legal liability if mishandled",
        ],
    },
    "needs_human_review": {
        "type": "noul",
        "instructions": "Is `record` ambiguous enough that a human "
                        "compliance officer should review the classification "
                        "before it is routed?",
        "criteria": {
            "true": "the record mixes data types, names multiple "
                    "jurisdictions, or contains conflicting signals",
            "false": "the classification is straightforward",
        },
    },
}


def _call_jev(url: str, record_text: str, timeout: float = 30.0) -> dict:
    """Call a Jev /v1/systemone endpoint and return the raw response."""
    resp = httpx.post(
        f"{url}/v1/systemone",
        json={
            "state": {"record": record_text},
            "questions": COMPLIANCE_QUESTIONS,
        },
        timeout=timeout,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_answers(raw: dict, backend: str, elapsed: float) -> dict:
    """Parse a Jev response into our standard result dict."""
    answers = raw.get("answers", {})

    data_type = answers.get("data_type", {})
    pii = answers.get("pii_detected", {})
    location = answers.get("data_subject_location", {})
    sensitivity = answers.get("sensitivity", {})
    review = answers.get("needs_human_review", {})

    data_type_choice = data_type.get("choice", "unknown")
    data_type_probs = data_type.get("probabilities", {})
    # Laya returns answer_confidence; DiffusionGemma returns confidence
    data_type_conf = data_type.get("answer_confidence",
                                   data_type.get("confidence",
                                   max(data_type_probs.values())
                                   if data_type_probs else 0.0))

    location_choice = location.get("choice", "other")
    location_probs = location.get("probabilities", {})
    location_conf = location.get("answer_confidence",
                                 location.get("confidence",
                                 max(location_probs.values())
                                 if location_probs else 0.0))

    return {
        "data_type": data_type_choice,
        "data_type_confidence": round(data_type_conf, 4),
        "data_type_probabilities": data_type_probs,
        "pii_detected": round(pii.get("noul", 0.0), 4),
        "data_subject_location": location_choice,
        "data_subject_location_confidence": round(location_conf, 4),
        "data_subject_location_probabilities": location_probs,
        "sensitivity_score": round(sensitivity.get("score", 0.0), 2),
        "sensitivity_legend": sensitivity.get("legend", []),
        "needs_human_review": round(review.get("noul", 0.0), 4),
        "backend": backend,
        "latency_ms": elapsed,
        "questions_answered": 5,
        "output_tokens": 0,
        "usage": raw.get("usage", {}),
    }


def classify_with_laya(record_text: str) -> dict:
    """Classify a data record for cross-border compliance routing.

    Calls Laya first (~150ms, 421M encoder).  If Laya's
    ``data_type_confidence`` or ``data_subject_location_confidence``
    falls below the confidence gate (default 0.55), DiffusionGemma 26B
    is called automatically with the **same five questions** for a
    deeper read.  Both models speak the same Jev ``/v1/systemone``
    protocol and return the same answer shapes.

    The confidence gate follows ``examples/18_confidence_gating.py``:
    gate on ``answer_confidence`` — the calibrated probability on the
    reported answer — not on ``confidence`` (normalised entropy).

    When DiffusionGemma is called, the result includes both sets of
    answers so the full evidence chain is visible.

    Args:
        record_text: The data record or description to classify.

    Returns:
        A dict with all five typed answers, their probabilities, the
        backend used (``laya`` or ``laya+diffusiongemma``), latency,
        and whether escalation occurred.
    """
    # ── Step 1: Laya (fast) ──────────────────────────────────────
    t0 = time.monotonic()
    try:
        raw = _call_jev(LAYA_URL, record_text, timeout=10.0)
    except Exception as exc:
        _log.error("Laya /v1/systemone call failed: %s", exc)
        return {
            "error": str(exc),
            "backend": "laya",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    laya_elapsed = round((time.monotonic() - t0) * 1000, 1)
    laya_result = _parse_answers(raw, "laya", laya_elapsed)

    _log.info(
        "Laya: data_type=%s (conf=%.2f), location=%s (conf=%.2f) in %.0fms",
        laya_result["data_type"], laya_result["data_type_confidence"],
        laya_result["data_subject_location"],
        laya_result["data_subject_location_confidence"],
        laya_elapsed,
    )

    # ── Step 2: confidence gate ──────────────────────────────────
    needs_escalation = (
        laya_result["data_type_confidence"] < CONFIDENCE_GATE
        or laya_result["data_subject_location_confidence"] < CONFIDENCE_GATE
    )

    if not needs_escalation:
        laya_result["escalated_to_diffusiongemma"] = False
        return laya_result

    # ── Step 3: DiffusionGemma (deep) — same questions ───────────
    _log.info(
        "Confidence below %.2f — escalating to DiffusionGemma",
        CONFIDENCE_GATE,
    )
    t1 = time.monotonic()
    try:
        deep_raw = _call_jev(DIFFUSIONGEMMA_URL, record_text, timeout=60.0)
    except Exception as exc:
        _log.warning("DiffusionGemma call failed, using Laya result: %s", exc)
        laya_result["escalated_to_diffusiongemma"] = False
        laya_result["escalation_error"] = str(exc)
        return laya_result

    deep_elapsed = round((time.monotonic() - t1) * 1000, 1)
    deep_result = _parse_answers(deep_raw, "diffusiongemma", deep_elapsed)

    _log.info(
        "DiffusionGemma: data_type=%s (conf=%.2f), location=%s (conf=%.2f) in %.0fms",
        deep_result["data_type"], deep_result["data_type_confidence"],
        deep_result["data_subject_location"],
        deep_result["data_subject_location_confidence"],
        deep_elapsed,
    )

    # ── Merge: use DiffusionGemma's answers as the primary result,
    #    attach Laya's as evidence ─────────────────────────────────
    total_elapsed = round(laya_elapsed + deep_elapsed, 1)
    deep_result["backend"] = "laya+diffusiongemma"
    deep_result["latency_ms"] = total_elapsed
    deep_result["escalated_to_diffusiongemma"] = True
    deep_result["escalation_reason"] = (
        "data_type_confidence=%.2f or location_confidence=%.2f below gate=%.2f"
        % (laya_result["data_type_confidence"],
           laya_result["data_subject_location_confidence"],
           CONFIDENCE_GATE)
    )
    deep_result["laya_fast_result"] = {
        "data_type": laya_result["data_type"],
        "data_type_confidence": laya_result["data_type_confidence"],
        "pii_detected": laya_result["pii_detected"],
        "data_subject_location": laya_result["data_subject_location"],
        "data_subject_location_confidence": laya_result["data_subject_location_confidence"],
        "sensitivity_score": laya_result["sensitivity_score"],
        "needs_human_review": laya_result["needs_human_review"],
        "latency_ms": laya_result["latency_ms"],
    }

    return deep_result

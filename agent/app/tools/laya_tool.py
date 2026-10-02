"""Jev System 1 compliance classification tool.

One call, five typed questions, one forward pass, zero output tokens.
Returns calibrated probabilities that feed directly into the application
policy (``compliance_policy.py``) — the model provides the signal, the
policy provides the decision.

Question design follows Laya's own preset patterns (``laya.presets``):
  - Names the state field in backticks so the encoder reads the right key.
  - Mixes ``choice``, ``noul`` and ``score`` in one pass — different heads
    on the same encoder forward pass, no extra cost per question.
  - Criteria worded to be mutually exclusive and to share vocabulary with
    the data records the tool will see.

See ``examples/30_custom_schema_design.py`` in the Laya repo for the
design rationale behind these choices.
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


def classify_with_laya(record_text: str) -> dict:
    """Classify a data record for cross-border compliance routing.

    Makes one Jev ``/v1/systemone`` call with five typed questions — all
    answered in a single forward pass (~145 ms on CPU, ~28 ms on GPU).
    Zero output tokens: Laya is non-autoregressive and returns calibrated
    probability distributions, not generated text.

    The five answers are:

    - ``data_type`` (choice): PII / financial / health / public — with a
      probability distribution over all four.
    - ``pii_detected`` (noul): probability 0.0–1.0 that the record
      contains personally identifiable information.
    - ``data_subject_location`` (choice): EU / UK / US / other — where
      the data subject is located, with probability distribution.
    - ``sensitivity`` (score): ordered scale from public (0) to
      restricted (3), with a probability distribution.
    - ``needs_human_review`` (noul): probability that a human compliance
      officer should review before routing.

    These probabilities feed the application policy, not the other way
    around — the model provides the signal, you set the thresholds.

    Args:
        record_text: The data record or description to classify.

    Returns:
        A dict with all five typed answers, their probabilities, the
        backend used, and latency in milliseconds.
    """
    t0 = time.monotonic()

    try:
        resp = httpx.post(
            f"{LAYA_URL}/v1/systemone",
            json={
                "state": {"record": record_text},
                "questions": COMPLIANCE_QUESTIONS,
            },
            timeout=10.0,
            verify=False,
        )
        resp.raise_for_status()
    except Exception as exc:
        _log.error("Laya /v1/systemone call failed: %s", exc)
        return {
            "error": str(exc),
            "backend": "laya",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    raw = resp.json()
    answers = raw.get("answers", {})

    # ── Parse each typed answer ──────────────────────────────────
    data_type = answers.get("data_type", {})
    pii = answers.get("pii_detected", {})
    regime = answers.get("data_subject_location", {})
    sensitivity = answers.get("sensitivity", {})
    review = answers.get("needs_human_review", {})

    data_type_choice = data_type.get("choice", "unknown")
    data_type_probs = data_type.get("probabilities", {})
    data_type_conf = data_type.get("answer_confidence",
                                   max(data_type_probs.values())
                                   if data_type_probs else 0.0)

    location_choice = regime.get("choice", "other")
    location_probs = regime.get("probabilities", {})

    sensitivity_score = sensitivity.get("score", 0.0)
    sensitivity_legend = sensitivity.get("legend", [])

    pii_prob = pii.get("noul", 0.0)
    review_prob = review.get("noul", 0.0)

    _log.info(
        "Laya: data_type=%s (conf=%.2f), pii=%.2f, location=%s, "
        "sensitivity=%.1f, review=%.2f in %.0fms",
        data_type_choice, data_type_conf, pii_prob, location_choice,
        sensitivity_score, review_prob, elapsed,
    )

    return {
        # ── The five typed answers ───────────────────────────────
        "data_type": data_type_choice,
        "data_type_confidence": round(data_type_conf, 4),
        "data_type_probabilities": data_type_probs,
        "pii_detected": round(pii_prob, 4),
        "data_subject_location": location_choice,
        "data_subject_location_probabilities": location_probs,
        "sensitivity_score": round(sensitivity_score, 2),
        "sensitivity_legend": sensitivity_legend,
        "needs_human_review": round(review_prob, 4),
        # ── Metadata ─────────────────────────────────────────────
        "backend": "laya",
        "latency_ms": elapsed,
        "questions_answered": 5,
        "output_tokens": 0,
        "usage": raw.get("usage", {}),
    }

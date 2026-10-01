"""Laya System 1 classification tool for the ADK agent.

Calls the Laya GPU InferenceService on RHOAI to get an instant (~28ms)
neural classification of data records before the policy engine runs.
This is the "System 1" (fast, intuitive) half of the pipeline; the LLM
orchestrator is "System 2" (slow, deliberative).
"""

from __future__ import annotations

import os

import httpx

LAYA_URL = os.getenv(
    "LAYA_API_URL",
    "https://laya-api-gpu-laya-demo.apps.ocp.qn6c5.sandbox1388.opentlc.com",
)


def classify_with_laya(text: str) -> dict:
    """Classify a data record using the Laya System 1 decision engine.

    Calls the Laya GPU endpoint running on RHOAI to instantly classify
    text as PII, financial, health, or public data. Use this BEFORE
    calling ``evaluate_and_route`` to get a reliable, model-based
    classification rather than guessing from the user's description.
    The caller may still override the classification if they explicitly
    state a different one.

    Args:
        text: The data record or description to classify.

    Returns:
        A dict with ``data_classification`` (the winning label),
        ``has_pii`` (bool), ``confidence`` (per-label probabilities
        from Laya), and metadata about the engine used.
    """
    resp = httpx.post(
        f"{LAYA_URL}/v1/systemone",
        json={
            "state": text,
            "questions": {
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
                    "type": "choice",
                    "instructions": (
                        "Does this record contain personally identifiable "
                        "information?"
                    ),
                    "criteria": {
                        "yes": {
                            "description": (
                                "Contains PII such as names, addresses, "
                                "social security numbers, or dates of birth"
                            ),
                        },
                        "no": {
                            "description": "No PII detected in this record",
                        },
                    },
                },
            },
        },
        timeout=10.0,
        verify=False,
    )
    resp.raise_for_status()
    result = resp.json()

    # Laya API returns answers under "answers" with "choice" and "probabilities"
    data_type_answer = result["answers"]["data_type"]
    has_pii_answer = result["answers"]["has_pii"]

    return {
        "data_classification": data_type_answer["choice"],
        "has_pii": has_pii_answer["choice"] == "yes",
        "confidence": data_type_answer.get("probabilities", {}),
        "engine": "laya-system1-gpu",
        "latency_note": "sub-30ms on-device neural classification",
    }

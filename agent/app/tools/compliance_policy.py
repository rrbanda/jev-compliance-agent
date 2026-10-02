"""Application policy: maps Jev probabilities to routing decisions.

This is the critical separation that makes Jev useful in regulated settings.
The model returns calibrated probabilities.  This module provides the
**thresholds, rules and logic** that turn those probabilities into
actionable routing decisions.

    "routing, answers and every probability are model output.
     action, team, human_review and the thresholds are application policy --
     swap the policy without retraining."
       — Laya example 41

Design choice: Laya answers "what is in the data?" and "where is the data
subject?".  This module answers "which law applies?" and "where may it be
stored?".  That split is deliberate — regulatory applicability is *law*,
not text classification.  When a regulation changes or a new one is added,
update the policy table below, not the model or its question schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ── Tuneable thresholds ──────────────────────────────────────────────
# Each threshold is a probability boundary.  Raise it to be more
# conservative (more human review); lower it to automate more.

PII_THRESHOLD = 0.70           # P(pii_detected) above this → treat as PII
REVIEW_THRESHOLD = 0.60        # P(needs_human_review) above this → flag for human
SENSITIVITY_RESTRICTED = 2.5   # sensitivity score above this → escalate
CONFIDENCE_FLOOR = 0.50        # data_type answer_confidence below this → escalate
LOCATION_CONFIDENCE_FLOOR = 0.45  # location confidence below this → escalate


# ── Regime inference table ───────────────────────────────────────────
# (data_subject_location, data_type) → (regime, residency, excluded)
# The model tells us WHERE and WHAT.  The law tells us the rest.

_REGIME_TABLE = {
    # EU subjects — GDPR
    ("EU", "PII"):       ("GDPR", ["EU", "EEA"], ["US", "CHINA", "RUSSIA"]),
    ("EU", "financial"): ("GDPR", ["EU", "EEA"], ["CHINA", "RUSSIA"]),
    ("EU", "health"):    ("GDPR", ["EU", "EEA"], ["US", "CHINA", "RUSSIA"]),
    ("EU", "public"):    ("none", [], []),
    # UK subjects — UK GDPR
    ("UK", "PII"):       ("UK_GDPR", ["UK", "EU", "EEA"], ["CHINA", "RUSSIA"]),
    ("UK", "financial"): ("UK_GDPR", ["UK", "EU", "EEA"], ["CHINA", "RUSSIA"]),
    ("UK", "health"):    ("UK_GDPR", ["UK", "EU", "EEA"], ["CHINA", "RUSSIA"]),
    ("UK", "public"):    ("none", [], []),
    # US subjects
    ("US", "PII"):       ("CCPA", ["US"], []),
    ("US", "financial"): ("SOX", ["US"], []),
    ("US", "health"):    ("HIPAA", ["US"], ["CHINA", "RUSSIA"]),
    ("US", "public"):    ("none", [], []),
}

_DEFAULT_REGIME = ("none", [], [])


@dataclass
class ComplianceDecision:
    """The output of the application policy.

    Every field is deterministic given the Jev probabilities and the
    thresholds above — no randomness, no model inference, fully auditable.
    """

    data_classification: str
    regulatory_regime: str
    data_subject_location: str
    required_residency: list[str]
    excluded_jurisdictions: list[str]
    human_review_required: bool
    human_review_reasons: list[str] = field(default_factory=list)
    policy_notes: list[str] = field(default_factory=list)


def apply_compliance_policy(
    data_type: str,
    data_type_confidence: float,
    pii_detected: float,
    data_subject_location: str,
    data_subject_location_confidence: float,
    sensitivity_score: float,
    needs_human_review: float,
) -> dict:
    """Apply deterministic compliance rules to Laya's probability outputs.

    The model answers "what is in the data?" and "where is the data
    subject?".  This function answers "which law applies?" and "where may
    the data be stored?".

    Pass the key fields from the ``classify_with_laya`` result:

    Args:
        data_type: The winning choice for data type (e.g. "PII",
            "financial", "health", "public").
        data_type_confidence: Confidence of the data_type answer (0.0–1.0).
        pii_detected: Probability that PII is present (0.0–1.0).
        data_subject_location: Where the data subject is ("EU", "UK",
            "US", or "other").
        data_subject_location_confidence: Confidence of the location
            answer (0.0–1.0).
        sensitivity_score: Sensitivity level on a 0–3 scale.
        needs_human_review: Probability that a human should review (0.0–1.0).

    Returns:
        A dict with the compliance decision: data_classification,
        regulatory_regime, data_subject_location, required_residency,
        excluded_jurisdictions, human_review_required, reasons and notes.
    """
    human_review = False
    review_reasons: list[str] = []
    notes: list[str] = []

    # ── Rule 1: PII override ────────────────────────────────────
    # Record the original data_type before any override — the regime
    # table needs the original to distinguish health (→ HIPAA) from
    # generic PII (→ CCPA).
    original_data_type = data_type
    if pii_detected >= PII_THRESHOLD and data_type != "PII":
        notes.append(
            "PII override: pii_detected=%.2f >= %.2f, "
            "upgrading data_type from %s to PII" % (pii_detected, PII_THRESHOLD, data_type)
        )
        data_type = "PII"

    # ── Rule 2: regime inference from table ──────────────────────
    # Look up using original data_type first (health → HIPAA even when
    # PII override is active), then fall back to the overridden type.
    regime, residency, excluded = _REGIME_TABLE.get(
        (data_subject_location, original_data_type),
        _REGIME_TABLE.get((data_subject_location, data_type), _DEFAULT_REGIME),
    )
    if regime != "none":
        notes.append(
            "%s: location=%s + data_type=%s → residency=%s, excluded=%s"
            % (regime, data_subject_location, data_type, residency, excluded)
        )
    else:
        notes.append(
            "No regulated regime: location=%s + data_type=%s"
            % (data_subject_location, data_type)
        )

    # ── Rule 3: human review triggers ────────────────────────────
    if needs_human_review >= REVIEW_THRESHOLD:
        human_review = True
        review_reasons.append(
            "needs_human_review=%.2f >= %.2f" % (needs_human_review, REVIEW_THRESHOLD)
        )

    if data_type_confidence < CONFIDENCE_FLOOR:
        human_review = True
        review_reasons.append(
            "data_type confidence=%.2f < %.2f floor"
            % (data_type_confidence, CONFIDENCE_FLOOR)
        )

    if data_subject_location_confidence < LOCATION_CONFIDENCE_FLOOR:
        human_review = True
        review_reasons.append(
            "data_subject_location confidence=%.2f < %.2f floor"
            % (data_subject_location_confidence, LOCATION_CONFIDENCE_FLOOR)
        )

    if sensitivity_score >= SENSITIVITY_RESTRICTED:
        human_review = True
        review_reasons.append(
            "sensitivity=%.1f >= %.1f (restricted)"
            % (sensitivity_score, SENSITIVITY_RESTRICTED)
        )

    # ── Rule 4: conflicting signals ──────────────────────────────
    if data_type != "PII" and pii_detected >= 0.40:
        notes.append(
            "Mixed signal: data_type=%s but pii_detected=%.2f — "
            "record may contain PII alongside %s data"
            % (data_type, pii_detected, data_type)
        )
        if not human_review:
            human_review = True
            review_reasons.append(
                "Conflicting PII signal: data_type=%s but pii_detected=%.2f"
                % (data_type, pii_detected)
            )

    decision = ComplianceDecision(
        data_classification=data_type,
        regulatory_regime=regime,
        data_subject_location=data_subject_location,
        required_residency=residency,
        excluded_jurisdictions=excluded,
        human_review_required=human_review,
        human_review_reasons=review_reasons,
        policy_notes=notes,
    )
    return asdict(decision)

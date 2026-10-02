"""Unit tests for the application policy layer.

These tests verify that deterministic policy rules map Laya's calibrated
probabilities to the correct routing decisions.  No model inference, no
network calls — just thresholds and logic.
"""

from app.tools.compliance_policy import (
    apply_compliance_policy,
    PII_THRESHOLD,
    REVIEW_THRESHOLD,
    SENSITIVITY_RESTRICTED,
    CONFIDENCE_FLOOR,
)


def _call(**overrides) -> dict:
    """Call apply_compliance_policy with baseline defaults + overrides."""
    defaults = {
        "data_type": "financial",
        "data_type_confidence": 0.92,
        "pii_detected": 0.10,
        "data_subject_location": "EU",
        "data_subject_location_confidence": 0.80,
        "sensitivity_score": 2.0,
        "needs_human_review": 0.15,
    }
    defaults.update(overrides)
    return apply_compliance_policy(**defaults)


def test_eu_financial_gets_gdpr() -> None:
    d = _call(data_subject_location="EU", data_type="financial")
    assert d["regulatory_regime"] == "GDPR"
    assert d["required_residency"] == ["EU", "EEA"]
    assert "CHINA" in d["excluded_jurisdictions"]


def test_eu_pii_gets_gdpr_with_us_excluded() -> None:
    d = _call(data_subject_location="EU", data_type="PII", data_type_confidence=0.90)
    assert d["regulatory_regime"] == "GDPR"
    assert "US" in d["excluded_jurisdictions"]


def test_us_health_gets_hipaa() -> None:
    d = _call(data_subject_location="US", data_type="health", data_type_confidence=0.85)
    assert d["regulatory_regime"] == "HIPAA"
    assert d["required_residency"] == ["US"]
    assert "CHINA" in d["excluded_jurisdictions"]


def test_us_pii_gets_ccpa() -> None:
    d = _call(data_subject_location="US", data_type="PII", data_type_confidence=0.88)
    assert d["regulatory_regime"] == "CCPA"
    assert d["required_residency"] == ["US"]


def test_uk_pii_gets_uk_gdpr() -> None:
    d = _call(data_subject_location="UK", data_type="PII", data_type_confidence=0.85)
    assert d["regulatory_regime"] == "UK_GDPR"
    assert "UK" in d["required_residency"]
    assert "EU" in d["required_residency"]


def test_public_data_no_regime() -> None:
    d = _call(data_subject_location="EU", data_type="public", data_type_confidence=0.95)
    assert d["regulatory_regime"] == "none"
    assert d["required_residency"] == []


def test_other_location_no_regime() -> None:
    d = _call(data_subject_location="other", data_type="financial")
    assert d["regulatory_regime"] == "none"


def test_pii_override_when_pii_detected_high() -> None:
    d = _call(data_type="financial", pii_detected=0.85, data_subject_location="EU")
    assert d["data_classification"] == "PII"
    assert any("PII override" in n for n in d["policy_notes"])
    assert d["regulatory_regime"] == "GDPR"


def test_no_pii_override_when_below_threshold() -> None:
    d = _call(data_type="financial", pii_detected=0.30)
    assert d["data_classification"] == "financial"


def test_human_review_when_needs_review_high() -> None:
    d = _call(needs_human_review=0.75)
    assert d["human_review_required"] is True
    assert any("needs_human_review" in r for r in d["human_review_reasons"])


def test_human_review_when_low_data_type_confidence() -> None:
    d = _call(data_type_confidence=0.35)
    assert d["human_review_required"] is True
    assert any("confidence" in r for r in d["human_review_reasons"])


def test_human_review_when_low_location_confidence() -> None:
    d = _call(data_subject_location_confidence=0.30)
    assert d["human_review_required"] is True
    assert any("location" in r.lower() for r in d["human_review_reasons"])


def test_human_review_when_restricted_sensitivity() -> None:
    d = _call(sensitivity_score=2.8)
    assert d["human_review_required"] is True
    assert any("restricted" in r for r in d["human_review_reasons"])


def test_no_human_review_when_everything_clear() -> None:
    d = _call(data_type_confidence=0.92, needs_human_review=0.10, sensitivity_score=1.5)
    assert d["human_review_required"] is False


def test_conflicting_pii_signal_triggers_review() -> None:
    d = _call(data_type="health", pii_detected=0.55)
    assert d["human_review_required"] is True
    assert any("Conflicting" in r for r in d["human_review_reasons"])


def test_us_health_with_pii_override_stays_hipaa() -> None:
    """Health data with PII override should stay HIPAA, not become CCPA."""
    d = _call(data_subject_location="US", data_type="health",
              data_type_confidence=0.84, pii_detected=0.78)
    assert d["data_classification"] == "PII"
    assert d["regulatory_regime"] == "HIPAA"

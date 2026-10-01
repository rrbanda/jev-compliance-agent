"""End-to-end integration tests against live RHOAI endpoints.

These tests hit the Laya GPU InferenceService and the Qwen 2.5 Coder 7B
LLMInferenceService running on the cluster.  They are excluded from the
default pytest run (see pyproject.toml addopts) and must be invoked
explicitly::

    cd agent && uv run pytest tests/integration/test_rhoai_e2e.py -v

Skip automatically when the endpoints are unreachable.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

LAYA_URL = os.getenv(
    "LAYA_API_URL",
    "https://laya-api-gpu-laya-demo.apps.ocp.qn6c5.sandbox1388.opentlc.com",
)
QWEN_URL = os.getenv(
    "LLM_API_BASE",
    "https://maas.apps.ocp.qn6c5.sandbox1388.opentlc.com"
    "/private-assistant-ai-serving/qwen25-coder-7b/v1",
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")


def _laya_reachable() -> bool:
    try:
        r = httpx.get(f"{LAYA_URL}/health", timeout=5, verify=False)
        return r.status_code == 200
    except Exception:
        return False


def _qwen_reachable() -> bool:
    try:
        r = httpx.get(f"{QWEN_URL}/models", timeout=5, verify=False)
        return r.status_code == 200
    except Exception:
        return False


skip_laya = pytest.mark.skipif(
    not _laya_reachable(), reason="Laya GPU endpoint unreachable"
)
skip_qwen = pytest.mark.skipif(
    not _qwen_reachable(), reason="Qwen vLLM endpoint unreachable"
)


# ── Laya endpoint tests ────────────────────────────────────────────


@skip_laya
class TestLayaEndpoint:
    def test_health(self) -> None:
        r = httpx.get(f"{LAYA_URL}/health", timeout=5, verify=False)
        assert r.status_code == 200

    def test_classify_pii(self) -> None:
        r = httpx.post(
            f"{LAYA_URL}/v1/systemone",
            json={
                "state": (
                    "Customer Hans Mueller, Friedrichstrasse 42, Berlin. "
                    "DOB: 1985-03-15. Account DE89370400440532013000."
                ),
                "questions": {
                    "data_type": {
                        "type": "choice",
                        "instructions": "What type of sensitive data is present?",
                        "criteria": {
                            "PII": {"description": "Personal identifiable information"},
                            "financial": {"description": "Financial records"},
                            "public": {"description": "Non-sensitive public data"},
                        },
                    },
                },
            },
            timeout=10,
            verify=False,
        )
        assert r.status_code == 200
        result = r.json()
        assert result["answers"]["data_type"]["choice"] == "PII"

    def test_classify_financial(self) -> None:
        r = httpx.post(
            f"{LAYA_URL}/v1/systemone",
            json={
                "state": (
                    "Wire transfer USD 45,000 from account 4532-0198 "
                    "to beneficiary 8891-2233, Wells Fargo, ref TXN-2026-4421."
                ),
                "questions": {
                    "data_type": {
                        "type": "choice",
                        "instructions": "What type of sensitive data is present?",
                        "criteria": {
                            "PII": {"description": "Personal identifiable information"},
                            "financial": {"description": "Financial records"},
                            "public": {"description": "Non-sensitive public data"},
                        },
                    },
                },
            },
            timeout=10,
            verify=False,
        )
        assert r.status_code == 200
        result = r.json()
        assert result["answers"]["data_type"]["choice"] == "financial"


# ── Qwen tool-calling tests ────────────────────────────────────────


@skip_qwen
class TestQwenToolCalling:
    def test_models_endpoint(self) -> None:
        r = httpx.get(f"{QWEN_URL}/models", timeout=5, verify=False)
        assert r.status_code == 200
        data = r.json()
        assert any(
            m["id"] == MODEL_NAME for m in data.get("data", [])
        ), f"Model {MODEL_NAME} not found in {data}"

    def test_tool_call_routing(self) -> None:
        """Qwen must produce a structured tool_calls response."""
        r = httpx.post(
            f"{QWEN_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data-routing assistant. Use tools.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Route this PII data from Germany under GDPR."
                        ),
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "evaluate_and_route",
                            "description": "Evaluate cross-border residency policy",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "data_classification": {"type": "string"},
                                    "origin_region": {"type": "string"},
                                    "required_residency": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "data_classification",
                                    "origin_region",
                                    "required_residency",
                                ],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "max_tokens": 256,
            },
            timeout=30,
            verify=False,
        )
        assert r.status_code == 200
        result = r.json()
        msg = result["choices"][0]["message"]
        assert msg.get("tool_calls"), "Expected tool_calls in response"
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "evaluate_and_route"
        args = json.loads(tc["function"]["arguments"])
        assert "data_classification" in args
        assert "origin_region" in args


# ── Laya tool wrapper test ──────────────────────────────────────────


@skip_laya
class TestLayaTool:
    def test_classify_with_laya_function(self) -> None:
        """The ADK FunctionTool wrapper must return the expected dict shape."""
        from app.tools.laya_tool import classify_with_laya

        result = classify_with_laya(
            "Customer Hans Mueller, Berlin. DOB: 1985-03-15."
        )
        assert result["data_classification"] in {"PII", "financial", "health", "public"}
        assert isinstance(result["has_pii"], bool)
        assert result["engine"] == "laya-system1-gpu"
        assert "confidence" in result

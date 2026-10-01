"""Shared LLM model configuration for all agents.

Uses ``rh-maas-litellm`` — a Google ADK LiteLlm adapter for Red Hat MaaS
endpoints — to connect to the Qwen 2.5 Coder 7B vLLM instance on RHOAI.
The adapter overrides ``LlmCapabilities(output_schema_and_tools=False)``
so ADK never sends ``tools`` and ``response_format`` in the same request
(which vLLM / MaaS rejects with HTTP 400).

Configure via environment variables (see .env.example):
  MAAS_BASE_URL, MAAS_API_KEY, MAAS_URL_PATH, MODEL_NAME
"""

from __future__ import annotations

import os


def get_model():
    """Returns a MaaSLiteLlm instance for the on-cluster vLLM endpoint.

    Import is deferred so that unit tests for the policy engine can run
    without ``litellm`` / ``rh-maas-litellm`` installed.
    """
    from rh_maas_litellm import MaaSConfig, MaaSLiteLlm, bootstrap

    bootstrap(ssl_verify=False)

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
    cfg = MaaSConfig(
        base_url=os.getenv(
            "MAAS_BASE_URL",
            "https://maas.apps.ocp.qn6c5.sandbox1388.opentlc.com",
        ),
        api_key=os.getenv("MAAS_API_KEY", "unused"),
        url_path=os.getenv(
            "MAAS_URL_PATH",
            "/private-assistant-ai-serving/qwen25-coder-7b/v1",
        ),
        ssl_verify=False,
    )

    return MaaSLiteLlm(
        model=f"openai/{model_name}",
        api_base=cfg.api_base(model_name),
        api_key=cfg.api_key,
        drop_params=True,
    )

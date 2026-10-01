"""Telemetry setup — self-hosted variant.

The original recipe exported traces to Cloud Trace via GCS.  This version
uses only stdlib logging; add an OTLP exporter env var if you want to
ship spans to a local Jaeger / Tempo instance on the cluster.
"""

import logging


def setup_telemetry() -> None:
    """No-op in the self-hosted variant — telemetry is optional."""
    logging.info("Telemetry: running in self-hosted mode (no Cloud export)")

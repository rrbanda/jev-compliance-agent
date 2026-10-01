# Cross-Border Data Router Agent — ADK + Laya + Qwen on RHOAI
#
# Follows the Red Hat agentic-starter-kits ADK template pattern:
# https://github.com/red-hat-data-services/agentic-starter-kits
#
# Build:
#   podman build -t cross-border-data-router:latest agent/
#
# Run locally:
#   podman run --rm -p 8080:8080 \
#     -e MAAS_BASE_URL=https://maas.apps.ocp... \
#     -e MAAS_API_KEY=unused \
#     -e LAYA_API_URL=https://laya-api-gpu-laya-demo.apps.ocp... \
#     cross-border-data-router:latest

FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /opt/app-root/src

USER 0

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files for dependency installation
COPY pyproject.toml ./

# Install from pyproject.toml
COPY app/ ./app/
RUN uv pip install --python /opt/app-root/bin/python3 --no-cache ".[tracing]"

# Copy the entrypoint and playground UI
COPY main.py .
COPY playground/ ./playground/

# Ensure correct ownership
RUN chown -R 1001:0 /opt/app-root/src && chmod -R g=u /opt/app-root/src

USER 1001

EXPOSE 8080

ENV PORT=8080 \
    PYTHONPATH=/opt/app-root/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

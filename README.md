# Laya + Agentic AI on Red Hat OpenShift AI

## The Problem

Financial institutions operating across borders face a constant tension: **every piece of customer data must be classified and routed to the right jurisdiction before it can be processed** — and getting it wrong means regulatory fines (GDPR: up to 4% of global revenue, CCPA: $7,500 per violation).

Today this is typically handled by:
- **Manual review** — slow, expensive, doesn't scale
- **Rule-based systems** — brittle, can't handle unstructured text
- **LLMs** — can reason about regulations, but hallucinate classifications, take seconds per request, and often require sending data to external APIs (a compliance violation in itself)

None of these work well alone. What's needed is a system that is **fast** (classify data in milliseconds, not seconds), **reliable** (auditable confidence scores, not hallucinated labels), **smart** (reason about complex multi-jurisdiction rules), and **sovereign** (nothing leaves your infrastructure).

## The Solution

This demo combines two AI systems — each doing what it's best at — into a single agentic pipeline, running entirely on [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai) (RHOAI). No external API calls leave the cluster.

| Layer | Component | What it does | Latency |
|---|---|---|---|
| **System 1** — Fast Classifier | [Laya](https://github.com/convaiinnovations/laya) (421M params, GPU) | Classifies data as PII / financial / health / public with calibrated confidence scores | ~28 ms |
| **System 2** — Reasoning LLM | Qwen 2.5 Coder 7B (vLLM) | Extracts context, infers regulatory requirements, orchestrates multi-step routing | ~2-5 s |
| **Policy Engine** | Python rule engine | Enforces hard compliance filters — residency, jurisdiction exclusions | <1 ms |
| **Orchestrator** | [Google ADK 2.0](https://adk.dev/) | Wires it all together as an agentic tool-calling pipeline | — |
| **Observability** | [MLflow](https://mlflow.org/) tracing | Logs every LLM call and tool invocation for audit and debugging | — |

> **Why two models?** An LLM alone takes seconds and can hallucinate a "PII" classification that's actually financial data — a routing mistake with regulatory consequences. Laya gives a reliable, auditable classification in 28ms with calibrated probabilities. The LLM then reasons about *what to do* with that classification: which regulations apply, which jurisdictions are allowed, and which processor should handle the data. **Fast where it matters, smart where it counts.**

---

## Live Demo

Two playground UIs are deployed on the cluster:

| Playground | What to test | URL |
|---|---|---|
| **Agent** (combined) | Full pipeline: classify data → apply policy → route to jurisdiction | `https://cbdr-agent-laya-demo.apps.ocp.<cluster>/` |
| **Laya** (standalone) | Raw System 1 classification speed and accuracy | `https://laya-playground-laya-demo.apps.ocp.<cluster>/` |

**Try the Agent Playground** — type a routing request like:

> *Route this data: A customer record with full name, date of birth, and national insurance number from London, UK. Required residency: UK. Excluded jurisdictions: CN, RU.*

The chat UI shows every step: Laya classification (blue tool call), policy engine result (green), and the agent's final routing decision.

---

## How It Works

```
  User: "Route this German PII record under GDPR"
    │
    ▼
┌── ADK Agent (Qwen 2.5 on RHOAI) ────────────────────────┐
│                                                           │
│  Step 1: classify_with_laya(text)          ⚡ ~28ms       │
│          → Laya GPU: "PII, 73% confidence"               │
│                                                           │
│  Step 2: evaluate_and_route(...)           ⚡ <1ms        │
│          → Policy engine filters:                        │
│            ✓ eu_processor — residency EU/EEA covers EU   │
│            ✓ uk_processor — residency UK/EU covers EU    │
│            ✗ us_processor — residency US, no EU coverage │
│          → Scores: eu_processor=1.0 (preferred)          │
│                                                           │
│  Step 3: Delegate to eu_processor                        │
│          → "Processed in EU-WEST under GDPR controls"    │
└───────────────────────────────────────────────────────────┘
```

### Three-stage policy engine

The policy engine enforces hard compliance rules *before* any scoring. A non-compliant agent cannot out-score its way into eligibility.

1. **Residency filter** — does the agent's data-residency region cover the required residency?
2. **Jurisdiction exclusion** — is the agent's jurisdiction on the exclusion list?
3. **Scoring** — among compliant agents, score by preference, residency coverage, and compliance tags

### Agent-to-Agent protocol

Each regional processor (EU, UK, US) is registered as an [A2A AgentCard](https://google.github.io/A2A/) with [OpenEAGO](https://openeago.finos.org/) geographic metadata — jurisdiction, data-residency regions, compliance tags (GDPR, CCPA, etc.). This makes routing decisions auditable and standards-based.

---

## What's Deployed

### RHOAI InferenceServices (Laya)

| Component | Resource | Description |
|---|---|---|
| **CPU API** | `InferenceService/laya-cpu` | Laya on CPU (~145 ms), `english` + `multilingual` checkpoints |
| **GPU API** | `InferenceService/laya-gpu` | Laya on NVIDIA A10G (~28 ms), same checkpoints |
| **Laya Playground** | `InferenceService/laya-playground` | Interactive web UI for standalone Laya testing |

All three use **custom ServingRuntimes** — the same mechanism RHOAI uses for vLLM, OVMS, and Triton.

### LLM (Qwen 2.5 Coder 7B)

Served by vLLM via RHOAI's MaaS gateway (`LLMInferenceService`). The agent connects through [`rh-maas-litellm`](https://github.com/rrbanda/rh-maas-litellm), an ADK LiteLlm adapter that handles the `tools + response_format` conflict that vLLM rejects.

### ADK Agent

A standard OpenShift `Deployment` with an OpenAI-compatible `/chat/completions` endpoint and the playground chat UI at `/`. Built as a container image via OpenShift `BuildConfig`.

### MLflow Tracing (optional)

When `MLFLOW_TRACKING_URI` is set, the agent automatically logs every LLM call (via LiteLLM autolog) and can trace tool invocations to an MLflow server on the cluster. If the MLflow server is unreachable or the env var is unset, the agent continues without tracing — no crash, no degradation.

---

## Deploy It Yourself

### Prerequisites

- OpenShift 4.16+ with **RHOAI 3.5+** installed
- KServe component enabled (RawDeployment mode is sufficient)
- NVIDIA GPU Operator + at least one GPU node (for GPU variant)
- A vLLM-served LLM accessible via MaaS gateway (or any OpenAI-compatible endpoint)
- `oc` CLI authenticated as cluster-admin or project admin

### Step 1 — Laya infrastructure

```bash
# Create namespace, storage, build images
oc apply -f manifests/00-namespace.yaml
oc apply -f manifests/01-pvcs.yaml
oc apply -f manifests/02-build.yaml
oc start-build laya --from-dir=. -n laya-demo --follow
oc start-build laya-cuda --from-dir=. -n laya-demo --follow

# Deploy ServingRuntimes, InferenceServices, Routes
oc apply -f manifests/03-serving-runtimes.yaml
oc apply -f manifests/04-inference-services.yaml
oc apply -f manifests/05-routes.yaml

# Verify (12-point check)
scripts/verify.sh
```

### Step 2 — ADK agent

```bash
# Create the BuildConfig (one-time)
oc new-build --binary --name=cbdr-agent -n laya-demo

# Build the agent container image
cd agent && oc start-build cbdr-agent --from-dir=. -n laya-demo --follow

# Create the MaaS API key secret (get the key from your MaaS gateway)
oc create secret generic maas-apikey -n laya-demo \
  --from-literal=api_key=<your-maas-api-key>

# Deploy the agent (Deployment + Service + Route)
oc apply -f manifests/06-agent.yaml
```

The agent manifest (`06-agent.yaml`) configures:
- `MAAS_BASE_URL` — your MaaS gateway URL
- `MAAS_API_KEY` — pulled from the `maas-apikey` secret (never hardcoded)
- `LAYA_API_URL` — Laya GPU endpoint (route URL)
- `MODEL_NAME` — the vLLM model name

### Step 3 — Verify

```bash
# Agent health
curl -sk https://$(oc get route cbdr-agent -n laya-demo -o jsonpath='{.spec.host}')/health

# End-to-end test
curl -sk https://$(oc get route cbdr-agent -n laya-demo -o jsonpath='{.spec.host}')/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cross-border-data-router",
    "messages": [{"role": "user", "content": "Route this data: A financial transaction record with account numbers from Frankfurt, Germany. Required residency: EU. Excluded jurisdictions: US, CN."}]
  }'
```

---

## Running Locally

```bash
cd agent
cp .env.example .env          # Edit with your endpoint URLs
pip install -e .
adk run app                   # ADK dev server with built-in UI
```

Or run the automated demo script against live RHOAI endpoints:

```bash
scripts/agent-demo.sh         # 3 FSI scenarios with pass/fail
```

---

## Demo Scenarios

### Scenario 1: EU PII (GDPR)
German customer record (name, DOB, account number) → Laya classifies as **PII** → policy requires EU residency → routed to `eu_processor` (Frankfurt, GDPR controls).

### Scenario 2: US Financial (CCPA)
US bank wire transfer → Laya classifies as **financial** → policy requires US residency → routed to `us_processor` (Iowa, CCPA compliance).

### Scenario 3: Cross-Border Conflict
EU health record with *all* jurisdictions excluded → policy correctly **rejects** the request — no compliant processor available, escalates to human review.

---

## API Reference

### Laya — `/v1/systemone` (Jev wire protocol)

```bash
curl -sk -X POST \
  https://$(oc get route laya-api-gpu -n laya-demo -o jsonpath='{.spec.host}')/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Customer Hans Mueller, Berlin. DOB: 1985-03-15. Account DE89370400440532013000.",
    "questions": {
      "data_type": {
        "type": "choice",
        "instructions": "What type of sensitive data is present?",
        "criteria": {
          "PII": {"description": "Personal identifiable information"},
          "financial": {"description": "Financial records or transactions"},
          "public": {"description": "Non-sensitive public data"}
        }
      }
    }
  }'
```

Response:
```json
{
  "model": "laya-rl-agent",
  "answers": {
    "data_type": {
      "choice": "PII",
      "probabilities": {"PII": 0.73, "financial": 0.15, "public": 0.12},
      "confidence": 0.58,
      "answer_confidence": 0.73
    }
  },
  "usage": {"input_tokens": 48, "output_tokens": 0}
}
```

### Agent — `/chat/completions` (OpenAI-compatible)

```bash
curl -sk -X POST \
  https://$(oc get route cbdr-agent -n laya-demo -o jsonpath='{.spec.host}')/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cross-border-data-router",
    "stream": false,
    "messages": [{"role": "user", "content": "Route EU PII from Germany, required residency EU, exclude US and CN"}]
  }'
```

Supports `"stream": true` for Server-Sent Events (used by the playground UI).

---

## Performance

| Metric | CPU | GPU (A10G) |
|---|---|---|
| Laya classification | ~145 ms | ~28 ms |
| Laya via Route (edge TLS) | ~284 ms | ~173 ms |
| Full agent pipeline (Laya + Qwen + policy) | — | ~5-8 s |
| Memory (Laya) | ~4 Gi | ~4 Gi |

---

## Repository Structure

```
manifests/
  00-namespace.yaml              Namespace with RHOAI dashboard labels
  01-pvcs.yaml                   PVCs for HuggingFace model cache
  02-build.yaml                  BuildConfigs + ImageStream (CPU & CUDA)
  03-serving-runtimes.yaml       3 custom ServingRuntimes (CPU, GPU, Playground)
  04-inference-services.yaml     3 KServe InferenceServices
  05-routes.yaml                 External Routes with edge TLS
  06-agent.yaml                  Agent Deployment + Service + Route

agent/
  main.py                        FastAPI entrypoint (OpenAI-compatible API + Playground UI)
  Dockerfile                     UBI9 container image (uv + uvicorn)
  app/
    agent.py                     ADK root agent (orchestrator)
    model.py                     Shared LLM config (rh-maas-litellm → Qwen)
    prompt.py                    System 1+2 orchestrator prompt
    tools/
      laya_tool.py               classify_with_laya — calls Laya GPU
      routing_tool.py            evaluate_and_route — calls policy engine
    app_utils/
      telemetry.py               MLflow tracing (graceful degradation if unavailable)
    policy/
      engine.py                  3-stage routing policy engine
      cards.py                   A2A AgentCard registry (OpenEAGO metadata)
      models.py                  Data models for routing requests/decisions
    sub_agents/                  EU, UK, US regional processor agents
  playground/
    templates/index.html         Chat UI (streaming, tool-call visualization)
  tests/                         Unit + integration tests
  .env.example                   Endpoint configuration template

scripts/
  verify.sh                      12-point infrastructure verification
  demo.sh                        Interactive Laya FSI demo (5 scenarios)
  agent-demo.sh                  Agent + Laya + Qwen integration demo (3 scenarios)
```

---

## RHOAI Features Demonstrated

| Feature | How It's Used |
|---|---|
| **KServe InferenceService** | Managed model serving with health probes, scaling, lifecycle |
| **Custom ServingRuntime** | Laya packaged as a reusable RHOAI runtime template |
| **RawDeployment mode** | Lightweight KServe — no Knative/Istio required |
| **LLMInferenceService** | Qwen 2.5 Coder 7B served by vLLM via MaaS gateway |
| **MaaS Gateway** | Unified API gateway for LLM endpoints |
| **RHOAI Dashboard** | All resources labeled `opendatahub.io/dashboard: "true"` |
| **GPU scheduling** | `nodeSelector` + tolerations for NVIDIA A10G nodes |
| **OpenShift BuildConfig** | Binary Docker builds for all container images |
| **MLflow Tracing** | Optional LLM call logging and tool-invocation tracing for audit |

---

## Cleanup

```bash
oc delete project laya-demo
```

---

## Related

- [Laya](https://github.com/convaiinnovations/laya) — on-device decision engine
- [Jev](https://jev.ai) — wire protocol Laya speaks
- [rh-maas-litellm](https://github.com/rrbanda/rh-maas-litellm) — ADK LiteLlm adapter for Red Hat MaaS
- [Google ADK](https://adk.dev/) — Agent Development Kit 2.0
- [A2A Protocol](https://google.github.io/A2A/) — Agent-to-Agent communication
- [OpenEAGO](https://openeago.finos.org/) — geographic/compliance metadata for A2A
- [Agentic Starter Kits](https://github.com/red-hat-data-services/agentic-starter-kits) — Red Hat ADK deployment templates
- [RHOAI docs: Custom ServingRuntimes](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/configuring_your_model-serving_platform/configuring_model_servers)

## License

Demo manifests and agent code: Apache-2.0. Laya itself is licensed under its own terms — see the [Laya repository](https://github.com/convaiinnovations/laya).

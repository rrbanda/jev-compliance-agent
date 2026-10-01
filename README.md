# Laya on Red Hat OpenShift AI

Deploy [Laya](https://github.com/convaiinnovations/laya), a fast on-device **System 1 decision engine** (421M params, ~28 ms on GPU), as a [KServe](https://kserve.github.io/website/) `InferenceService` on [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai) (RHOAI) 3.5+.

> **TL;DR** — Three custom `ServingRuntime` definitions, three `InferenceService` deployments, edge-TLS routes, GPU scheduling, and RHOAI dashboard integration — all from `oc apply`.

---

## What Is Laya?

Laya is an open-source, on-device decision model that answers **typed questions** — yes/no, multiple choice, scored — in a single forward pass. It speaks the [Jev](https://jev.ai) `/v1/systemone` wire protocol, so any Jev client can point at Laya and keep working unchanged.

| Metric | Value |
|---|---|
| Parameters | 421 M |
| GPU latency (A10G) | ~28 ms |
| CPU latency | ~145 ms |
| Checkpoints | `english`, `multilingual`, `typed-decisions` |
| Protocol | Jev `/v1/systemone` (REST) |

---

## What This Deploys

| Component | KServe Resource | Description |
|---|---|---|
| **CPU API** | `InferenceService/laya-cpu` | Decision API on CPU (~145 ms) |
| **GPU API** | `InferenceService/laya-gpu` | Decision API on NVIDIA A10G GPU (~28 ms) |
| **Playground** | `InferenceService/laya-playground` | Interactive web UI for trying decisions |

Each uses a **custom `ServingRuntime`** — the same mechanism RHOAI uses for vLLM, OVMS, and Triton — but with Laya's own model server.

---

## Architecture

```
┌── RHOAI Cluster (OpenShift 4.16+, RHOAI 3.5+) ──────────┐
│                                                            │
│  ServingRuntime: laya-cpu-runtime                          │
│  └─ InferenceService: laya-cpu  ──→ Route (edge TLS)      │
│     └─ Pod: kserve-container · kube-rbac-proxy · agent     │
│                                                            │
│  ServingRuntime: laya-gpu-runtime                          │
│  └─ InferenceService: laya-gpu  ──→ Route (edge TLS)      │
│     └─ Pod: kserve-container · kube-rbac-proxy · agent     │
│        └─ nvidia.com/gpu: 1  (NVIDIA A10G)                 │
│                                                            │
│  ServingRuntime: laya-playground-runtime                   │
│  └─ InferenceService: laya-playground ──→ Route            │
│     └─ Pod: kserve-container · kube-rbac-proxy · agent     │
│                                                            │
│  Storage: 3 × PVC (10 Gi, HuggingFace model cache)        │
│  Images:  BuildConfig → ImageStream (CPU + CUDA)           │
└────────────────────────────────────────────────────────────┘
```

---

## RHOAI Features Demonstrated

| Feature | How It's Used |
|---|---|
| **KServe InferenceService** | Managed model serving with health probes, HPA, lifecycle |
| **Custom ServingRuntime** | Laya's Python server packaged as a reusable runtime template |
| **RawDeployment mode** | Lightweight KServe — no Knative/Istio dependency |
| **RHOAI Dashboard** | All resources labeled `opendatahub.io/dashboard: "true"` |
| **Auto-injected sidecars** | `kube-rbac-proxy` + `kserve-agent` added to every pod |
| **GPU scheduling** | `nodeSelector` + tolerations for NVIDIA A10G nodes |
| **OpenShift BuildConfig** | Binary Docker builds for CPU and CUDA images |

---

## Prerequisites

- OpenShift 4.16+ with **RHOAI 3.5+** installed
- KServe component enabled (RawDeployment mode is sufficient)
- NVIDIA GPU Operator + at least one GPU node (for the GPU variant)
- `oc` CLI authenticated as cluster-admin or project admin

---

## Quick Start

```bash
# 1. Create namespace and storage
oc apply -f manifests/00-namespace.yaml
oc apply -f manifests/01-pvcs.yaml

# 2. Build container images (run from the Laya source checkout)
oc apply -f manifests/02-build.yaml
oc start-build laya --from-dir=. -n laya-demo --follow
oc start-build laya-cuda --from-dir=. -n laya-demo --follow

# 3. Deploy ServingRuntimes + InferenceServices
oc apply -f manifests/03-serving-runtimes.yaml
oc apply -f manifests/04-inference-services.yaml

# 4. Expose via Routes
oc apply -f manifests/05-routes.yaml

# 5. Verify (12-point check)
scripts/verify.sh
```

---

## API Usage

### Health check

```bash
curl -sk https://$(oc get route laya-api-gpu -n laya-demo -o jsonpath='{.spec.host}')/health
```

### Inference (Jev `/v1/systemone` wire protocol)

```bash
curl -sk -X POST \
  https://$(oc get route laya-api-gpu -n laya-demo -o jsonpath='{.spec.host}')/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "The customer wants to transfer $50,000 overseas immediately",
    "questions": {
      "risk": {
        "type": "choice",
        "instructions": "What is the compliance risk level?",
        "criteria": {
          "high_risk": {},
          "medium_risk": {},
          "low_risk": {}
        }
      }
    }
  }'
```

### Example response

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "risk": {
      "type": "choice",
      "choice": "high_risk",
      "probabilities": { "high_risk": 0.87, "medium_risk": 0.09, "low_risk": 0.04 },
      "confidence": 0.82,
      "answer_confidence": 0.87
    }
  },
  "usage": { "input_tokens": 42, "output_tokens": 0 },
  "routing": { "model": "english", "reason": "English Latin text" }
}
```

---

## Performance

| Metric | CPU | GPU (A10G) |
|---|---|---|
| In-cluster latency | ~145 ms | ~28 ms |
| External (via Route) | ~284 ms | ~173 ms |
| Memory usage | ~4 Gi | ~4 Gi |
| Checkpoints loaded | english, multilingual | english, multilingual |

---

## Repository Structure

```
manifests/
  00-namespace.yaml              Namespace with RHOAI dashboard labels
  01-pvcs.yaml                   PersistentVolumeClaims for model cache
  02-build.yaml                  BuildConfigs + ImageStream (CPU & CUDA)
  03-serving-runtimes.yaml       3 custom ServingRuntimes
  04-inference-services.yaml     3 KServe InferenceServices
  05-routes.yaml                 External Routes with edge TLS
scripts/
  verify.sh                     12-point end-to-end verification
  demo.sh                       Interactive FSI scenario demo
Dockerfile.demo                  Multi-stage container image
```

---

## Demo Scenarios (FSI)

Run `scripts/demo.sh` to walk through five financial-services scenarios:

1. **Trade Routing** — dark pool vs. exchange vs. hold (97% confidence)
2. **Compliance Triage** — insider trading detection (95% confidence)
3. **Jailbreak Detection** — prompt injection guardrails
4. **Multilingual** — French compliance routing via the `multilingual` checkpoint
5. **CPU vs GPU** — side-by-side latency comparison

---

## Cleanup

```bash
oc delete project laya-demo
```

---

## Related

- [Laya](https://github.com/convaiinnovations/laya) — the decision engine
- [Jev](https://jev.ai) — the wire protocol Laya speaks
- [RHOAI 3.5 docs: Custom ServingRuntimes](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/configuring_your_model-serving_platform/configuring_model_servers)
- [RHOAI 3.5 docs: Deploying models](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploying_models/deploying_models)

## License

Demo manifests: Apache-2.0. Laya itself is licensed under its own terms — see the [Laya repository](https://github.com/convaiinnovations/laya).

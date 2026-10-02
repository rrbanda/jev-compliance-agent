# Cross-Border Data Router

A compliance-aware data routing agent that combines **calibrated neural classification** (Laya) with **LLM reasoning** (Gemini) and a **deterministic policy engine**.

The key idea: Laya answers five typed questions in one forward pass (~145ms, zero generated tokens).  An application policy maps those calibrated probabilities to routing decisions.  Gemini explains the result.  The model provides the signal; the policy provides the decision.

## The Problem

Financial institutions processing data across borders face cascading regulatory requirements:

- **GDPR (EU)**: Personal data of EU residents cannot leave the EU/EEA without adequacy decisions.  Fines up to 4% of global annual revenue.
- **HIPAA (US)**: Protected health information must stay in approved jurisdictions.
- **CCPA (US)**: California consumers' personal data has transfer restrictions.
- **Conflicting obligations**: A German customer's health records may simultaneously require EU residency (GDPR), US reporting (FATCA), and contractual restrictions forbidding certain jurisdictions.

Current approaches fail because:
- **Manual classification** is slow, error-prone, and doesn't scale.
- **LLM-only classification** hallucinates categories, produces uncalibrated confidence, and takes seconds per request.

Neither provides **auditable probability distributions** that compliance teams need.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     User Request                             │
│  "Route this: Maria Schmidt, Berlin, Tax ID 12/345/67890"   │
└─────────────┬───────────────────────────────────────────────┘
              │
     Step 1   ▼   ONE Jev call, FIVE typed questions, ONE forward pass
┌─────────────────────────────────────────────────────────────┐
│                    Laya (System 1)                            │
│  421M encoder · ~145ms · 0 output tokens · non-autoregressive│
│                                                              │
│  data_type           choice → financial (0.70)               │
│  pii_detected        noul   → 0.33                           │
│  data_subject_location choice → EU (0.57)                    │
│  sensitivity         score  → 1.72 / 3                       │
│  needs_human_review  noul   → 0.17                           │
└─────────────┬───────────────────────────────────────────────┘
              │
     Step 2   ▼   Deterministic rules, tuneable thresholds
┌─────────────────────────────────────────────────────────────┐
│               Application Policy                             │
│                                                              │
│  PII override:  pii_detected=0.33 < 0.70 → no override      │
│  Regime table:  EU + financial → GDPR                        │
│  Residency:     EU/EEA required                              │
│  Excluded:      CHINA, RUSSIA                                │
│  Human review:  not needed (all signals clear)               │
└─────────────┬───────────────────────────────────────────────┘
              │
     Step 3   ▼   A2A Agent Cards + OpenEAGO geographic metadata
┌─────────────────────────────────────────────────────────────┐
│               Routing Engine                                 │
│  Residency filter → Exclusion filter → Score → EU Agent ✓   │
└─────────────┬───────────────────────────────────────────────┘
              │
     Step 4   ▼   Gemini reasons about the whole pipeline
┌─────────────────────────────────────────────────────────────┐
│               Gemini 2.5 Flash (System 2)                    │
│  Explains: "Classified as financial data for an EU subject.  │
│  GDPR applies. Routed to EU Agent (Frankfurt). No human      │
│  review needed — all confidence thresholds met."             │
└─────────────────────────────────────────────────────────────┘
```

## The Jev Question Schema

The five questions are answered **simultaneously** in one forward pass.  Each uses the right question type for its answer shape:

```python
COMPLIANCE_QUESTIONS = {
    "data_type": {                          # What kind of data?
        "type": "choice",
        "criteria": {
            "PII": "names, addresses, national IDs, tax IDs",
            "financial": "account numbers, transactions, invoices",
            "health": "diagnoses, treatment plans, insurance claims",
            "public": "non-sensitive public information",
        },
    },
    "pii_detected": {                       # Contains PII?
        "type": "noul",                     # Returns probability 0.0–1.0
    },
    "data_subject_location": {              # Where is the person?
        "type": "choice",
        "criteria": {"EU": "...", "UK": "...", "US": "...", "other": "..."},
    },
    "sensitivity": {                        # How sensitive?
        "type": "score",                    # Ordered scale 0–3
        "criteria": ["public", "internal", "confidential", "restricted"],
    },
    "needs_human_review": {                 # Ambiguous?
        "type": "noul",
    },
}
```

The probabilities are **calibrated** — when Laya says 0.94, the true positive rate is approximately 94%.  This is what compliance requires: not "the AI said PII", but "94% probability of PII exceeding our 70% threshold".

## The Application Policy

The model reads the data.  The policy applies the law.  This separation is what makes the system auditable and adaptable:

```python
# The model tells us WHERE and WHAT.  The law tells us the rest.
_REGIME_TABLE = {
    ("EU", "PII"):    ("GDPR",    ["EU", "EEA"], ["US", "CHINA", "RUSSIA"]),
    ("EU", "health"): ("GDPR",    ["EU", "EEA"], ["US", "CHINA", "RUSSIA"]),
    ("UK", "PII"):    ("UK_GDPR", ["UK", "EU"],  ["CHINA", "RUSSIA"]),
    ("US", "health"): ("HIPAA",   ["US"],         ["CHINA", "RUSSIA"]),
    ("US", "PII"):    ("CCPA",    ["US"],         []),
    # ... swap the policy without retraining the model
}

# Tuneable thresholds — raise to be more conservative
PII_THRESHOLD        = 0.70   # above → treat as PII
REVIEW_THRESHOLD     = 0.60   # above → flag for human review
SENSITIVITY_FLOOR    = 2.5    # above → mandatory escalation
CONFIDENCE_FLOOR     = 0.50   # below → escalate for low confidence
```

When a regulation changes, update the table — no model retraining, no redeployment of inference services.

## Demo Scenarios

### 1. EU Financial — Clean Routing

> "Route: Maria Schmidt, Berlin, Germany. Tax ID 12/345/67890. Invoice refund."

| Question | Answer | Probability |
|---|---|---|
| data_type | financial | 0.70 |
| pii_detected | — | 0.33 |
| location | EU | 0.57 |
| sensitivity | — | 1.72 |
| needs_review | — | 0.17 |

**Policy**: EU + financial → GDPR → route to EU Agent.  165ms.

### 2. US Health + PII Override

> "Route: Patient John Doe, SSN 123-45-6789. Diagnosis: Type 2 diabetes. Cleveland Clinic, Ohio."

| Question | Answer | Probability |
|---|---|---|
| data_type | health | 0.84 |
| pii_detected | — | **0.78** |
| location | US | 0.62 |
| sensitivity | — | 1.57 |
| needs_review | — | 0.18 |

**Policy**: pii_detected (0.78) ≥ 0.70 → PII override.  But original data_type was health → HIPAA still applies (not CCPA).  Route to US Agent.  140ms.

### 3. UK PII — Human Review Triggered

> "Route: James Wilson, 42 Baker Street, London NW1 6XE. NI Number: QQ 12 34 56 C. HSBC account. GBP 450 disputed."

| Question | Answer | Probability |
|---|---|---|
| data_type | financial | 0.94 |
| pii_detected | — | **0.57** |
| location | UK | **0.48** (other: 0.33) |
| sensitivity | — | 1.57 |
| needs_review | — | 0.21 |

**Policy**: Location confidence (0.48) is borderline.  pii_detected (0.57) conflicts with data_type=financial.  → **Human review required**.  Route to UK Agent but flag for compliance officer review.  145ms.

### 4. Public Data — No Restrictions

> "Route: Q3 2026 Earnings Report, Acme Corp (NYSE: ACME). Revenue $4.2B."

| Question | Answer | Probability |
|---|---|---|
| data_type | financial | **0.94** |
| pii_detected | — | 0.05 |
| location | other | **0.79** |
| sensitivity | — | 1.80 |
| needs_review | — | 0.16 |

**Policy**: No specific location + no PII → no regime → route anywhere.  143ms.

## Setup

### Local Development

```bash
cd agent
cp .env.example .env
# Edit .env — set GOOGLE_API_KEY to your Gemini key
uv sync
source .venv/bin/activate
uv run adk web .   # Opens ADK playground at http://localhost:8000
```

### Testing from the UI

There are two ways to interact with the agent through a browser:

**Option 1: ADK Playground** (recommended for development)

```bash
cd agent
uv run adk web .
```

Opens the Google ADK built-in playground at `http://localhost:8000`.  Shows the
full agent conversation with tool calls expanded — you can see each of the
four pipeline steps: `classify_with_laya` → `apply_compliance_policy` →
`evaluate_and_route` → sub-agent.

**Option 2: Custom Playground** (deployed endpoint)

```bash
cd agent
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

Opens the custom chat UI at `http://localhost:8080`.  This is the same UI
served when the agent is deployed to OpenShift.  It shows tool calls and
tool responses as styled cards alongside the agent's final answer.

**Try these example prompts** (click the buttons in the UI or paste):

1. **EU PII** → GDPR routing:
   > Classify and route: Customer Maria Schmidt, born 15 March 1988, Friedrichstrasse 42, 10117 Berlin, Germany. Tax ID: 12/345/67890. Requesting refund for duplicate charge on invoice #DE-2026-4411.

2. **US Health** → HIPAA + PII override:
   > Classify and route: Patient John Doe, SSN 123-45-6789, DOB 07/22/1975. Blue Cross Blue Shield policy #BCBS-2026-99887. Diagnosis: Type 2 diabetes mellitus. Cleveland Clinic, Ohio.

3. **UK PII** → human review triggered:
   > Classify and route: Account holder James Wilson, 42 Baker Street, London NW1 6XE. National Insurance Number: QQ 12 34 56 C. HSBC account ending 7891. Disputed direct debit of GBP 450.

4. **Public data** → no restrictions:
   > Classify and route: Q3 2026 Earnings Report for Acme Corp (NYSE: ACME). Revenue grew 12% YoY to $4.2B. Operating margin expanded to 18.5%.

### OpenShift Deployment

```bash
# Create secrets
oc create secret generic gemini-apikey -n laya-demo \
  --from-literal=api_key=<YOUR_GEMINI_KEY>

# Build the agent image
oc new-build --binary --name=cbdr-agent -n laya-demo
cd agent && oc start-build cbdr-agent --from-dir=. -n laya-demo --follow

# Deploy
oc apply -f manifests/06-agent.yaml
```

## Project Structure

```
agent/
├── app/
│   ├── agent.py                # Root agent — Gemini + Laya tool + policy + routing
│   ├── model.py                # Gemini model configuration
│   ├── prompt.py               # Orchestrator instructions (4-step pipeline)
│   ├── policy/                 # Deterministic routing engine
│   │   ├── cards.py            # A2A Agent Cards with OpenEAGO metadata
│   │   ├── engine.py           # Three-stage routing (residency → exclusion → score)
│   │   └── models.py           # DataRequest, RoutingDecision
│   ├── sub_agents/             # Regional processor agents (EU, UK, US)
│   └── tools/
│       ├── laya_tool.py        # Jev System 1 — 5 typed questions, 1 forward pass
│       ├── compliance_policy.py # Application policy — maps probabilities to decisions
│       └── routing_tool.py     # Policy engine tool wrapper
├── tests/
│   └── unit/
│       ├── test_compliance_policy.py  # 13 policy rule tests
│       └── test_tools.py              # Routing tool + sub-agent tests
├── Dockerfile
└── pyproject.toml
```

## Key Design Decisions

1. **The model answers "what is in the data?" — the policy answers "which law applies?"**  
   Regulatory applicability is law, not text classification.  Laya detects PII and geography; the regime table maps (location × data_type) → regulation.

2. **PII override preserves the original data type for regime lookup.**  
   Health data with PII (pii_detected ≥ 0.70) gets classified as PII for routing, but the original type is preserved — US health stays HIPAA, not CCPA.

3. **Low confidence triggers human review, not a fallback model.**  
   When any signal is below its confidence floor, the system flags for human review rather than calling a different model.  The probabilities are the evidence; the human makes the call.

4. **All five questions share one forward pass.**  
   Laya answers choice, noul, and score questions simultaneously — different heads on the same encoder pass.  Adding a question costs one extra row in the batch, not another inference call.

## RHOAI Features Used

| Feature | Usage |
|---|---|
| **KServe InferenceService** | Serves Laya with RawDeployment mode |
| **NVIDIA GPU Operator** | Manages GPU allocation |
| **Kueue** | GPU workload admission with booking-based quotas |
| **OpenShift BuildConfig** | Binary Docker builds for the agent container |
| **Routes** | TLS-terminated external access |

## License

Apache 2.0 — see [LICENSE](LICENSE).

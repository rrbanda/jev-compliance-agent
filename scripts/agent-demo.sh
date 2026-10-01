#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# agent-demo.sh — Cross-Border Data Router + Laya on RHOAI
#
# Demonstrates the System 1 (Laya) + System 2 (Qwen) agentic pipeline
# by exercising three FSI scenarios against the live RHOAI endpoints.
#
# Prerequisites:
#   - Laya GPU InferenceService running in laya-demo namespace
#   - Qwen 2.5 Coder 7B LLMInferenceService in private-assistant-ai-serving
#
# Usage:
#   ./scripts/agent-demo.sh                      # run all scenarios
#   ./scripts/agent-demo.sh --scenario 1         # run single scenario
# ────────────────────────────────────────────────────────────────────
set -uo pipefail

LAYA_URL="${LAYA_API_URL:-https://laya-api-gpu-laya-demo.apps.ocp.qn6c5.sandbox1388.opentlc.com}"
QWEN_URL="${LLM_API_BASE:-https://maas.apps.ocp.qn6c5.sandbox1388.opentlc.com/private-assistant-ai-serving/qwen25-coder-7b/v1}"
MODEL="${MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
PASS=0
FAIL=0

print_header() {
  echo ""
  echo -e "${BOLD}════════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}  $1${NC}"
  echo -e "${BOLD}════════════════════════════════════════════════════════${NC}"
}

print_step() {
  echo -e "\n${CYAN}► $1${NC}"
}

print_result() {
  if [ "$1" = "PASS" ]; then
    echo -e "  ${GREEN}✓ $2${NC}"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}✗ $2${NC}"
    FAIL=$((FAIL + 1))
  fi
}

# ── Scenario 1: EU PII — German customer record ────────────────────
scenario_1() {
  print_header "Scenario 1: EU PII Record (GDPR)"
  echo "  A German customer's personal data must stay within the EU."

  TEXT="Customer Hans Mueller, address: Friedrichstrasse 42, Berlin, Germany. Date of birth: 1985-03-15. Account ID: DE89370400440532013000."

  # Step 1: Laya System 1 classification
  print_step "System 1 — Laya classification (expect: PII, sub-30ms)"
  START=$(python3 -c 'import time; print(int(time.time()*1000))')
  LAYA_RESULT=$(curl -sk "${LAYA_URL}/v1/systemone" \
    -H "Content-Type: application/json" \
    -d "{
      \"state\": \"${TEXT}\",
      \"questions\": {
        \"data_type\": {
          \"type\": \"choice\",
          \"instructions\": \"What type of sensitive data is present?\",
          \"criteria\": {
            \"PII\": {\"description\": \"Personal identifiable information\"},
            \"financial\": {\"description\": \"Financial records or transactions\"},
            \"public\": {\"description\": \"Non-sensitive public data\"}
          }
        }
      }
    }" 2>/dev/null)
  END=$(python3 -c 'import time; print(int(time.time()*1000))')
  LAYA_MS=$((END - START))

  CLASSIFICATION=$(echo "$LAYA_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['answers']['data_type']['choice'])" 2>/dev/null)
  echo "  Laya says: ${CLASSIFICATION} (${LAYA_MS}ms)"

  if [ "$CLASSIFICATION" = "PII" ]; then
    print_result "PASS" "Laya correctly classified as PII in ${LAYA_MS}ms"
  else
    print_result "FAIL" "Expected PII, got: ${CLASSIFICATION}"
  fi

  # Step 2: Qwen System 2 tool-calling (routing decision)
  print_step "System 2 — Qwen routing decision via tool call"
  START=$(python3 -c 'import time; print(int(time.time()*1000))')
  QWEN_RESULT=$(curl -sk "${QWEN_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a data-routing assistant. Use the provided tool to route data.\"},
        {\"role\": \"user\", \"content\": \"Route this PII record from Germany under GDPR: ${TEXT}\"}
      ],
      \"tools\": [{\"type\": \"function\", \"function\": {
        \"name\": \"evaluate_and_route\",
        \"description\": \"Evaluate cross-border data-residency policy\",
        \"parameters\": {\"type\": \"object\", \"properties\": {
          \"data_classification\": {\"type\": \"string\"},
          \"origin_region\": {\"type\": \"string\"},
          \"required_residency\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}
        }, \"required\": [\"data_classification\", \"origin_region\", \"required_residency\"]}
      }}],
      \"tool_choice\": \"required\",
      \"max_tokens\": 256
    }" 2>/dev/null)
  END=$(python3 -c 'import time; print(int(time.time()*1000))')
  QWEN_MS=$((END - START))

  TOOL_CALL=$(echo "$QWEN_RESULT" | python3 -c "
import sys, json
r = json.load(sys.stdin)
tc = r['choices'][0]['message'].get('tool_calls', [])
if tc:
    args = json.loads(tc[0]['function']['arguments'])
    print(f\"classification={args.get('data_classification','?')} origin={args.get('origin_region','?')} residency={args.get('required_residency',[])}\" )
else:
    print('NO_TOOL_CALL')
" 2>/dev/null)
  echo "  Qwen says: ${TOOL_CALL} (${QWEN_MS}ms)"

  if echo "$TOOL_CALL" | grep -qi "PII"; then
    print_result "PASS" "Qwen produced correct tool call in ${QWEN_MS}ms"
  else
    print_result "FAIL" "Qwen tool call unexpected: ${TOOL_CALL}"
  fi

  echo -e "\n  ${YELLOW}Combined latency: Laya ${LAYA_MS}ms + Qwen ${QWEN_MS}ms = $((LAYA_MS + QWEN_MS))ms${NC}"
}

# ── Scenario 2: US Financial — CCPA transaction ────────────────────
scenario_2() {
  print_header "Scenario 2: US Financial Record (CCPA)"
  echo "  A US bank transaction must be processed within the US."

  TEXT="Wire transfer USD 45,000 from account 4532-0198-7766 to beneficiary account 8891-2233-4455, Wells Fargo San Francisco branch, ref TXN-2026-09-28-4421."

  print_step "System 1 — Laya classification (expect: financial)"
  START=$(python3 -c 'import time; print(int(time.time()*1000))')
  LAYA_RESULT=$(curl -sk "${LAYA_URL}/v1/systemone" \
    -H "Content-Type: application/json" \
    -d "{
      \"state\": \"${TEXT}\",
      \"questions\": {
        \"data_type\": {
          \"type\": \"choice\",
          \"instructions\": \"What type of sensitive data is present?\",
          \"criteria\": {
            \"PII\": {\"description\": \"Personal identifiable information\"},
            \"financial\": {\"description\": \"Financial records or transactions\"},
            \"public\": {\"description\": \"Non-sensitive public data\"}
          }
        }
      }
    }" 2>/dev/null)
  END=$(python3 -c 'import time; print(int(time.time()*1000))')
  LAYA_MS=$((END - START))

  CLASSIFICATION=$(echo "$LAYA_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['answers']['data_type']['choice'])" 2>/dev/null)
  echo "  Laya says: ${CLASSIFICATION} (${LAYA_MS}ms)"

  if [ "$CLASSIFICATION" = "financial" ]; then
    print_result "PASS" "Laya correctly classified as financial in ${LAYA_MS}ms"
  else
    print_result "FAIL" "Expected financial, got: ${CLASSIFICATION}"
  fi

  print_step "System 2 — Qwen routing decision"
  START=$(python3 -c 'import time; print(int(time.time()*1000))')
  QWEN_RESULT=$(curl -sk "${QWEN_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [
        {\"role\": \"system\", \"content\": \"You are a data-routing assistant. Use the provided tool to route data.\"},
        {\"role\": \"user\", \"content\": \"Route this financial record from the US under CCPA: ${TEXT}\"}
      ],
      \"tools\": [{\"type\": \"function\", \"function\": {
        \"name\": \"evaluate_and_route\",
        \"description\": \"Evaluate cross-border data-residency policy\",
        \"parameters\": {\"type\": \"object\", \"properties\": {
          \"data_classification\": {\"type\": \"string\"},
          \"origin_region\": {\"type\": \"string\"},
          \"required_residency\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}
        }, \"required\": [\"data_classification\", \"origin_region\", \"required_residency\"]}
      }}],
      \"tool_choice\": \"required\",
      \"max_tokens\": 256
    }" 2>/dev/null)
  END=$(python3 -c 'import time; print(int(time.time()*1000))')
  QWEN_MS=$((END - START))

  TOOL_CALL=$(echo "$QWEN_RESULT" | python3 -c "
import sys, json
r = json.load(sys.stdin)
tc = r['choices'][0]['message'].get('tool_calls', [])
if tc:
    args = json.loads(tc[0]['function']['arguments'])
    print(f\"classification={args.get('data_classification','?')} origin={args.get('origin_region','?')} residency={args.get('required_residency',[])}\" )
else:
    print('NO_TOOL_CALL')
" 2>/dev/null)
  echo "  Qwen says: ${TOOL_CALL} (${QWEN_MS}ms)"

  if echo "$TOOL_CALL" | grep -qi "financial\|US"; then
    print_result "PASS" "Qwen produced correct tool call in ${QWEN_MS}ms"
  else
    print_result "FAIL" "Qwen tool call unexpected: ${TOOL_CALL}"
  fi

  echo -e "\n  ${YELLOW}Combined latency: Laya ${LAYA_MS}ms + Qwen ${QWEN_MS}ms = $((LAYA_MS + QWEN_MS))ms${NC}"
}

# ── Scenario 3: Cross-Border Conflict — EU data excluded from US ───
scenario_3() {
  print_header "Scenario 3: Cross-Border Conflict (Rejection)"
  echo "  EU PII with a contractual ban on ALL jurisdictions — must be rejected."

  TEXT="Patient Maria Garcia, Barcelona, Spain. Diagnosis: Type 2 Diabetes. Insurance ID: ES-SALUD-44221."

  print_step "System 1 — Laya classification"
  START=$(python3 -c 'import time; print(int(time.time()*1000))')
  LAYA_RESULT=$(curl -sk "${LAYA_URL}/v1/systemone" \
    -H "Content-Type: application/json" \
    -d "{
      \"state\": \"${TEXT}\",
      \"questions\": {
        \"data_type\": {
          \"type\": \"choice\",
          \"instructions\": \"What type of sensitive data is present?\",
          \"criteria\": {
            \"PII\": {\"description\": \"Personal identifiable information\"},
            \"financial\": {\"description\": \"Financial records\"},
            \"health\": {\"description\": \"Medical or health records\"},
            \"public\": {\"description\": \"Non-sensitive public data\"}
          }
        }
      }
    }" 2>/dev/null)
  END=$(python3 -c 'import time; print(int(time.time()*1000))')
  LAYA_MS=$((END - START))

  CLASSIFICATION=$(echo "$LAYA_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['answers']['data_type']['choice'])" 2>/dev/null)
  echo "  Laya says: ${CLASSIFICATION} (${LAYA_MS}ms)"

  if [ "$CLASSIFICATION" = "health" ] || [ "$CLASSIFICATION" = "PII" ]; then
    print_result "PASS" "Laya classified as ${CLASSIFICATION} in ${LAYA_MS}ms"
  else
    print_result "FAIL" "Expected health or PII, got: ${CLASSIFICATION}"
  fi

  print_step "Policy Engine — testing rejection (exclude EU, UK, US)"
  # Directly test the policy engine logic: when all jurisdictions are excluded,
  # the routing must be rejected.
  POLICY_RESULT=$(python3 -c "
import sys
sys.path.insert(0, '$(dirname "$0")/../agent')
from app.policy.engine import evaluate_routing_policy
from app.policy.models import DataRequest
req = DataRequest(
    data_classification='health',
    origin_region='Spain',
    required_residency=('EU',),
    excluded_jurisdictions=('EU', 'UK', 'US'),
)
d = evaluate_routing_policy(req)
print(f'decision={d.decision} reason={d.reason}')
" 2>/dev/null)
  echo "  Policy: ${POLICY_RESULT}"

  if echo "$POLICY_RESULT" | grep -q "rejected"; then
    print_result "PASS" "Policy correctly rejected — no compliant processor available"
  else
    print_result "FAIL" "Expected rejection, got: ${POLICY_RESULT}"
  fi
}

# ── Main ───────────────────────────────────────────────────────────
main() {
  print_header "Cross-Border Data Router + Laya Demo on RHOAI"
  echo -e "  ${CYAN}System 1${NC}: Laya GPU — ${LAYA_URL}"
  echo -e "  ${CYAN}System 2${NC}: Qwen 2.5 — ${QWEN_URL}"
  echo ""

  SCENARIO="${1:-all}"

  case "$SCENARIO" in
    --scenario)
      shift
      case "$1" in
        1) scenario_1 ;;
        2) scenario_2 ;;
        3) scenario_3 ;;
        *) echo "Unknown scenario: $1"; exit 1 ;;
      esac
      ;;
    *)
      scenario_1
      scenario_2
      scenario_3
      ;;
  esac

  echo ""
  print_header "Results"
  echo -e "  ${GREEN}Passed: ${PASS}${NC}  ${RED}Failed: ${FAIL}${NC}"
  echo ""

  if [ "$FAIL" -gt 0 ]; then
    exit 1
  fi
}

main "$@"

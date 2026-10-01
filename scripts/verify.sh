#!/usr/bin/env bash
# End-to-end verification of the Laya RHOAI demo deployment.
# Usage: ./scripts/verify.sh
set -uo pipefail

NS="laya-demo"
PASS=0
FAIL=0

check() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "  ✅ $label"
    PASS=$((PASS + 1))
  else
    echo "  ❌ $label"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== Checking InferenceServices ==="
for isvc in laya-cpu laya-gpu laya-playground; do
  ready=$(oc get inferenceservice "$isvc" -n "$NS" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Missing")
  check "$isvc Ready=$ready" [ "$ready" = "True" ]
done

echo ""
echo "=== Checking Pods ==="
for dep in laya-cpu-predictor laya-gpu-predictor laya-playground-predictor; do
  ready=$(oc get deployment "$dep" -n "$NS" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
  check "$dep replicas=$ready" [ "${ready:-0}" -ge 1 ]
done

echo ""
echo "=== Checking Routes ==="
CPU_HOST=$(oc get route laya-api-cpu -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null)
GPU_HOST=$(oc get route laya-api-gpu -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null)
PG_HOST=$(oc get route laya-playground -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null)

check "CPU health (https://$CPU_HOST/health)" \
  curl -sfk "https://$CPU_HOST/health"
check "GPU health (https://$GPU_HOST/health)" \
  curl -sfk "https://$GPU_HOST/health"
check "Playground UI (https://$PG_HOST/)" \
  curl -sfk "https://$PG_HOST/"

echo ""
echo "=== Checking GPU device ==="
device=$(curl -sfk "https://$GPU_HOST/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['device'])" 2>/dev/null || echo "unknown")
check "GPU reports device=cuda" [ "$device" = "cuda" ]

echo ""
echo "=== Checking CPU inference ==="
PAYLOAD='{"state":"hello world","questions":{"q":{"type":"choice","instructions":"pick one","criteria":{"a":{},"b":{}}}}}'
code=$(curl -sfk -o /dev/null -w '%{http_code}' -X POST "https://$CPU_HOST/v1/systemone" \
  -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null || echo "000")
check "CPU inference returns 200 (got $code)" [ "$code" = "200" ]

echo ""
echo "=== Checking GPU inference ==="
code=$(curl -sfk -o /dev/null -w '%{http_code}' -X POST "https://$GPU_HOST/v1/systemone" \
  -H 'Content-Type: application/json' -d "$PAYLOAD" 2>/dev/null || echo "000")
check "GPU inference returns 200 (got $code)" [ "$code" = "200" ]

echo ""
echo "────────────────────────────────"
echo "  Results: $PASS passed, $FAIL failed"
echo "────────────────────────────────"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1

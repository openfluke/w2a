#!/usr/bin/env bash
# Run full Go↔C++ comparison + all-train determinism suite; overwrite logs/mac.txt
# (also logs/darwin_arm64.txt). logs/ is gitignored — re-run anytime to refresh.
#
# Usage (from w2a or anywhere):
#   ./scripts/run_go_v_cpp_mac_snapshot.sh
#   WELVET_CPP_ROOT=/path/to/welvet.cpp ./scripts/run_go_v_cpp_mac_snapshot.sh
set -euo pipefail

W2A_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_CPP_ROOT="${WELVET_CPP_ROOT:-$W2A_ROOT/../welvet.cpp}"
BUILD="${BUILD_DIR:-$WELVET_CPP_ROOT/build}"
LOG_DIR="$W2A_ROOT/logs"
OUT_MAC="$LOG_DIR/mac.txt"
OUT_DARWIN="$LOG_DIR/darwin_arm64.txt"

export TMPDIR="${TMPDIR:-/tmp}"
export GOTMPDIR="${GOTMPDIR:-/tmp/gotmp-w2a-mac-snapshot}"
mkdir -p "$GOTMPDIR" "$LOG_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: mac snapshot requires Darwin (got $(uname -s))" >&2
  exit 1
fi

if [[ ! -d "$WELVET_CPP_ROOT" ]]; then
  echo "error: welvet.cpp not found at $WELVET_CPP_ROOT (set WELVET_CPP_ROOT)" >&2
  exit 1
fi

echo "==> building welvet.cpp → $BUILD"
bash "$WELVET_CPP_ROOT/scripts/build.sh"
TOOL="$BUILD/welvet_dense_fwd"
if [[ ! -x "$TOOL" ]]; then
  echo "error: missing tool $TOOL" >&2
  exit 1
fi

export WELVET_DENSE_FWD="$TOOL"
HOST="$(uname -s)/$(uname -m)"
VER="$("$TOOL" --version)"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  echo "=== welvet.cpp ↔ Go w2a mac snapshot ==="
  echo "date:    $START_UTC"
  echo "host:    $HOST"
  echo "tool:    $VER"
  echo "WELVET_DENSE_FWD=$WELVET_DENSE_FWD"
  echo "W2A_ROOT=$W2A_ROOT"
  echo
} > "$OUT_MAC"

rc_fwd=0
rc_train=0

echo "==> PART 1/2: TestGoVCpp (full Go↔C++ comparison + metrics tables)" | tee -a "$OUT_MAC"
(
  cd "$W2A_ROOT"
  go test ./tests/go_v_cpp/ -run '^TestGoVCpp$' -v -count=1 -timeout 120m
) 2>&1 | tee -a "$OUT_MAC" || rc_fwd=$?

echo | tee -a "$OUT_MAC"
echo "==> PART 2/2: TestGoVCppAllTrain (section timing table + totals)" | tee -a "$OUT_MAC"
(
  cd "$W2A_ROOT"
  go test ./tests/go_v_cpp/ -run '^TestGoVCppAllTrain$' -v -count=1 -timeout 180m
) 2>&1 | tee -a "$OUT_MAC" || rc_train=$?

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo
  echo "=== snapshot complete $END_UTC rc_go_v_cpp=$rc_fwd rc_all_train=$rc_train ==="
} | tee -a "$OUT_MAC"

cp -f "$OUT_MAC" "$OUT_DARWIN"
echo "==> wrote $OUT_MAC (+ $OUT_DARWIN)"

if [[ "$rc_fwd" -ne 0 || "$rc_train" -ne 0 ]]; then
  echo "error: snapshot tests failed (rc_fwd=$rc_fwd rc_train=$rc_train); log retained" >&2
  exit 1
fi
echo "==> ok"

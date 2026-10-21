#!/usr/bin/env bash
# Run full Go↔C++ comparison + all-train determinism suite; overwrite logs/linux_x86.txt
# (also logs/linux.txt). logs/ is gitignored — re-run anytime to refresh the snapshot.
#
# Usage (from w2a or anywhere):
#   ./scripts/run_go_v_cpp_linux_snapshot.sh
#   WELVET_CPP_ROOT=/path/to/welvet.cpp ./scripts/run_go_v_cpp_linux_snapshot.sh
set -euo pipefail

W2A_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WELVET_CPP_ROOT="${WELVET_CPP_ROOT:-$W2A_ROOT/../welvet.cpp}"
BUILD="${BUILD_DIR:-$WELVET_CPP_ROOT/build}"
LOG_DIR="$W2A_ROOT/logs"
OUT_X86="$LOG_DIR/linux_x86.txt"
OUT_LINUX="$LOG_DIR/linux.txt"

# Avoid sandbox/CI TMPDIR pointing at read-only trees
export TMPDIR="${TMPDIR:-/tmp}"
export GOTMPDIR="${GOTMPDIR:-/tmp/gotmp-w2a-snapshot}"
mkdir -p "$GOTMPDIR" "$LOG_DIR"

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
DISTRO="$(. /etc/os-release 2>/dev/null && echo "${ID:-unknown}" || echo unknown)"
VER="$("$TOOL" --version)"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  echo "=== welvet.cpp ↔ Go w2a snapshot ==="
  echo "date:    $START_UTC"
  echo "host:    $HOST  $DISTRO"
  echo "tool:    $VER"
  echo "WELVET_DENSE_FWD=$WELVET_DENSE_FWD"
  echo "W2A_ROOT=$W2A_ROOT"
  echo
} > "$OUT_X86"

rc_fwd=0
rc_train=0

echo "==> PART 1/2: TestGoVCpp (full Go↔C++ comparison + metrics tables)" | tee -a "$OUT_X86"
(
  cd "$W2A_ROOT"
  go test ./tests/go_v_cpp/ -run '^TestGoVCpp$' -v -count=1 -timeout 120m
) 2>&1 | tee -a "$OUT_X86" || rc_fwd=$?

echo | tee -a "$OUT_X86"
echo "==> PART 2/2: TestGoVCppAllTrain (section timing table + totals)" | tee -a "$OUT_X86"
(
  cd "$W2A_ROOT"
  go test ./tests/go_v_cpp/ -run '^TestGoVCppAllTrain$' -v -count=1 -timeout 180m
) 2>&1 | tee -a "$OUT_X86" || rc_train=$?

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo
  echo "=== snapshot complete $END_UTC rc_go_v_cpp=$rc_fwd rc_all_train=$rc_train ==="
} | tee -a "$OUT_X86"

cp -f "$OUT_X86" "$OUT_LINUX"
echo "==> wrote $OUT_X86 (+ $OUT_LINUX)"

if [[ "$rc_fwd" -ne 0 || "$rc_train" -ne 0 ]]; then
  echo "error: snapshot tests failed (rc_fwd=$rc_fwd rc_train=$rc_train); log retained" >&2
  exit 1
fi
echo "==> ok"

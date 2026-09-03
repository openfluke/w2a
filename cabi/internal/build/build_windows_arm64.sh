#!/usr/bin/env bash
# Windows arm64 via llvm-mingw (set LLVM_MINGW_HOME). Soft-skip if missing.
set -euo pipefail
cd "$(dirname "$0")"
./build_unix.sh --soft windows arm64 "$@"

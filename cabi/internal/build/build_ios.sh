#!/usr/bin/env bash
# iOS device archive (macOS only). Soft-skip elsewhere.
set -euo pipefail
cd "$(dirname "$0")"
./build_unix.sh --soft ios arm64 "$@"

#!/usr/bin/env bash
# Android NDK build (arm64 + x86_64). Soft-skip if NDK missing.
set -euo pipefail
cd "$(dirname "$0")"
./build_unix.sh --soft android arm64 "$@"
./build_unix.sh --soft android amd64

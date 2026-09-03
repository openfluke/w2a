#!/usr/bin/env bash
# Mirror apps/w2a/cabi/internal/build/dist → src/welvet/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/../cabi/internal/build"
exec "$BUILD/copy_to_python.sh" "$BUILD/dist"

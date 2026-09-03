#!/usr/bin/env bash
# Copy C-ABI dist/* into apps/w2a/python/src/welvet/
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-./dist}"
DST="$(cd ../../../python/src/welvet && pwd)"

if [[ ! -d "$SRC" ]]; then
	echo "Source not found: $SRC — build first" >&2
	exit 1
fi

mkdir -p "$DST"
cp -rv "$SRC"/* "$DST"/
echo "Mirrored $SRC → $DST"
ls -lh "$DST"/*/welvet.* 2>/dev/null | head -20 || true

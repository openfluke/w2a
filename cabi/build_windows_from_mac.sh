#!/usr/bin/env bash
# Cross-compile Welvet C-ABI Windows amd64 DLL from macOS via mingw-w64.
#
# Prereq (Homebrew):
#   brew install mingw-w64
#
# Usage:
#   ./build_windows_from_mac.sh           # windows amd64
#   ./build_windows_from_mac.sh --clean
#   ./build_windows_from_mac.sh --soft    # skip instead of fail if mingw missing
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
	cat >&2 <<'EOF'
ERROR: x86_64-w64-mingw32-gcc not found.

Install the Windows cross toolchain on macOS:
  brew install mingw-w64

Then re-run this script.
EOF
	exit 1
fi

echo "→ mingw: $(command -v x86_64-w64-mingw32-gcc)"
echo "→ $(x86_64-w64-mingw32-gcc --version | head -1)"
./internal/build/build_unix.sh "$@" windows amd64
echo "→ output: internal/build/dist/windows_amd64/welvet.dll"

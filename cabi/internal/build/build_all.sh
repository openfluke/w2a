#!/usr/bin/env bash
# Master try: every platform; SKIP missing toolchains; FAIL only if host linux build fails.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Welvet CABI build_all (soft) ==="
go run builder.go -os all -soft -clean "$@"

HOST_DIR=""
case "$(uname -s)-$(uname -m)" in
	Linux-x86_64|Linux-amd64) HOST_DIR=linux_amd64 ;;
	Linux-aarch64|Linux-arm64) HOST_DIR=linux_arm64 ;;
	Darwin-arm64) HOST_DIR=macos_arm64 ;;
	Darwin-x86_64) HOST_DIR=macos_amd64 ;;
esac

if [[ -n "$HOST_DIR" && ! -f "dist/$HOST_DIR/welvet.so" && ! -f "dist/$HOST_DIR/welvet.dylib" ]]; then
	echo "ERROR: host slice $HOST_DIR missing" >&2
	exit 1
fi

./copy_to_python.sh || true
echo "=== build_all done ==="

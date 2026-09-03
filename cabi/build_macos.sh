#!/usr/bin/env bash
# macOS dylib only (native arch).
set -euo pipefail
cd "$(dirname "$0")/internal/build"
case "$(uname -m)" in
	arm64|aarch64) arch=arm64 ;;
	*) arch=amd64 ;;
esac
./build_unix.sh --test darwin "$arch"

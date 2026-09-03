#!/usr/bin/env bash
# Linux C-ABI build (amd64 / arm64).
set -euo pipefail
cd "$(dirname "$0")"

FLAGS=()
ARCHES=()
host_arch="$(uname -m)"
case "$host_arch" in
	x86_64|amd64) native=amd64 ;;
	aarch64|arm64) native=arm64 ;;
	*) native=amd64 ;;
esac

for arg in "$@"; do
	case "$arg" in
		--clean) FLAGS+=(-clean) ;;
		--test) FLAGS+=(-test) ;;
		--soft) FLAGS+=(-soft) ;;
		all) ARCHES=(amd64 arm64) ;;
		amd64|arm64) ARCHES+=("$arg") ;;
		*)
			echo "usage: $0 [--clean] [--test] [--soft] [all|amd64|arm64]..." >&2
			exit 1
			;;
	esac
done
[[ ${#ARCHES[@]} -eq 0 ]] && ARCHES=("$native")

first=1
for arch in "${ARCHES[@]}"; do
	extra=("${FLAGS[@]}")
	if [[ $first -eq 0 ]]; then
		extra=()
		for f in "${FLAGS[@]}"; do
			[[ "$f" == "-clean" ]] || extra+=("$f")
		done
	fi
	./build_unix.sh "${extra[@]}" linux "$arch"
	first=0
done

#!/usr/bin/env bash
# Copy C-ABI dist/* into apps/w2a/python/src/welvet/
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-./dist}"
DST="$(cd ../../../python/src/welvet && pwd)"
WELVET_ROOT="$(cd ../../../../welvet && pwd)"

if [[ ! -d "$SRC" ]]; then
	echo "Source not found: $SRC — build first" >&2
	exit 1
fi

mkdir -p "$DST"
cp -rv "$SRC"/* "$DST"/

# Sibling accel libs next to welvet.so (Python pins WELVET_*_LIBRARY to these).
sync_native() {
	local plat="$1" src="$2" name="$3"
	local dest_dir="$DST/$plat"
	[[ -d "$dest_dir" ]] || return 0
	[[ -f "$src" ]] || return 0
	cp -v "$src" "$dest_dir/$name"
}
sync_native linux_amd64 "$WELVET_ROOT/native/cuda/linux_x86_64/libwelvet_cuda.so" libwelvet_cuda.so
sync_native linux_amd64 "$WELVET_ROOT/native/rocm/linux_x86_64/libwelvet_rocm.so" libwelvet_rocm.so
sync_native linux_amd64 "$WELVET_ROOT/native/vulkan/linux_x86_64/libwelvet_vulkan.so" libwelvet_vulkan.so
sync_native linux_amd64 "$WELVET_ROOT/native/cpp/linux_amd64/libwelvet.so" libwelvet.so
sync_native windows_amd64 "$WELVET_ROOT/native/cuda/windows_amd64/libwelvet_cuda.dll" libwelvet_cuda.dll
sync_native macos_arm64 "$WELVET_ROOT/native/metal/darwin_arm64/libwelvet_metal.dylib" libwelvet_metal.dylib

echo "Mirrored $SRC → $DST (+ native accel siblings when present)"
ls -lh "$DST"/*/welvet.* 2>/dev/null | head -20 || true

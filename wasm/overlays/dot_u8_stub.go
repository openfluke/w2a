//go:build !amd64 && !arm64

package simd

// WASM overlay: welvet simd stubs use invalid pointer indexing; no-ops are fine
// because simdEnabled() is false on js/wasm.
func dotU8AccSimd(q, k *uint8, n int, prev int32) int32 {
	_ = q
	_ = k
	_ = n
	return prev
}

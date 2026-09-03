//go:build !amd64 && !arm64

package simd

func simdEnabled() bool { return false }

func dotTileSimd(x, w *float32, n int, prev float64) float64 {
	_ = x
	_ = w
	_ = n
	return prev
}

// WASM overlay: missing on non-amd64/arm64 upstream.
func saxpyF32AccF64Simd(acc *float64, alpha float64, x *float32, n int) {
	_ = acc
	_ = alpha
	_ = x
	_ = n
}

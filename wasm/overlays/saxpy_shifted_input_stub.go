//go:build !amd64 && !arm64

package simd

func saxpyI8ShiftedInputGradAccSimd(gradIn *int32, weights *int8, gradOut int32, n int) {
	_ = gradIn
	_ = weights
	_ = gradOut
	_ = n
}

func saxpyU8ScaleI32AccSimd(gradW *int32, input *uint8, scale int32, n int) {
	_ = gradW
	_ = input
	_ = scale
	_ = n
}

func saxpyU8ShiftedInputGradAccSimd(gradIn *int32, weights *uint8, gradOut int32, n int) {
	_ = gradIn
	_ = weights
	_ = gradOut
	_ = n
}

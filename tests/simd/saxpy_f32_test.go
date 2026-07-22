package simd_test

import (
	"math"
	"testing"

	"github.com/openfluke/welvet/simd"
)

func TestSaxpyF32(t *testing.T) {
	n := 1000
	y := make([]float32, n)
	x := make([]float32, n)
	for i := 0; i < n; i++ {
		y[i] = float32(i) * 0.01
		x[i] = float32(i%7) * 0.1
	}
	want := make([]float32, n)
	copy(want, y)
	a := float32(1.25)
	for i := 0; i < n; i++ {
		want[i] += a * x[i]
	}
	simd.SaxpyF32(y, a, x, n)
	var maxErr float64
	for i := 0; i < n; i++ {
		e := math.Abs(float64(y[i] - want[i]))
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 1e-5 {
		t.Fatalf("max err %g", maxErr)
	}
}

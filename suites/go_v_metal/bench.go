package go_v_metal

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

// lossWithin1Ulp is true when a and b are identical or adjacent float64 bit patterns.
// On ARM, Go vs native scalar MSE often differs by 1–2 ULPs while f32 tensors stay exact_bit.
func lossWithin1Ulp(a, b float64) bool {
	// Allow a few ULPs: MSE reduction can differ by 1–3 ULPs when f32 posts
	// match under FormatNone×34 (seen on ConvT3 vol111 binary/int5).
	return lossWithinNUlp(a, b, 4)
}

func lossWithinNUlp(a, b float64, n uint64) bool {
	if a == b {
		return true
	}
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return false
	}
	ai, bi := math.Float64bits(a), math.Float64bits(b)
	if ai > bi {
		ai, bi = bi, ai
	}
	return bi-ai <= n
}

var defaultGoBenchIters = func() int {
	if value := os.Getenv("WELVET_METAL_BENCH_ITERS"); value != "" {
		n, err := strconv.Atoi(value)
		if err != nil || n < 1 {
			panic("WELVET_METAL_BENCH_ITERS must be a positive integer")
		}
		return n
	}
	return 2000
}()

const goBenchMinWall = 5 * time.Millisecond
const goBenchMaxIters = 2_000_000

func benchGoUS(fn func() error, minIters int) (us float64, err error) {
	if minIters < 1 {
		minIters = defaultGoBenchIters
	}
	for i := 0; i < 3; i++ {
		if err := fn(); err != nil {
			return 0, err
		}
	}
	n := minIters
	var d time.Duration
	for {
		t0 := time.Now()
		for i := 0; i < n; i++ {
			if err := fn(); err != nil {
				return 0, err
			}
		}
		d = time.Since(t0)
		if d >= goBenchMinWall || n >= goBenchMaxIters {
			break
		}
		if d <= 0 {
			n *= 10
			if n > goBenchMaxIters {
				n = goBenchMaxIters
			}
			continue
		}
		next := int(float64(n) * float64(goBenchMinWall) / float64(d) * 1.25)
		if next <= n {
			n *= 2
		} else {
			n = next
		}
		if n > goBenchMaxIters {
			n = goBenchMaxIters
		}
	}
	if n < 1 || d <= 0 {
		return 0, fmt.Errorf("benchGoUS: unmeasurable after %d iters (dur=%s)", n, d)
	}
	return float64(d.Nanoseconds()) / float64(n) / 1000.0, nil
}

func FormatUS(us float64) string {
	neg := us < 0
	if neg {
		us = -us
	}
	var s string
	switch {
	case us < 0.01:
		s = fmt.Sprintf("%.2fns", us*1000)
	case us < 10:
		s = fmt.Sprintf("%.3fµs", us)
	default:
		s = fmt.Sprintf("%.2fµs", us)
	}
	if neg {
		return "-" + s
	}
	return s
}

// formatDiffUS = mount − exec (positive ⇒ mount costlier than one warm exec).
func formatDiffUS(mountUS, execUS float64) string {
	return FormatUS(mountUS - execUS)
}

func perIterUS(d time.Duration, n int) float64 {
	if n < 1 || d <= 0 {
		return 0
	}
	return float64(d.Nanoseconds()) / float64(n) / 1000.0
}

// metalLossWithinBand validates the GPU float2 reduction independently of exact
// tensor checks. Float2 is not IEEE float64; this band is part of the Metal contract.
func metalLossWithinBand(a, b float64) bool {
	return !math.IsNaN(a) && !math.IsNaN(b) && !math.IsInf(a, 0) && !math.IsInf(b, 0) &&
		math.Abs(a-b) <= 1e-12*math.Max(1, math.Abs(a))
}

// Metal may flush subnormal FP32 values. Normal values still require exact bits;
// this exception applies only when both values lie below the smallest normal FP32.
func metalExactOrSubnormal(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	const smallestNormal = 0x1p-126
	for i, x := range a {
		y := b[i]
		if math.IsNaN(float64(x)) || math.IsNaN(float64(y)) || math.IsInf(float64(x), 0) || math.IsInf(float64(y), 0) {
			return false
		}
		if math.Float32bits(x) == math.Float32bits(y) {
			continue
		}
		if math.IsNaN(float64(x)) || math.IsNaN(float64(y)) || math.Abs(float64(x)) >= smallestNormal || math.Abs(float64(y)) >= smallestNormal {
			return false
		}
	}
	return true
}

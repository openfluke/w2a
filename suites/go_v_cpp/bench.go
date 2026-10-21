package go_v_cpp

import (
	"fmt"
	"math"
	"time"
)

// lossWithin1Ulp is true when a and b are identical or adjacent float64 bit patterns.
// On ARM, Go vs C++ scalar MSE often differs by 1 ULP while f32 tensors stay exact_bit.
func lossWithin1Ulp(a, b float64) bool {
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
	return bi-ai <= 1
}

// defaultGoBenchIters is the starting iteration count for Go wall-clock benches.
// On fast ARM hosts, 40 iters of a tiny op can finish in <1 QPC tick → go=0.00µs.
const defaultGoBenchIters = 2000

// goBenchMinWall keeps doubling iters until the Go sample is at least this long.
const goBenchMinWall = 5 * time.Millisecond

const goBenchMaxIters = 2_000_000

// benchGoUS warms up, then runs fn until wall time ≥ goBenchMinWall (or max iters).
// Returns average microseconds per call (fractional; may be ≪ 1).
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

// FormatUS formats a µs/op value so tiny Go benches are still readable (ns).
func FormatUS(us float64) string {
	if us < 0.01 {
		return fmt.Sprintf("%.2fns", us*1000)
	}
	if us < 10 {
		return fmt.Sprintf("%.3fµs", us)
	}
	return fmt.Sprintf("%.2fµs", us)
}

// perIterUS converts a total duration over n calls into µs/op.
func perIterUS(d time.Duration, n int) float64 {
	if n < 1 || d <= 0 {
		return 0
	}
	return float64(d.Nanoseconds()) / float64(n) / 1000.0
}

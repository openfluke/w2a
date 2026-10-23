package go_v_qnn

import (
	"fmt"
	"time"
)

const defaultGoBenchIters = 200
const goBenchMinWall = 20 * time.Millisecond
const goBenchMaxIters = 50_000 // cap — QNN online graphs accumulate until context free

// benchUS warms up, then averages until wall ≥ goBenchMinWall (or max iters).
func benchUS(fn func() error, minIters int) (us float64, err error) {
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
		return 0, fmt.Errorf("benchUS: unmeasurable after %d iters (dur=%s)", n, d)
	}
	return float64(d.Nanoseconds()) / float64(n) / 1000.0, nil
}

func formatUS(us float64) string {
	if us < 0.01 {
		return fmt.Sprintf("%.2fns", us*1000)
	}
	if us < 10 {
		return fmt.Sprintf("%.3fµs", us)
	}
	if us < 1000 {
		return fmt.Sprintf("%.2fµs", us)
	}
	return fmt.Sprintf("%.2fms", us/1000)
}

func speedPct(ref, other float64) float64 {
	if ref <= 1e-12 {
		return 0
	}
	return (other - ref) / ref * 100
}

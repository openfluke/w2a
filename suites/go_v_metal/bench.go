package go_v_metal

import (
	"fmt"
	"time"
)

const defaultGoBenchIters = 2000
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

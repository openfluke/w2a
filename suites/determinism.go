package suites

import (
	"fmt"
	"math"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
)

// Default det tolerances (Lucy-[7] spirit: tight fwd, looser bwd).
const (
	DetTolFwd = 1e-5
	DetTolBwd = 1e-4
	DetTolTrain = 5e-2
)

// MaxAbsDiff returns max |a[i]-b[i]| over the shorter length.
func MaxAbsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var max float64
	for i := 0; i < n; i++ {
		e := math.Abs(float64(a[i] - b[i]))
		if e > max {
			max = e
		}
	}
	return max
}

// CloneF32 copies a float32 slice.
func CloneF32(v []float32) []float32 {
	out := make([]float32, len(v))
	copy(out, v)
	return out
}

// SetGridMultiCore stamps MultiCore on the grid Exec (Place copies it onto layers).
func SetGridMultiCore(g *architecture.Grid, multi bool) {
	if g == nil {
		return
	}
	g.Exec.MultiCore = multi
}

// ApplyMultiCore sets MultiCore on Exec and Core descriptors.
func ApplyMultiCore(exec *core.ExecConfig, coreLayer *core.Layer, multi bool) {
	if exec != nil {
		exec.MultiCore = multi
	}
	if coreLayer != nil {
		coreLayer.MultiCore = multi
	}
}

// RequireDet fails if maxΔ exceeds tol.
func RequireDet(label string, maxDelta, tol float64) error {
	if maxDelta > tol {
		return fmt.Errorf("%s maxΔ=%g > tol=%g", label, maxDelta, tol)
	}
	return nil
}

// ShapeTier returns S/M/L dims for shape-tier smokes (not LLM model sizes).
func ShapeTier() []struct {
	Name string
	Dim  int
} {
	return []struct {
		Name string
		Dim  int
	}{
		{Name: "S", Dim: 32},
		{Name: "M", Dim: 64},
		{Name: "L", Dim: 128},
	}
}

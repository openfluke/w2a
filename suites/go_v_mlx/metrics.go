package go_v_mlx

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/nativemlx"
)

type LayerReport struct {
	Name string
	Geom string

	TransferUS float64
	CreateUS   float64
	FinalizeUS float64
	ExecUS     float64
	E2EUS      float64
	GoExecUS   float64

	HasAccuracy bool
	ExactPct    float64
	MaxAbs      float64
	RMSE        float64
	Band        suites.DetBand

	Err error
}

func rmse(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	var s float64
	for i := 0; i < n; i++ {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return math.Sqrt(s / float64(n))
}

func timeUS(fn func()) float64 {
	t0 := time.Now()
	fn()
	return float64(time.Since(t0).Nanoseconds()) / 1e3
}

func benchUS(fn func() error, iters int) (float64, error) {
	if iters < 1 {
		iters = 1
	}
	if err := fn(); err != nil {
		return 0, err
	}
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		if err := fn(); err != nil {
			return 0, err
		}
	}
	return float64(time.Since(t0).Nanoseconds()) / 1e3 / float64(iters), nil
}

func fillAccuracy(r *LayerReport, goOut, mlxOut []float32) {
	m := suites.CompareF32Spectrum(goOut, mlxOut)
	r.HasAccuracy = true
	r.ExactPct = m.ExactPct
	r.MaxAbs = m.MaxAbs
	r.RMSE = rmse(goOut, mlxOut)
	r.Band = m.Band
}

func takePhase(r *LayerReport, sess *nativemlx.Session) {
	ph := sess.LastPhase()
	r.CreateUS = float64(ph.CreateNs) / 1e3
	r.FinalizeUS = float64(ph.FinalizeNs) / 1e3
	r.ExecUS = float64(ph.ExecuteNs) / 1e3
}

func finishE2E(r *LayerReport) {
	r.E2EUS = r.TransferUS + r.CreateUS + r.FinalizeUS + r.ExecUS
}

func printReport(r LayerReport) {
	if r.Err != nil {
		fmt.Printf("      %s FAIL: %v\n", r.Name, r.Err)
		return
	}
	fmt.Printf("      %s %s  transfer=%.1fµs create=%.1fµs exec=%.1fµs e2e=%.1fµs",
		r.Name, r.Geom, r.TransferUS, r.CreateUS, r.ExecUS, r.E2EUS)
	if r.GoExecUS > 0 {
		fmt.Printf(" go=%.1fµs", r.GoExecUS)
	}
	if r.HasAccuracy {
		fmt.Printf("  exact=%.2f%% max|Δ|=%.3g rmse=%.3g band=%s",
			r.ExactPct, r.MaxAbs, r.RMSE, r.Band)
	}
	fmt.Println()
}

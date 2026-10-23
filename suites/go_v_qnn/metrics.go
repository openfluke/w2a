package go_v_qnn

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/nativeqnn"
)

// LayerReport is one README HTP layer measurement row.
//
// Protocol: transfer weights → create/finalize graph → time execute only.
// ExecUS is the headline (post-transfer NPU run). Transfer/create/finalize/e2e are reported too.
type LayerReport struct {
	Name string
	Geom string

	TransferUS float64 // host weight materialize / Sync push (Go → mirror)
	CreateUS   float64 // graphCreate + tensors + addNode
	FinalizeUS float64 // graphFinalize (compile)
	ExecUS     float64 // graphExecute only (after weights transferred)
	E2EUS      float64 // transfer + create + finalize + exec

	GoExecUS float64 // pure Go CPU forward (same x)

	HasAccuracy bool
	ExactPct    float64 // go↔qnn bit exact %
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

func fillAccuracy(r *LayerReport, goOut, qnnOut []float32) {
	m := suites.CompareF32Spectrum(goOut, qnnOut)
	r.HasAccuracy = true
	r.ExactPct = m.ExactPct
	r.MaxAbs = m.MaxAbs
	r.RMSE = rmse(goOut, qnnOut)
	r.Band = m.Band
}

func takePhase(r *LayerReport, sess *nativeqnn.Session) {
	ph := sess.LastPhase()
	r.CreateUS = ph.CreateUS()
	r.FinalizeUS = ph.FinalizeUS()
	r.ExecUS = ph.ExecuteUS()
}

func finishE2E(r *LayerReport) {
	r.E2EUS = r.TransferUS + r.CreateUS + r.FinalizeUS + r.ExecUS
}

func printReport(r LayerReport) {
	if r.Err != nil {
		fmt.Printf("      %-12s %-36s  FAIL: %v\n", r.Name, r.Geom, r.Err)
		return
	}
	fmt.Printf("      %-12s %-36s\n", r.Name, r.Geom)
	fmt.Printf("        transfer=%-10s  create=%-10s  finalize=%-10s  exec=%-10s  e2e=%-10s  go=%-10s\n",
		formatUS(r.TransferUS), formatUS(r.CreateUS), formatUS(r.FinalizeUS),
		formatUS(r.ExecUS), formatUS(r.E2EUS), formatUS(r.GoExecUS))
	if r.HasAccuracy {
		fmt.Printf("        go↔qnn  exact=%.2f%%  band=%s  max|Δ|=%.3g  RMSE=%.3g\n",
			r.ExactPct, r.Band, r.MaxAbs, r.RMSE)
	} else {
		fmt.Printf("        go↔qnn  (accuracy N/A — QNN phase timing only)\n")
	}
	if r.GoExecUS > 0 && r.ExecUS > 0 {
		fmt.Printf("        spdΔ(exec vs go)=%+.1f%%  (neg ⇒ QNN exec faster)\n",
			speedPct(r.GoExecUS, r.ExecUS))
	}
}

func gateReport(r LayerReport) error {
	if r.Err != nil {
		return r.Err
	}
	if !r.HasAccuracy {
		return nil
	}
	// Re-gate via band/NaN policy used by the rest of the suite.
	m := suites.DetMetrics{ExactPct: r.ExactPct, MaxAbs: r.MaxAbs, Band: r.Band}
	if m.Band < qnnPassBand {
		fmt.Printf("      WARN go↔qnn band=%s (<%s) exact=%.2f%% max|Δ|=%.3g RMSE=%.3g\n",
			m.Band, qnnPassBand, r.ExactPct, r.MaxAbs, r.RMSE)
	}
	return nil
}

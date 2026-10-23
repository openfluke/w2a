package go_v_qnn

import (
	"fmt"

	"github.com/openfluke/w2a/suites"
)

// qnnPassBand — HTP compose ≠ Go f32 exact; suite fails only on NaN mismatch.
const qnnPassBand = suites.DetBallpark

type tripleTiming struct {
	goUS, natUS, qnnUS float64
	natRan, qnnRan     bool
	qnnPrepared        bool // true ⇒ timed loop was execute-only after PrepareQNN
	goVsNat            suites.DetMetrics
	goVsQNN            suites.DetMetrics
}

func printTriple(label string, t tripleTiming) {
	natS := "skip"
	if t.natRan {
		natS = formatUS(t.natUS)
	}
	qnnS := "skip"
	qnnNote := ""
	if t.qnnRan {
		qnnS = formatUS(t.qnnUS)
		if t.qnnPrepared {
			qnnNote = " [exec-only]"
		} else {
			qnnNote = " [online graph]"
		}
	}
	fmt.Printf("      %-28s  go=%-10s  native=%-10s  qnn=%-10s%s\n",
		label, formatUS(t.goUS), natS, qnnS, qnnNote)
	// Determinism % is Go vs QNN only (user request).
	if t.qnnRan {
		fmt.Printf("        go↔qnn determinism=%.2f%%  band=%s  max|Δ|=%.3g  max_rel=%.3g  spdΔ(qnn vs go)=%+.1f%%\n",
			t.goVsQNN.ExactPct, t.goVsQNN.Band, t.goVsQNN.MaxAbs, t.goVsQNN.MaxRel,
			speedPct(t.goUS, t.qnnUS))
	}
	if t.natRan {
		fmt.Printf("        go↔native band=%s exact=%.2f%%  spdΔ(native vs go)=%+.1f%%\n",
			t.goVsNat.Band, t.goVsNat.ExactPct, speedPct(t.goUS, t.natUS))
	}
	if t.natRan && t.qnnRan {
		fmt.Printf("        spdΔ(qnn vs native)=%+.1f%%\n", speedPct(t.natUS, t.qnnUS))
	}
}

func gateQNN(m suites.DetMetrics) error {
	if m.HasNaNMismatch {
		return fmt.Errorf("go↔qnn NaN mismatch: %s", suites.FormatSpectrum(m))
	}
	if m.Band < qnnPassBand {
		fmt.Printf("      WARN go↔qnn band=%s (<%s): %s\n", m.Band, qnnPassBand, suites.FormatSpectrum(m))
	}
	return nil
}

func gateNative(m suites.DetMetrics) error {
	if m.Band < suites.DetStrict {
		return fmt.Errorf("go↔native band=%s (need ≥strict): %s", m.Band, suites.FormatSpectrum(m))
	}
	return nil
}

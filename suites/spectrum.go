package suites

import (
	"fmt"
	"math"
)

// Determinism spectrum for cross-impl / cross-backend numerical agreement.
// Welvet's aim for go_v_cpp is ExactBit (100% identical bits).
//
// Existing DetTolFwd (1e-5) / DetTolBwd (1e-4) sit in the Strict / Industry band.
// This ladder makes the grade explicit when reporting Go vs C++ (or SC↔MC) deltas.
type DetBand int

const (
	// DetBroken — outputs disagree wildly or contain NaN/Inf mismatch.
	DetBroken DetBand = iota
	// DetBallpark — "absolutely not deterministic" but same order of magnitude (~1e-1 abs/rel).
	DetBallpark
	// DetLoose — middle ground; often seen in casual float compares (~1e-2).
	DetLoose
	// DetIndustry — common ML / GPU parity tolerances (~1e-3 .. DetTolFwd).
	DetIndustry
	// DetStrict — Welvet suite forward gate (DetTolFwd = 1e-5).
	DetStrict
	// DetExactBit — 100% identical IEEE bits (Welvet / go_v_cpp target).
	DetExactBit
)

func (b DetBand) String() string {
	switch b {
	case DetBroken:
		return "broken"
	case DetBallpark:
		return "ballpark"
	case DetLoose:
		return "loose"
	case DetIndustry:
		return "industry"
	case DetStrict:
		return "strict"
	case DetExactBit:
		return "exact_bit"
	default:
		return "unknown"
	}
}

// PercentAlongSpectrum maps a band to 0..100 (ExactBit = 100).
func (b DetBand) Percent() float64 {
	switch b {
	case DetBroken:
		return 0
	case DetBallpark:
		return 20
	case DetLoose:
		return 40
	case DetIndustry:
		return 60
	case DetStrict:
		return 80
	case DetExactBit:
		return 100
	default:
		return 0
	}
}

// DetMetrics summarizes agreement between two float32 tensors (e.g. Go vs C++).
type DetMetrics struct {
	N           int
	ExactBits   int
	ExactPct    float64 // 0..100
	MaxAbs      float64
	MaxRel      float64
	MeanAbs     float64
	HasNaNMismatch bool
	Band        DetBand
}

// CompareF32Spectrum compares a vs b elementwise (bit-exact + abs/rel).
func CompareF32Spectrum(a, b []float32) DetMetrics {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	m := DetMetrics{N: n}
	if n == 0 {
		m.Band = DetExactBit
		m.ExactPct = 100
		return m
	}
	var sumAbs float64
	for i := 0; i < n; i++ {
		ai, bi := a[i], b[i]
		aNaN, bNaN := math.IsNaN(float64(ai)), math.IsNaN(float64(bi))
		if aNaN || bNaN {
			if aNaN != bNaN {
				m.HasNaNMismatch = true
			} else if aNaN && bNaN {
				m.ExactBits++
			}
			continue
		}
		if math.Float32bits(ai) == math.Float32bits(bi) {
			m.ExactBits++
		}
		d := math.Abs(float64(ai) - float64(bi))
		sumAbs += d
		if d > m.MaxAbs {
			m.MaxAbs = d
		}
		den := math.Abs(float64(ai))
		if math.Abs(float64(bi)) > den {
			den = math.Abs(float64(bi))
		}
		rel := d
		if den > 0 {
			rel = d / den
		}
		if rel > m.MaxRel {
			m.MaxRel = rel
		}
	}
	m.MeanAbs = sumAbs / float64(n)
	m.ExactPct = 100.0 * float64(m.ExactBits) / float64(n)
	m.Band = ClassifySpectrum(m)
	return m
}

// ClassifySpectrum picks the highest band the metrics satisfy.
func ClassifySpectrum(m DetMetrics) DetBand {
	if m.HasNaNMismatch {
		return DetBroken
	}
	if m.N > 0 && m.ExactBits == m.N {
		return DetExactBit
	}
	// Use the worse of abs / rel against fixed thresholds.
	score := m.MaxAbs
	if m.MaxRel > score {
		score = m.MaxRel
	}
	switch {
	case score <= DetTolFwd:
		return DetStrict
	case score <= 1e-3:
		return DetIndustry
	case score <= 1e-2:
		return DetLoose
	case score <= 1e-1:
		return DetBallpark
	default:
		return DetBroken
	}
}

// FormatSpectrum is a one-line readout for suite logs.
func FormatSpectrum(m DetMetrics) string {
	return fmt.Sprintf("band=%s (%.0f%%) exact_bits=%.2f%% max|Δ|=%.3g max_rel=%.3g",
		m.Band, m.Band.Percent(), m.ExactPct, m.MaxAbs, m.MaxRel)
}

// SpeedPctDiff returns how much faster ref is vs other as a percent:
// positive ⇒ other slower than ref; negative ⇒ other faster.
// pct = (other - ref) / ref * 100
func SpeedPctDiff(ref, other float64) float64 {
	if ref <= 0 {
		return 0
	}
	return (other - ref) / ref * 100
}

package evolution

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/dna"
	"github.com/openfluke/welvet/evolution"
	"github.com/openfluke/welvet/quant"
)

// MatrixFormatNoneAllDTypes — clone+splice every layer × FormatNone × all dtypes.
func MatrixFormatNoneAllDTypes() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes)
	fmt.Printf("\n  Evolution matrix — FormatNone × %d dtypes × %d layers (%d cells)\n",
		len(core.AllDTypes), len(kinds), total)
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			status, note := evoCell(k, dt, quant.FormatNone)
			rec(k.Name, dt.String(), "none", status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			default:
				failN++
			}
		}
	}
	fmt.Printf("  %s\n", polyops.Summary(okN, gapN, failN, total))
	if failN > 0 {
		return fmt.Errorf("evolution FormatNone matrix: %d FAIL", failN)
	}
	return nil
}

// MatrixAllQuantsFloat32 — clone+splice every layer × all quants × Float32.
func MatrixAllQuantsFloat32() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(quant.AllFormats)
	fmt.Printf("\n  Evolution matrix — %d quants × Float32 × %d layers (%d cells)\n",
		len(quant.AllFormats), len(kinds), total)
	for _, k := range kinds {
		for _, f := range quant.AllFormats {
			status, note := evoCell(k, core.DTypeFloat32, f)
			rec(k.Name, "f32", f.String(), status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			default:
				failN++
			}
		}
	}
	fmt.Printf("  %s\n", polyops.Summary(okN, gapN, failN, total))
	if failN > 0 {
		return fmt.Errorf("evolution quant matrix: %d FAIL", failN)
	}
	return nil
}

// FullMatrixCensus — layers × dtypes × quants clone+splice.
func FullMatrixCensus() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes) * len(quant.AllFormats)
	fmt.Printf("\n  Evolution FULL census — %d layers × %d dtypes × %d quants (%d cells)\n",
		len(kinds), len(core.AllDTypes), len(quant.AllFormats), total)
	var failNotes []string
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			for _, f := range quant.AllFormats {
				status, note := evoCell(k, dt, f)
				rec(k.Name, dt.String(), f.String(), status, note)
				switch status {
				case "OK":
					okN++
				case "GAP":
					gapN++
				default:
					failN++
					if len(failNotes) < 8 {
						failNotes = append(failNotes, fmt.Sprintf("%s/%s/%s: %s", k.Name, dt, f, note))
					}
				}
			}
		}
	}
	fmt.Printf("  %s\n", polyops.Summary(okN, gapN, failN, total))
	if failN > 0 {
		return fmt.Errorf("evolution full census: %d FAIL — %s", failN, strings.Join(failNotes, " | "))
	}
	return nil
}

func evoCell(k polyops.Kind, dt core.DType, format quant.Format) (status, note string) {
	a, err := k.Make(dt, format)
	if err != nil {
		return "GAP", trim(err.Error())
	}
	clone, err := evolution.CloneGrid(a)
	if err != nil {
		return "FAIL", "clone: " + trim(err.Error())
	}
	cmp := dna.CompareNetworks(dna.ExtractDNA(a), dna.ExtractDNA(clone))
	// FormatNone must round-trip bit-faithfully. Lossy packs (ternary/IQ/…)
	// re-encode via FlattenF32→Pack — DNA drift is expected; still require
	// clone+splice to succeed and preserve format on stores when present.
	if format == quant.FormatNone && cmp.OverallOverlap < 0.999 {
		return "FAIL", fmt.Sprintf("clone overlap=%v", cmp.OverallOverlap)
	}
	if err := formatsMatch(a, clone); err != nil {
		return "FAIL", err.Error()
	}
	b, err := perturb(k, dt, format)
	if err != nil {
		return "GAP", "perturb: " + trim(err.Error())
	}
	child, err := evolution.SpliceDNA(a, b, evolution.DefaultSpliceConfig())
	if err != nil {
		return "FAIL", "splice: " + trim(err.Error())
	}
	if child == nil {
		return "FAIL", "nil child"
	}
	if err := formatsMatch(a, child); err != nil {
		return "FAIL", "child " + err.Error()
	}
	if format != quant.FormatNone && cmp.OverallOverlap < 0.999 {
		return "OK", fmt.Sprintf("lossy-clone overlap=%.4f", cmp.OverallOverlap)
	}
	return "OK", ""
}

func formatsMatch(a, b *architecture.Grid) error {
	ca, cb := a.At(0, 0, 0, 0), b.At(0, 0, 0, 0)
	if ca == nil || cb == nil {
		return fmt.Errorf("nil cell")
	}
	sa, sb := dna.CollectStores(ca.Op), dna.CollectStores(cb.Op)
	if len(sa) != len(sb) {
		return fmt.Errorf("store count %d != %d", len(sa), len(sb))
	}
	for i := range sa {
		if sa[i] == nil || sb[i] == nil {
			continue
		}
		if sa[i].Format != sb[i].Format {
			return fmt.Errorf("format %s != %s", sa[i].Format, sb[i].Format)
		}
		if sa[i].DType != sb[i].DType {
			return fmt.Errorf("dtype %s != %s", sa[i].DType, sb[i].DType)
		}
		if sa[i].Rows != sb[i].Rows || sa[i].Cols != sb[i].Cols {
			return fmt.Errorf("shape mismatch")
		}
	}
	return nil
}

func perturb(k polyops.Kind, dt core.DType, format quant.Format) (*architecture.Grid, error) {
	g, err := k.Make(dt, format)
	if err != nil {
		return nil, err
	}
	for _, s := range dna.CollectStores(g.At(0, 0, 0, 0).Op) {
		if s == nil {
			continue
		}
		w, err := s.FlattenF32()
		if err != nil {
			return nil, err
		}
		for i := range w {
			w[i] += 0.07
		}
		if err := s.SetFromF32(w); err != nil {
			return nil, err
		}
	}
	return g, nil
}

func trim(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
}

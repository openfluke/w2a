package dna

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/quant"
)

// MatrixFormatNoneAllDTypes — every layer × FormatNone × all 34 dtypes.
func MatrixFormatNoneAllDTypes() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes)
	fmt.Printf("\n  DNA matrix — FormatNone × %d dtypes × %d layers (%d cells)\n",
		len(core.AllDTypes), len(kinds), total)
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			status, note := dnaCell(k, dt, quant.FormatNone)
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
		return fmt.Errorf("dna FormatNone matrix: %d FAIL", failN)
	}
	return nil
}

// MatrixAllQuantsFloat32 — every layer × all quants × Float32.
func MatrixAllQuantsFloat32() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(quant.AllFormats)
	fmt.Printf("\n  DNA matrix — %d quants × Float32 × %d layers (%d cells)\n",
		len(quant.AllFormats), len(kinds), total)
	for _, k := range kinds {
		for _, f := range quant.AllFormats {
			status, note := dnaCell(k, core.DTypeFloat32, f)
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
		return fmt.Errorf("dna quant matrix: %d FAIL", failN)
	}
	return nil
}

// FullMatrixCensus — layers × all dtypes × all quants (GAP ok; FAIL fails the case).
func FullMatrixCensus() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes) * len(quant.AllFormats)
	fmt.Printf("\n  DNA FULL census — %d layers × %d dtypes × %d quants (%d cells)\n",
		len(kinds), len(core.AllDTypes), len(quant.AllFormats), total)
	var failNotes []string
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			for _, f := range quant.AllFormats {
				status, note := dnaCell(k, dt, f)
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
		return fmt.Errorf("dna full census: %d FAIL — %s", failN, strings.Join(failNotes, " | "))
	}
	return nil
}

func dnaCell(k polyops.Kind, dt core.DType, format quant.Format) (status, note string) {
	g, err := k.Make(dt, format)
	if err != nil {
		return "GAP", trim(err.Error())
	}
	sig := dna.ExtractDNA(g)
	if len(sig) != 1 {
		return "FAIL", fmt.Sprintf("want 1 sig got %d", len(sig))
	}
	cmp := dna.CompareNetworks(sig, sig)
	if cmp.OverallOverlap < 0.999 {
		return "FAIL", fmt.Sprintf("self-overlap=%v", cmp.OverallOverlap)
	}
	return "OK", ""
}

func trim(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
}

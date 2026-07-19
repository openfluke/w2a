package tween

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/training"
)

// MatrixFormatNoneAllDTypes — StepTween every layer × FormatNone × all dtypes.
func MatrixFormatNoneAllDTypes() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes)
	fmt.Printf("\n  Tween matrix — FormatNone × %d dtypes × %d layers (%d cells)\n",
		len(core.AllDTypes), len(kinds), total)
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			status, note := tweenCell(k, dt, quant.FormatNone)
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
		return fmt.Errorf("tween FormatNone matrix: %d FAIL", failN)
	}
	return nil
}

// MatrixAllQuantsFloat32 — StepTween every layer × all quants × Float32.
func MatrixAllQuantsFloat32() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(quant.AllFormats)
	fmt.Printf("\n  Tween matrix — %d quants × Float32 × %d layers (%d cells)\n",
		len(quant.AllFormats), len(kinds), total)
	for _, k := range kinds {
		for _, f := range quant.AllFormats {
			status, note := tweenCell(k, core.DTypeFloat32, f)
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
		return fmt.Errorf("tween quant matrix: %d FAIL", failN)
	}
	return nil
}

// FullMatrixCensus — layers × dtypes × quants StepTween.
func FullMatrixCensus() error {
	kinds := polyops.AllKinds()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes) * len(quant.AllFormats)
	fmt.Printf("\n  Tween FULL census — %d layers × %d dtypes × %d quants (%d cells)\n",
		len(kinds), len(core.AllDTypes), len(quant.AllFormats), total)
	var failNotes []string
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			for _, f := range quant.AllFormats {
				status, note := tweenCell(k, dt, f)
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
		return fmt.Errorf("tween full census: %d FAIL — %s", failN, strings.Join(failNotes, " | "))
	}
	return nil
}

func tweenCell(k polyops.Kind, dt core.DType, format quant.Format) (status, note string) {
	g, err := k.Make(dt, format)
	if err != nil {
		return "GAP", trim(err.Error())
	}
	x, target := polyops.MakeIO(k, 1.15)
	if _, _, err := training.StepTween(g, x, target, 0.01); err != nil {
		// Construct OK but forward/update failed → real FAIL (not unimplemented Pack).
		return "FAIL", trim(err.Error())
	}
	return "OK", ""
}

func trim(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
}

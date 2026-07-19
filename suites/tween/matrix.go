package tween

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/runtime/training"
)

func backends() []core.Backend {
	return []core.Backend{core.BackendCPUTiled, core.BackendSIMD}
}

// MatrixFormatNoneAllDTypes — StepTween every layer × FormatNone × all dtypes × CPU/SIMD.
func MatrixFormatNoneAllDTypes() error {
	kinds := polyops.AllKinds()
	bes := backends()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes) * len(bes)
	fmt.Printf("\n  Tween matrix — FormatNone × %d dtypes × %d layers × %d backends (%d cells)\n",
		len(core.AllDTypes), len(kinds), len(bes), total)
	fmt.Printf("  SIMD=%v\n", simd.Enabled())
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			for _, be := range bes {
				status, note := tweenCell(k, dt, quant.FormatNone, be)
				recBackend(k.Name, dt.String(), "none", be.String(), status, note)
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
	}
	fmt.Printf("  %s\n", polyops.Summary(okN, gapN, failN, total))
	if failN > 0 {
		return fmt.Errorf("tween FormatNone matrix: %d FAIL", failN)
	}
	return nil
}

// MatrixAllQuantsFloat32 — StepTween every layer × all quants × Float32 × CPU/SIMD.
func MatrixAllQuantsFloat32() error {
	kinds := polyops.AllKinds()
	bes := backends()
	var okN, gapN, failN int
	total := len(kinds) * len(quant.AllFormats) * len(bes)
	fmt.Printf("\n  Tween matrix — %d quants × Float32 × %d layers × %d backends (%d cells)\n",
		len(quant.AllFormats), len(kinds), len(bes), total)
	fmt.Printf("  SIMD=%v\n", simd.Enabled())
	for _, k := range kinds {
		for _, f := range quant.AllFormats {
			for _, be := range bes {
				status, note := tweenCell(k, core.DTypeFloat32, f, be)
				recBackend(k.Name, "f32", f.String(), be.String(), status, note)
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
	}
	fmt.Printf("  %s\n", polyops.Summary(okN, gapN, failN, total))
	if failN > 0 {
		return fmt.Errorf("tween quant matrix: %d FAIL", failN)
	}
	return nil
}

// FullMatrixCensus — layers × dtypes × quants × CPU/SIMD StepTween.
func FullMatrixCensus() error {
	kinds := polyops.AllKinds()
	bes := backends()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes) * len(quant.AllFormats) * len(bes)
	fmt.Printf("\n  Tween FULL census — %d layers × %d dtypes × %d quants × %d backends (%d cells)\n",
		len(kinds), len(core.AllDTypes), len(quant.AllFormats), len(bes), total)
	fmt.Printf("  SIMD=%v\n", simd.Enabled())
	var failNotes []string
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			for _, f := range quant.AllFormats {
				for _, be := range bes {
					status, note := tweenCell(k, dt, f, be)
					recBackend(k.Name, dt.String(), f.String(), be.String(), status, note)
					switch status {
					case "OK":
						okN++
					case "GAP":
						gapN++
					default:
						failN++
						if len(failNotes) < 8 {
							failNotes = append(failNotes, fmt.Sprintf("%s/%s/%s/%s: %s", k.Name, dt, f, be, note))
						}
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

// MatrixSIMDSmoke — Dense/SwiGLU chain-rule + layerwise on BackendSIMD (DotTile/Saxpy).
func MatrixSIMDSmoke() error {
	if !simd.Enabled() {
		fmt.Printf("(SIMD off — GAP) ")
		recBackend("dense", "f32", "none", "simd", "GAP", "simd off")
		return nil
	}
	kinds := []string{"dense", "swiglu", "mha", "rmsnorm"}
	var failN int
	for _, name := range kinds {
		var k polyops.Kind
		for _, kk := range polyops.AllKinds() {
			if kk.Name == name {
				k = kk
				break
			}
		}
		if k.Name == "" {
			continue
		}
		g, err := polyops.MakeBackend(k, core.DTypeFloat32, quant.FormatNone, core.BackendSIMD)
		if err != nil {
			recBackend(name, "f32", "none", "simd", "FAIL", err.Error())
			failN++
			continue
		}
		x, target := polyops.MakeIO(k, 1.2)
		if _, _, err := training.StepTween(g, x, target, 0.02); err != nil {
			recBackend(name, "f32", "none", "simd", "FAIL", "chain: "+err.Error())
			failN++
			continue
		}
		recBackend(name, "f32", "none", "simd", "OK", "chain-rule DotTile")
	}
	if failN > 0 {
		return fmt.Errorf("simd smoke: %d FAIL", failN)
	}
	fmt.Printf("(chain-rule BackendSIMD) ")
	return nil
}

func tweenCell(k polyops.Kind, dt core.DType, format quant.Format, be core.Backend) (status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return "GAP", "simd off"
	}
	g, err := polyops.MakeBackend(k, dt, format, be)
	if err != nil {
		return "GAP", trim(err.Error())
	}
	x, target := polyops.MakeIO(k, 1.15)
	if _, _, err := training.StepTween(g, x, target, 0.01); err != nil {
		return "FAIL", trim(err.Error())
	}
	return "OK", ""
}

func recBackend(op, dt, format, backend, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer: "tween", Op: op, DType: dt, Format: format, Backend: backend, Grid: "1x1x1x1", Status: status, Note: note,
	})
}

func trim(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
}

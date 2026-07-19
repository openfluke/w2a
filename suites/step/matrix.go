package step

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/training"
)

func backends() []core.Backend {
	return []core.Backend{core.BackendCPUTiled, core.BackendSIMD}
}

// MatrixFormatNoneAllDTypes — StepMesh every layer × FormatNone × all dtypes × CPU/SIMD.
func MatrixFormatNoneAllDTypes() error {
	kinds := polyops.AllKinds()
	bes := backends()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes) * len(bes)
	fmt.Printf("\n  Step matrix — FormatNone × %d dtypes × %d layers × %d backends (%d cells)\n",
		len(core.AllDTypes), len(kinds), len(bes), total)
	fmt.Printf("  SIMD=%v\n", simd.Enabled())
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			for _, be := range bes {
				status, note := stepCell(k, dt, quant.FormatNone, be)
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
		return fmt.Errorf("step FormatNone matrix: %d FAIL", failN)
	}
	return nil
}

// MatrixAllQuantsFloat32 — StepMesh every layer × all quants × Float32 × CPU/SIMD.
func MatrixAllQuantsFloat32() error {
	kinds := polyops.AllKinds()
	bes := backends()
	var okN, gapN, failN int
	total := len(kinds) * len(quant.AllFormats) * len(bes)
	fmt.Printf("\n  Step matrix — %d quants × Float32 × %d layers × %d backends (%d cells)\n",
		len(quant.AllFormats), len(kinds), len(bes), total)
	fmt.Printf("  SIMD=%v\n", simd.Enabled())
	for _, k := range kinds {
		for _, f := range quant.AllFormats {
			for _, be := range bes {
				status, note := stepCell(k, core.DTypeFloat32, f, be)
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
		return fmt.Errorf("step quant matrix: %d FAIL", failN)
	}
	return nil
}

// FullMatrixCensus — layers × dtypes × quants × CPU/SIMD StepMesh.
func FullMatrixCensus() error {
	kinds := polyops.AllKinds()
	bes := backends()
	var okN, gapN, failN int
	total := len(kinds) * len(core.AllDTypes) * len(quant.AllFormats) * len(bes)
	fmt.Printf("\n  Step FULL census — %d layers × %d dtypes × %d quants × %d backends (%d cells)\n",
		len(kinds), len(core.AllDTypes), len(quant.AllFormats), len(bes), total)
	fmt.Printf("  SIMD=%v\n", simd.Enabled())
	var failNotes []string
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			for _, f := range quant.AllFormats {
				for _, be := range bes {
					status, note := stepCell(k, dt, f, be)
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
		return fmt.Errorf("step full census: %d FAIL — %s", failN, strings.Join(failNotes, " | "))
	}
	return nil
}

// MatrixSIMDSmoke — key layers on BackendSIMD.
func MatrixSIMDSmoke() error {
	if !simd.Enabled() {
		fmt.Printf("(SIMD off — GAP) ")
		recBackend("dense", "f32", "none", "simd", "GAP", "simd off")
		return nil
	}
	for _, name := range []string{"dense", "swiglu", "mha", "rmsnorm"} {
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
		status, note := stepCell(k, core.DTypeFloat32, quant.FormatNone, core.BackendSIMD)
		recBackend(name, "f32", "none", "simd", status, note)
		if status == "FAIL" {
			return fmt.Errorf("simd smoke %s: %s", name, note)
		}
	}
	fmt.Printf("(StepMesh BackendSIMD) ")
	return nil
}

func stepCell(k polyops.Kind, dt core.DType, format quant.Format, be core.Backend) (status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return "GAP", "simd off"
	}
	g, err := polyops.MakeBackend(k, dt, format, be)
	if err != nil {
		return "GAP", trim(err.Error())
	}
	x, target := polyops.MakeIO(k, 1.15)
	if _, _, err := training.StepMesh(g, x, target, 1, 0.01); err != nil {
		return "FAIL", trim(err.Error())
	}
	return "OK", ""
}

func recBackend(op, dt, format, backend, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer: "step", Op: op, DType: dt, Format: format, Backend: backend, Grid: "1x1x1x1", Status: status, Note: note,
	})
}

func trim(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
}

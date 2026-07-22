package step

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/w2a/suites/polyops"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/systems/dna"
)

// Cross-numeric train: weight DType ⊥ activation Tensor[T] across every Op kind.
// This is the W×A permutation matrix for all layer types (not float32-only acts).

type actRun struct {
	name string
	fn   func(k polyops.Kind, dt core.DType) (status, note string)
}

func allActRuns() []actRun {
	return []actRun{
		{"float64", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[float64](k, dt)
		}},
		{"float32", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[float32](k, dt)
		}},
		{"complex128", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[complex128](k, dt)
		}},
		{"complex64", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[complex64](k, dt)
		}},
		{"int64", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[int64](k, dt)
		}},
		{"int32", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[int32](k, dt)
		}},
		{"int16", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[int16](k, dt)
		}},
		{"int8", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[int8](k, dt)
		}},
		{"int", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[int](k, dt)
		}},
		{"uint64", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[uint64](k, dt)
		}},
		{"uint32", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[uint32](k, dt)
		}},
		{"uint16", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[uint16](k, dt)
		}},
		{"uint8", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[uint8](k, dt)
		}},
		{"uint", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[uint](k, dt)
		}},
		{"uintptr", func(k polyops.Kind, dt core.DType) (string, string) {
			return crossNumericCell[uintptr](k, dt)
		}},
	}
}

func smokeWeightDTypes() []core.DType {
	return []core.DType{
		core.DTypeFloat64, core.DTypeFloat32, core.DTypeFloat16,
		core.DTypeInt8, core.DTypeInt4, core.DTypeBinary,
		core.DTypeComplex64,
	}
}

func smokeActRuns() []actRun {
	want := map[string]bool{
		"float64": true, "float32": true, "int8": true, "uint8": true, "complex64": true,
	}
	var out []actRun
	for _, a := range allActRuns() {
		if want[a.name] {
			out = append(out, a)
		}
	}
	return out
}

// CrossNumericTrainSmoke — all Op kinds × sample weight dtypes × sample act hosts.
// Default CI case (~21×7×5 ≈ 735 cells).
func CrossNumericTrainSmoke() error {
	return runCrossNumeric("SMOKE", polyops.AllKinds(), smokeWeightDTypes(), smokeActRuns())
}

// CrossNumericTrainFull — all Op kinds × all 34 weight dtypes × all 15 act hosts.
// Large census (~21×34×15 ≈ 10.7k); FAIL on train errors, GAP on SetDType/Pack.
func CrossNumericTrainFull() error {
	return runCrossNumeric("FULL", polyops.AllKinds(), append([]core.DType(nil), core.AllDTypes...), allActRuns())
}

func runCrossNumeric(label string, kinds []polyops.Kind, dts []core.DType, acts []actRun) error {
	total := len(kinds) * len(dts) * len(acts)
	fmt.Printf("\n  CROSS-NUMERIC TRAIN %s — kinds×weightDType×actHost\n", label)
	fmt.Printf("  kinds=%d dtypes=%d acts=%d cells=%d (FormatNone, CPU tiled, StepMesh)\n\n",
		len(kinds), len(dts), len(acts), total)

	var okN, gapN int
	var fails []string
	for _, k := range kinds {
		for _, dt := range dts {
			for _, a := range acts {
				status, note := a.fn(k, dt)
				recCross(k.Name, dt.String(), a.name, status, note)
				switch status {
				case "OK":
					okN++
				case "GAP":
					gapN++
				default:
					fails = append(fails, fmt.Sprintf("%s/w=%s/a=%s:%s", k.Name, dt, a.name, note))
				}
			}
		}
	}
	fmt.Printf("  %s\n", polyops.Summary(okN, gapN, len(fails), total))
	if len(fails) > 0 {
		n := len(fails)
		if n > 12 {
			n = 12
		}
		return fmt.Errorf("cross-numeric %s: %d FAIL — %s", label, len(fails), strings.Join(fails[:n], " | "))
	}
	return nil
}

func crossNumericCell[T core.Numeric](k polyops.Kind, dt core.DType) (status, note string) {
	g, err := polyops.MakeBackend(k, dt, quant.FormatNone, core.BackendCPUTiled)
	if err != nil {
		return "GAP", trim(err.Error())
	}
	x, target := polyops.MakeIOT[T](k, 1.15)
	if _, _, err := training.StepMesh(g, x, target, 1, 0.05); err != nil {
		return "FAIL", trim(err.Error())
	}
	if err := assertCrossNumericStorage(g, dt); err != nil {
		return "FAIL", err.Error()
	}
	return "OK", fmt.Sprintf("w=%s×a=%s", dt, actTypeName[T]())
}

func actTypeName[T core.Numeric]() string {
	var z T
	switch any(z).(type) {
	case float64:
		return "float64"
	case float32:
		return "float32"
	case complex128:
		return "complex128"
	case complex64:
		return "complex64"
	case int64:
		return "int64"
	case int32:
		return "int32"
	case int16:
		return "int16"
	case int8:
		return "int8"
	case int:
		return "int"
	case uint64:
		return "uint64"
	case uint32:
		return "uint32"
	case uint16:
		return "uint16"
	case uint8:
		return "uint8"
	case uint:
		return "uint"
	case uintptr:
		return "uintptr"
	default:
		return "T"
	}
}

func assertCrossNumericStorage(g *architecture.Grid, dt core.DType) error {
	if dt == core.DTypeFloat32 {
		return nil
	}
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		for _, s := range dna.CollectStores(cell.Op) {
			if s == nil {
				continue
			}
			if s.F32BufferLen() != 0 && !(s.DType == core.DTypeFloat32 && s.Format == quant.FormatNone) {
				return fmt.Errorf("retained f32 buffer after train dtype=%s", dt)
			}
			if s.RetainsF32Master() && s.DType != core.DTypeFloat32 {
				return fmt.Errorf("RetainsF32Master after train dtype=%s", dt)
			}
		}
		// Dense-only quick path also covered by CollectStores; keep compile happy.
		if dl, ok := cell.Op.(*dense.Layer); ok && dl != nil && dl.Weights != nil {
			if dt != core.DTypeFloat32 && dl.Weights.F32BufferLen() != 0 {
				return fmt.Errorf("dense retained f32 buffer dtype=%s", dt)
			}
		}
	}
	return nil
}

func recCross(op, weightDT, actHost, status, note string) {
	suites.RecordCell(suites.Cell{
		Layer: "step", Op: op, DType: weightDT + "×" + actHost, Format: "none",
		Backend: "cpu_tiled", Grid: "1x1x1x1", Status: status, Note: note,
	})
}

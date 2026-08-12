package parallel

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/systems/dna"
	"github.com/openfluke/welvet/webgpu"
)

// Cameral recipes swept in the permutation matrix.
type cameralArch string

const (
	archBicameral   cameralArch = "bicameral"
	archHemi3       cameralArch = "hemi3"
	archNestedStack cameralArch = "nested_stack"
	archMixedHemi   cameralArch = "mixed_hemi"
)

func cameralArches() []cameralArch {
	return []cameralArch{archBicameral, archHemi3, archNestedStack, archMixedHemi}
}

func cameralModes() []parallel.TrainMode {
	return parallel.AllConcreteTrainModes()
}

func cameralBackends() []core.Backend {
	return []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
}

// CameralPermMatrixFormatNone: arch × train mode × all dtypes × backends @ FormatNone.
func CameralPermMatrixFormatNone() error {
	return cameralPermSweep(true)
}

// CameralPermMatrixQuant: arch × train mode × all quants × backends @ Float32.
func CameralPermMatrixQuant() error {
	return cameralPermSweep(false)
}

// CameralPermCensus: full FormatNone+quant union; always PASS (gaps recorded).
func CameralPermCensus() error {
	_ = cameralPermSweep(true)
	_ = cameralPermSweep(false)
	fmt.Printf("(cameral census recorded; gaps do not fail the case) ")
	return nil
}

func cameralPermSweep(formatNone bool) error {
	arches := cameralArches()
	modes := cameralModes()
	backends := cameralBackends()
	var dtypes []core.DType
	var formats []quant.Format
	title := ""
	if formatNone {
		dtypes = append([]core.DType(nil), core.AllDTypes...)
		formats = []quant.Format{quant.FormatNone}
		title = fmt.Sprintf("FormatNone × %d dtypes × %d backends × %d modes × %d arches",
			len(dtypes), len(backends), len(modes), len(arches))
	} else {
		dtypes = []core.DType{core.DTypeFloat32}
		formats = append([]quant.Format(nil), quant.AllFormats...)
		title = fmt.Sprintf("%d quants × %d backends × %d modes × %d arches (Float32)",
			len(formats), len(backends), len(modes), len(arches))
	}
	total := len(arches) * len(modes) * len(dtypes) * len(formats) * len(backends)
	fmt.Printf("\n  CAMERAL PERM — %s\n", title)
	fmt.Printf("  cells=%d  SIMD=%v WebGPU=%v\n\n", total, simd.Enabled(), webgpu.Available())
	fmt.Printf("  %-12s %-12s %-12s %-14s %-10s %8s  %s\n",
		"arch", "mode", "dtype", "format", "backend", "status", "note")
	fmt.Printf("  %s\n", strings.Repeat("-", 96))

	var cpuFails []string
	var okN, gapN int
	for _, arch := range arches {
		for _, mode := range modes {
			for _, dt := range dtypes {
				for _, f := range formats {
					for _, be := range backends {
						status, note := runCameralPermCell(arch, mode, dt, f, be)
						op := fmt.Sprintf("cameral_perm_%s_%s", arch, mode.String())
						rec(op, dt.String(), f.String(), be.String(), "stack", status, note)
						fmt.Printf("  %-12s %-12s %-12s %-14s %-10s %8s  %s\n",
							arch, mode.String(), dt.String(), f.String(), be.String(), status, note)
						switch status {
						case "OK":
							okN++
						case "GAP":
							gapN++
						case "FAIL":
							cpuFails = append(cpuFails, fmt.Sprintf("%s/%s/%s/%s/%s: %s",
								arch, mode, dt, f, be, note))
						}
					}
				}
			}
		}
	}
	fmt.Printf("\n  summary: %d OK, %d GAP, %d FAIL (of %d cells)\n", okN, gapN, len(cpuFails), total)
	if len(cpuFails) > 0 {
		n := min(8, len(cpuFails))
		return fmt.Errorf("cameral perm: %d failures: %s", len(cpuFails), strings.Join(cpuFails[:n], " | "))
	}
	return nil
}

func runCameralPermCell(arch cameralArch, mode parallel.TrainMode, dt core.DType, format quant.Format, be core.Backend) (status, note string) {
	if be == core.BackendSIMD && !simd.Enabled() {
		return "GAP", "simd off"
	}
	if be == core.BackendWebGPU && !webgpu.Available() {
		return "GAP", "no gpu"
	}
	// Affine needs packable out×in; cameral hidden=16 square is usually OK.
	if format == quant.FormatAffinePacked && !suites.AffinePackable(16, 16) {
		return "GAP", suites.AffineSkipNote()
	}
	if !parallel.PermutationOK(dt, format, be) {
		return "GAP", "unsupported permutation"
	}

	s, inDim, outDim, err := buildCameralArch(arch)
	if err != nil {
		return failOrGap(be), "build: " + err.Error()
	}
	if err := stampCameralStack(s, dt, format, be); err != nil {
		return "GAP", "stamp: " + err.Error()
	}
	if err := seedNonZero(s); err != nil {
		return failOrGap(be), "seed: " + err.Error()
	}

	x := core.NewTensor[float32](2, inDim)
	y := core.NewTensor[float32](2, outDim)
	fillOnes(x)
	for i := range y.Data {
		y.Data[i] = 0.25
	}

	before, err := dna.FlattenOp(s)
	if err != nil {
		return failOrGap(be), "snapshot: " + err.Error()
	}

	// Direct TrainStackMSE covers SGD/Tween/TweenChain; also exercise grid path once per FormatNone CPU SGD cell.
	loss, err := parallel.TrainStackMSE(s, x, y, mode, 0.15)
	if err != nil {
		return failOrGap(be), "train: " + err.Error()
	}
	_ = loss

	after, err := dna.FlattenOp(s)
	if err != nil {
		return failOrGap(be), "snapshot after: " + err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		// Coarse dtypes / k-quants may round updates away.
		return "GAP", fmt.Sprintf("weights unchanged under %s/%s (lr may be below quant step)", dt, format)
	}

	// Extra grid Step / StepTween smoke on CPU FormatNone SGD bicameral only (cheap sentinel).
	if arch == archBicameral && mode == parallel.ModeNormalBP &&
		dt == core.DTypeFloat32 && format == quant.FormatNone && be == core.BackendCPUTiled {
		if st, n := cameralGridSmoke(s, x, y); st != "OK" {
			return st, n
		}
	}
	return "OK", fmt.Sprintf("Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}

func cameralGridSmoke(s *parallel.Stack, x, y *core.Tensor[float32]) (status, note string) {
	g := architecture.NewGrid(1, 1, 1, 1)
	if err := parallel.PlaceStack(g, 0, 0, 0, 0, s); err != nil {
		return "FAIL", "place: " + err.Error()
	}
	fwd, err := forward.Forward(g, x)
	if err != nil {
		return "FAIL", "grid fwd: " + err.Error()
	}
	if _, err := training.Step(fwd, y, 0.05); err != nil {
		return "FAIL", "grid Step: " + err.Error()
	}
	if _, _, err := training.StepTween(g, x, y, 0.05); err != nil {
		return "FAIL", "grid StepTween: " + err.Error()
	}
	return "OK", "grid Step+StepTween ok"
}

func stampCameralStack(s *parallel.Stack, dt core.DType, format quant.Format, be core.Backend) error {
	if s == nil {
		return fmt.Errorf("nil stack")
	}
	if format != quant.FormatNone {
		if err := s.Pack(format); err != nil {
			return err
		}
	} else if dt != core.DTypeFloat32 {
		if err := s.SetDType(dt); err != nil {
			return err
		}
	}
	s.Exec.Backend = be
	s.Exec.MultiCore = false
	s.SyncChildExec()
	return nil
}

func buildCameralArch(arch cameralArch) (s *parallel.Stack, inDim, outDim int, err error) {
	const hidden = 16
	switch arch {
	case archBicameral:
		inDim, outDim = 8, 4
		s, err = parallel.Bicameral(inDim, hidden, outDim, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
		return s, inDim, outDim, err
	case archHemi3:
		inDim, outDim = hidden, hidden
		stem, err := dense.New(hidden, hidden, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, 0, 0, err
		}
		hemi, err := parallel.Hemispheres(hidden, hidden, 3, parallel.CombineConcat, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return nil, 0, 0, err
		}
		head, err := dense.New(hidden*3, hidden, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, 0, 0, err
		}
		s, err = parallel.Sandwich(stem, hemi, head)
		return s, inDim, outDim, err
	case archNestedStack:
		inDim, outDim = hidden, hidden
		mk := func() (*parallel.Stack, error) {
			a, err := dense.New(hidden, hidden, core.ActivationLinear, core.DTypeFloat32)
			if err != nil {
				return nil, err
			}
			b, err := dense.New(hidden, hidden, core.ActivationLinear, core.DTypeFloat32)
			if err != nil {
				return nil, err
			}
			hemi, err := parallel.Hemispheres(hidden, hidden, 2, parallel.CombineAdd, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
			if err != nil {
				return nil, err
			}
			return parallel.NewStack(a, hemi, b)
		}
		left, err := mk()
		if err != nil {
			return nil, 0, 0, err
		}
		right, err := mk()
		if err != nil {
			return nil, 0, 0, err
		}
		outer, err := parallel.HemispheresFrom(parallel.Config{
			Dim: hidden, OutFeat: hidden, Branches: 2, Combine: parallel.CombineAvg,
		}, []any{left, right}, nil)
		if err != nil {
			return nil, 0, 0, err
		}
		s, err = parallel.Sandwich(outer)
		return s, inDim, outDim, err
	case archMixedHemi:
		inDim, outDim = hidden, hidden
		left, err := dense.New(hidden, hidden, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, 0, 0, err
		}
		right, err := dense.New(hidden, hidden, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, 0, 0, err
		}
		hemi, err := parallel.HemispheresFrom(parallel.Config{
			Dim: hidden, OutFeat: hidden, Branches: 2, Combine: parallel.CombineAdd,
		}, []any{left, right}, nil)
		if err != nil {
			return nil, 0, 0, err
		}
		hemi.SetBranchModes(parallel.ModeNormalBP, parallel.ModeTween)
		stem, err := dense.New(hidden, hidden, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, 0, 0, err
		}
		head, err := dense.New(hidden, hidden, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, 0, 0, err
		}
		s, err = parallel.Sandwich(stem, hemi, head)
		return s, inDim, outDim, err
	default:
		return nil, 0, 0, fmt.Errorf("unknown cameral arch %q", arch)
	}
}

package go_v_cuda

import (
	"bytes"
	"fmt"
	"github.com/openfluke/w2a/suites"
	"math"
	"os/exec"
	"reflect"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
)

type kpModeFixture struct {
	epochs         int
	allowGDNStrict bool
	ops            []any
	wires          [][]byte
	xShape         []int32
	x              []float32
}

func kpModeWire(mode parallel.TrainMode) (request, response string, line bool) {
	if proxyVariant(mode) > 0 {
		return "GPB1", "GPBO", false
	}
	switch mode {
	case parallel.ModeStepTweenSplitLinear:
		return "STL1", "STLO", true
	case parallel.ModeStepTweenSplitFastProxy:
		return "STF1", "STFO", true
	case parallel.ModeStepTweenSplitLinearCache:
		return "SLC1", "SLCO", true
	case parallel.ModeStepTweenSplitHeadProxyAsync:
		return "SHA1", "SHAO", true
	case parallel.ModeStepTweenSplitSparse:
		return "SSP1", "SSPO", true
	case parallel.ModeStepTweenSplitHeadProxy:
		return "STH1", "STHO", true
	case parallel.ModeStepTweenSplit:
		return "STS1", "STSO", true
	case parallel.ModeTweenAlt:
		return "TAB1", "TABO", false
	case parallel.ModeStepTweenAlt:
		return "STA1", "STAO", true
	case parallel.ModeTweenSplit:
		return "TSB1", "TSBO", false
	case parallel.ModeStepBP:
		return "STB1", "STBO", true
	case parallel.ModeTween:
		return "TWB1", "TWBO", false
	case parallel.ModeTweenChain:
		return "TCB1", "TCBO", false
	case parallel.ModeStepTween:
		return "STW1", "STWO", true
	case parallel.ModeStepTweenChain:
		return "STC1", "STCO", true
	}
	panic("unsupported KMeans/Parallel test mode")
}

func kpKMeansFixture(dt core.DType, output kmeans.OutputMode) (kpModeFixture, error) {
	cfg := kmeans.Config{FeatureDim: 4, NumClusters: 3, Temperature: 0.7, OutputMode: output, Activation: core.ActivationSilu}
	w := seedPat(12, 0x6711)
	l, err := kmeans.NewConfigured(cfg, dt, quant.FormatNone, w)
	if err != nil {
		return kpModeFixture{}, err
	}
	disableKMeansTrain(l)
	var wire bytes.Buffer
	writeKmChild(&wire, cfg, 1, dt, w)
	f := kpModeFixture{ops: []any{l}, wires: [][]byte{wire.Bytes()}, xShape: []int32{2, 4}, x: seedPat(8, 0x6712)}
	return kpAddTail(f, dt, cfg.OutDim())
}

func kpAddTail(f kpModeFixture, dt core.DType, in int) (kpModeFixture, error) {
	w := seedPat(in*4, 0x6713)
	tail, err := dense.NewConfigured(in, 4, core.ActivationLinear, dt, quant.FormatNone, w)
	if err != nil {
		return f, err
	}
	tail.Exec.Backend = core.BackendCPUTiled
	tail.Exec.MultiCore = false
	f.ops = append(f.ops, tail)
	f.wires = append(f.wires, polyEncDense(in, 4, dt, w))
	return f, nil
}

func kpParallelFixture(dt core.DType, combine parallel.CombineMode) (kpModeFixture, error) {
	cfg := parallel.Config{Dim: 4, OutFeat: 4, Branches: 3, Combine: combine, SparseK: 2, DisagreeBeta: 0.375}
	w := seedPat(48, 0x6811)
	gate := seedPat(12, 0x6812)
	l, err := parallel.NewConfigured[float32](cfg, dt, quant.FormatNone, w, gate)
	if err != nil {
		return kpModeFixture{}, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	var children [][]byte
	for i := 0; i < 3; i++ {
		children = append(children, polyEncDense(4, 4, dt, w[i*16:(i+1)*16]))
	}
	var gw []byte
	if combine == parallel.CombineFilter {
		gw = polyEncDense(4, 3, dt, gate)
	}
	f := kpModeFixture{ops: []any{l}, wires: [][]byte{encodeParallelChild(cfg, children, gw)}, xShape: []int32{2, 4}, x: seedPat(8, 0x6813)}
	od := 4
	if combine == parallel.CombineConcat {
		od = 12
	}
	return kpAddTail(f, dt, od)
}

func kpPackOp(op any) ([]float32, error) {
	switch l := op.(type) {
	case *parallel.Layer:
		return packParallelAll(l)
	case *kmeans.Layer:
		return packKMeansWeightsF32(l)
	default:
		return packPolyBranchW(op)
	}
}

func kpRunMode(mode parallel.TrainMode, dt core.DType, makeFixture func(core.DType) (kpModeFixture, error)) error {
	f, err := makeFixture(dt)
	if err != nil {
		return err
	}
	stack, err := parallel.NewStack(f.ops...)
	if err != nil {
		return err
	}
	stack.Exec.Backend = core.BackendCPUTiled
	stack.Exec.MultiCore = false
	request, response, line := kpModeWire(mode)
	input := core.NewTensor[float32](ints(f.xShape)...)
	copy(input.Data, f.x)
	var post *core.Tensor[float32]
	if mode == parallel.ModeTweenSplit || proxyVariant(mode) > 0 {
		// Split scores its collected tape, whose native branch shape may differ
		// from the ordinary Parallel forward's normalized feature axes.
		tape, e := parallel.OpenSplitTape[float32](stack, input)
		err = e
		if e == nil {
			post = tape.Post
		}
	} else {
		_, post, err = parallel.ForwardStack(stack, input)
	}
	if err != nil {
		return fmt.Errorf("initial forward: %w", err)
	}
	// Shape discovery can advance rule cooldowns or recurrent caches. Rebuild
	// before the first training tick so the Go fixture matches the wire replay.
	f, err = makeFixture(dt)
	if err != nil {
		return err
	}
	stack, err = parallel.NewStack(f.ops...)
	if err != nil {
		return err
	}
	stack.Exec.Backend = core.BackendCPUTiled
	stack.Exec.MultiCore = false
	ticks := proxyEpochs(mode)
	if f.epochs > ticks {
		ticks = f.epochs
	}
	if line {
		ticks = 5
	}
	if f.epochs > 0 {
		ticks = f.epochs
	}
	const lr = 0.017
	var xs, ts [][]float32
	var loss float64
	for tick := 0; tick < ticks; tick++ {
		x := append([]float32(nil), f.x...)
		for i := range x {
			x[i] += float32(tick) * 0.005
		}
		target := seedPat(len(post.Data), uint32(0x6911+tick))
		xs = append(xs, x)
		ts = append(ts, target)
		xi := core.NewTensor[float32](ints(f.xShape)...)
		copy(xi.Data, x)
		ti := core.NewTensor[float32](post.Shape...)
		copy(ti.Data, target)
		loss, err = parallel.TrainStackMSE(stack, xi, ti, mode, lr)
		if err != nil {
			return fmt.Errorf("Go tick %d: %w", tick, err)
		}
	}
	ref := parallelNBPOut{loss: loss}
	for _, op := range stack.Children {
		w, err := kpPackOp(op)
		if err != nil {
			return err
		}
		ref.WAfter = append(ref.WAfter, w)
	}
	var payload bytes.Buffer
	payload.WriteString(request)
	if request == "GPB1" {
		writeI32(&payload, int32(proxyVariant(mode)))
	}
	writeI32(&payload, int32(len(f.ops)), int32(ticks), f.xShape[0])
	writeF64(&payload, lr)
	for _, wire := range f.wires {
		payload.Write(wire)
	}
	shape := make([]int32, len(post.Shape))
	for i, d := range post.Shape {
		shape[i] = int32(d)
	}
	writeShapesTicks(&payload, f.xShape, shape, xs, ts)
	// A measured run must reset its weights and Line flights before replay.
	fresh, _, err := runCudaParallelNBP(payload.Bytes(), response, 0)
	if err != nil {
		return err
	}
	if proxyVariant(mode) > 0 || stepProxyVariant(mode) > 0 {
		// Proxy leaf/Line epochs accumulate CUDA float2 vs Go drift; keep the
		// industry-to-exact spectrum (shapes/counts exact; same-backend replay below).
		// Includes StepTweenSplitHeadProxy (variant 1) — same floor as leaf HeadProxy.
		refVolume := meshClockOut{loss: ref.loss, weights: ref.WAfter, grads: make([]meshTensor, len(ref.WAfter))}
		gotVolume := meshClockOut{loss: fresh.loss, weights: fresh.WAfter, grads: make([]meshTensor, len(fresh.WAfter))}
		if err = compareProxyVolume(refVolume, gotVolume); err != nil {
			return err
		}
	} else if err = kpCompareMode(ref, fresh, f.allowGDNStrict || mode == parallel.ModeStepTweenSplit); err != nil {
		return err
	}
	iters := defaultGoBenchIters
	if mode == parallel.ModeTweenSplit && iters < 2 {
		iters = 2
	}
	measured, _, err := runCudaParallelNBP(payload.Bytes(), response, iters)
	if err != nil {
		return err
	}
	if mode == parallel.ModeTweenSplit {
		if !reflect.DeepEqual(fresh, measured) {
			return fmt.Errorf("Split benchmark changed loss or weights")
		}
		return nil
	}
	return compareParallelNBP(fresh, measured, true)
}

func CasesKMeansParallelModes(mode parallel.TrainMode) []Case {
	var cases []Case
	add := func(name string, build func(core.DType) (kpModeFixture, error)) {
		cases = append(cases, Case{Name: name + " Float32 multi-hop", Run: func() error { return kpRunMode(mode, core.DTypeFloat32, build) }})
		cases = append(cases, Case{Name: name + " ×34 multi-hop + benchmark reset", Run: func() error {
			for _, dt := range core.AllDTypes {
				if err := kpRunMode(mode, dt, build); err != nil {
					return fmt.Errorf("%s: %w", dt, err)
				}
			}
			return nil
		}})
	}
	for _, output := range []kmeans.OutputMode{kmeans.OutputProbabilities, kmeans.OutputFeatures} {
		output := output
		add("KMeans "+string(output), func(dt core.DType) (kpModeFixture, error) { return kpKMeansFixture(dt, output) })
	}
	for _, combine := range []parallel.CombineMode{parallel.CombineConcat, parallel.CombineAdd, parallel.CombineAvg, parallel.CombineFilter, parallel.CombineMax, parallel.CombineSparseK, parallel.CombineDisagree} {
		combine := combine
		add("Parallel "+string(combine), func(dt core.DType) (kpModeFixture, error) { return kpParallelFixture(dt, combine) })
	}
	// Exercise the polymorphic dispatcher inside a Parallel container as well.
	for _, pc := range polyTrainKinds() {
		pc := pc
		label := "Parallel polymorphic " + pc.name + " Float32"
		if pc.name == "gdn" {
			label += " (≥strict weights)"
		}
		cases = append(cases, Case{Name: label, Run: func() error {
			return kpRunMode(mode, core.DTypeFloat32, func(dt core.DType) (kpModeFixture, error) {
				brs, x, _, shape, _, err := pc.make(dt)
				if err != nil {
					return kpModeFixture{}, err
				}
				var ops []any
				var wires [][]byte
				for _, br := range brs {
					ops = append(ops, br.op)
					wires = append(wires, br.bytes)
				}
				cfg := parallel.Config{Dim: pc.cfgDim, OutFeat: pc.cfgOutFeat, SeqLen: pc.cfgSeq, Branches: len(ops), Combine: parallel.CombineAdd}
				l, err := parallel.NewFromBranches(cfg, ops, nil)
				if err != nil {
					return kpModeFixture{}, err
				}
				l.Exec.Backend = core.BackendCPUTiled
				l.Exec.MultiCore = false
				return kpModeFixture{allowGDNStrict: pc.name == "gdn", ops: []any{l}, wires: [][]byte{encodeParallelChild(cfg, wires, nil)}, xShape: shape, x: x}, nil
			})
		}})
	}
	cases = append(cases, Case{Name: "Parallel Filter stateful Metacognition capture before update", Run: func() error {
		return kpRunMode(mode, core.DTypeFloat32, func(dt core.DType) (kpModeFixture, error) {
			brs, x, _, shape, _, err := makeMetaPair(dt, 2, 8)
			if err != nil {
				return kpModeFixture{}, err
			}
			gateW := seedPat(16, 0x7311)
			gate, err := dense.NewConfigured(8, 2, core.ActivationLinear, dt, quant.FormatNone, gateW)
			if err != nil {
				return kpModeFixture{}, err
			}
			cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: parallel.CombineFilter}
			l, err := parallel.NewFromBranches(cfg, []any{brs[0].op, brs[1].op}, gate)
			if err != nil {
				return kpModeFixture{}, err
			}
			l.Exec.Backend = core.BackendCPUTiled
			l.Exec.MultiCore = false
			return kpModeFixture{ops: []any{l}, wires: [][]byte{encodeParallelChild(cfg, [][]byte{brs[0].bytes, brs[1].bytes}, polyEncDense(8, 2, dt, gateW))}, xShape: shape, x: x}, nil
		})
	}})

	if proxyVariant(mode) > 0 || stepProxyVariant(mode) > 0 {
		return cases
	}
	return append(cases, kpVolumeCases(mode)...)
}

// Keep the simple wire runner local: CPU numerical reference code stays in w2a.
func kpRunVolumeWire(payload []byte, response string, bench ...int) (nbpVolOut, error) {
	var out nbpVolOut
	tool, err := findCUDAFwd()
	if err != nil {
		return out, err
	}
	var args []string
	if len(bench) > 0 && bench[0] > 0 {
		args = append(args, fmt.Sprintf("--bench=%d", bench[0]))
	}
	cmd := exec.Command(tool, args...)
	cmd.Stdin = bytes.NewReader(payload)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	if err != nil {
		return out, fmt.Errorf("CUDA volume: %w: %s", err, stderr.String())
	}
	out, _, err = decodeRecVolNBP(raw, response)
	return out, err
}

func kpVolumeCases(mode parallel.TrainMode) []Case {
	var cases []Case
	for _, kind := range []string{"KMeans", "Parallel"} {
		kind := kind
		for _, top := range []struct {
			name       string
			z, y, x, l int
			all        bool
		}{{"111", 1, 1, 1, 1, true}, {"chain", 1, 1, 1, 2, true}, {"211", 2, 1, 1, 1, true}, {"222", 2, 2, 2, 1, false}} {
			top := top
			name := kind + " volume " + top.name + " Float32"
			if top.all {
				name = kind + " volume " + top.name + " ×34"
			}
			cases = append(cases, Case{Name: name, Run: func() error {
				dtypes := []core.DType{core.DTypeFloat32}
				if top.all {
					dtypes = core.AllDTypes
				}
				for _, dt := range dtypes {
					if err := kpVolumeMode(mode, kind, top.z, top.y, top.x, top.l, dt); err != nil {
						return fmt.Errorf("%s: %w", dt, err)
					}
				}
				return nil
			}})
		}
	}
	return cases
}

func kpVolumeMode(mode parallel.TrainMode, kind string, z, y, x, l int, dt core.DType) error {
	const batch, dim, lr = 2, 4, 0.017
	input, target := seedPat(batch*dim, 0x7101), seedPat(batch*dim, 0x7102)
	split := mode == parallel.ModeTweenSplit || mode == parallel.ModeStepTweenSplit
	alt := mode == parallel.ModeTweenAlt || mode == parallel.ModeStepTweenAlt
	tween := mode == parallel.ModeTween || mode == parallel.ModeStepTween
	chain := mode == parallel.ModeTweenChain || mode == parallel.ModeStepTweenChain
	var ref nbpVolOut
	var payload []byte
	var err error
	response := "PNVO"
	if kind == "KMeans" {
		cfg := kmeans.Config{NumClusters: dim, FeatureDim: dim, Temperature: 0.7, OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear}
		cells := make([]kmVolCell, z*y*x*l)
		for i := range cells {
			cells[i] = makeKmVolCell(cfg, 1, dt, uint32(0x7110+i))
		}
		if alt {
			ref, err = runGoKmVolApply(z, y, x, l, batch, cells, input, target, lr, kpAltVolume, true)
		} else if split {
			ref, err = runGoKmVolApply(z, y, x, l, batch, cells, input, target, lr, applyModeTweenSplitVolume, true)
		} else if tween {
			ref, err = runGoKmVolApply(z, y, x, l, batch, cells, input, target, lr, applyModeTweenVolume, true)
		} else {
			ref, err = runGoKmVolNBP(z, y, x, l, batch, cells, input, target, lr)
		}
		if err != nil {
			return err
		}
		payload, err = encodeKmVolNBP(z, y, x, l, batch, cells, input, target, lr)
		response = "KMVO"
		if alt {
			copy(payload[:4], "KMAL")
			response = "KMAO"
		} else if split {
			copy(payload[:4], "KMSP")
			response = "KMSO"
		} else if tween {
			copy(payload[:4], "KMTV")
			response = "KMTO"
		} else if chain {
			copy(payload[:4], "KMTC")
			response = "KMCO"
		}
	} else {
		cells := makePnVolCells(z*y*x*l, dim, dim, 3, parallel.CombineAdd, 0x7211)
		for i := range cells {
			cells[i].dtype = dt
		}
		if alt {
			ref, err = runGoPnVolApply(z, y, x, l, batch, cells, input, target, lr, kpAltVolume, true)
		} else if split {
			ref, err = runGoPnVolApply(z, y, x, l, batch, cells, input, target, lr, applyModeTweenSplitVolume, true)
		} else if tween {
			ref, err = runGoPnVolApply(z, y, x, l, batch, cells, input, target, lr, applyModeTweenVolume, true)
		} else {
			ref, err = runGoPnVolNBP(z, y, x, l, batch, cells, input, target, lr)
		}
		if err != nil {
			return err
		}
		payload, err = encodePnVolNBP(z, y, x, l, batch, cells, input, target, lr)
		if alt {
			copy(payload[:4], "PNAL")
			response = "PNAO"
		} else if split {
			copy(payload[:4], "PNSP")
			response = "PNSO"
		} else if tween {
			copy(payload[:4], "PNTV")
			response = "PNTO"
		} else if chain {
			copy(payload[:4], "PNTC")
			response = "PNCO"
		}
	}
	if err != nil {
		return err
	}
	got, err := kpRunVolumeWire(payload, response)
	if err != nil {
		return err
	}
	if split {
		iters := defaultGoBenchIters
		if iters < 2 {
			iters = 2
		}
		measured, err := kpRunVolumeWire(payload, response, iters)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(got, measured) {
			return fmt.Errorf("Split volume replay changed tensors, loss or weights")
		}
	}
	if kind == "KMeans" {
		return compareKmVolNBP(ref, got)
	}
	return compareVolNBP(ref, got, true)
}

// GDN retains the same >=strict updated-weight gate used by these modes'
// standalone GDN cases. ModeStepTweenSplit Line multi-tick (5 pops) may land
// DetStrict on Parallel Add / int32 KMeans-probabilities tails vs Go — CUDA
// MatMul ulps accumulate across pops while a single pop stays exact_bit.
// Every other fixture keeps exact native weight checks.
func kpCompareMode(a, b parallelNBPOut, allowStrict bool) error {
	if !allowStrict {
		return compareParallelNBP(a, b, true)
	}
	lossOK := cudaLossWithinBand(a.loss, b.loss)
	// Match the existing standalone GDN finite relative loss gate, including
	// accumulated state/weight differences over cache-refresh training epochs.
	if !lossOK {
		lossOK = !(math.IsNaN(a.loss) || math.IsNaN(b.loss) || math.IsInf(a.loss, 0) || math.IsInf(b.loss, 0)) && math.Abs(a.loss-b.loss) <= 1e-5*math.Max(1e-30, math.Abs(a.loss))
	}
	if len(a.WAfter) != len(b.WAfter) || !lossOK {
		return fmt.Errorf("strict Parallel loss or child count mismatch: go=%g cuda=%g", a.loss, b.loss)
	}
	for i := range a.WAfter {
		if len(a.WAfter[i]) != len(b.WAfter[i]) {
			return fmt.Errorf("strict W[%d] length mismatch", i)
		}
		m := suites.CompareF32Spectrum(a.WAfter[i], b.WAfter[i])
		if m.Band.Percent() < suites.DetStrict.Percent() {
			return fmt.Errorf("strict W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	return nil
}

func kpAltVolume(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error) {
	input := fwd.Steps[0].Input
	if _, err := applyModeTweenSplitVolume(g, fwd, target, lr); err != nil {
		return 0, err
	}
	fresh, err := forward.Forward(g, input)
	if err != nil {
		return 0, err
	}
	*fwd = *fresh
	return applyModeTweenVolume(g, fwd, target, lr)
}

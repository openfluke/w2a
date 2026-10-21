package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesStackTrain — NormalBP NestedStack-of-Dense Go↔C++ (leaf + volumetric).
func CasesStackTrain() []Case {
	return []Case{
		{Name: "Stack NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseStNBPLeafF32},
		{Name: "Stack NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseStNBPLeafAllDTypes},
		{Name: "Stack NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseStNBPVol1x1x1},
		{Name: "Stack NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseStNBPVolChain},
		{Name: "Stack NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseStNBPVol111AllDTypes},
		{Name: "Stack NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseStNBPVol222F32},
		{Name: "Stack NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseStNBPVol211AllDTypes},
	}
}

func RunStackTrain() error {
	var fails []string
	cs := CasesStackTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [stnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_stack_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_stack_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp stack_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type stNBPLeaf struct {
	batch     int
	seq       int // 1 → [B,D]
	dim       int
	nChildren int
	dtype     core.DType
	W         []float32
	x         []float32
	target    []float32
	lr        float64
}

type stVolCell struct {
	enabled         bool
	dim, seq, depth int
	dtype           core.DType
	W               []float32
}

// Dense children need [batch, dim]; fold seq into batch (Sequential-style layout).
func smokeStCfg() (dim, nChildren, seq int) {
	return 6, 3, 1
}

func volStCfg() (dim, nChildren, seq int) {
	return 4, 2, 1
}

func stWeightCount(dim, nChildren int) int {
	return nChildren * dim * dim
}

func disableStackTrain(s *parallel.Stack) {
	be := core.BackendCPUTiled
	var walk func(op any)
	walk = func(op any) {
		switch v := op.(type) {
		case *parallel.Stack:
			v.Exec.Backend = be
			v.Exec.MultiCore = false
			v.Core.MultiCore = false
			for _, ch := range v.Children {
				walk(ch)
			}
		case *dense.Layer:
			v.Exec.Backend = be
			v.Exec.MultiCore = false
			v.Core.MultiCore = false
		}
	}
	walk(s)
}

// buildNestedStackDense — outer Stack whose single child is NestedStack of Dense Dim→Dim Linear.
func buildNestedStackDense(dim, nChildren int, dt core.DType, W []float32) (*parallel.Stack, error) {
	if dim <= 0 || nChildren <= 0 {
		return nil, fmt.Errorf("stack: bad dims")
	}
	need := stWeightCount(dim, nChildren)
	if len(W) < need {
		return nil, fmt.Errorf("stack: W short need %d have %d", need, len(W))
	}
	kids := make([]any, nChildren)
	off := 0
	for i := 0; i < nChildren; i++ {
		nw := dim * dim
		d, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, quant.FormatNone, W[off:off+nw])
		if err != nil {
			return nil, err
		}
		kids[i] = d
		off += nw
	}
	inner, err := parallel.NewStack(kids...)
	if err != nil {
		return nil, err
	}
	outer, err := parallel.NewStack(inner)
	if err != nil {
		return nil, err
	}
	disableStackTrain(outer)
	return outer, nil
}

func packStackWeightsF32(s *parallel.Stack) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("stack: nil")
	}
	out := make([]float32, 0, s.GradWSize())
	var walk func(op any) error
	walk = func(op any) error {
		switch v := op.(type) {
		case *parallel.Stack:
			for _, ch := range v.Children {
				if err := walk(ch); err != nil {
					return err
				}
			}
		case *dense.Layer:
			w, err := weights.MatrixF32(v.Weights)
			if err != nil {
				return err
			}
			out = append(out, w...)
		default:
			return fmt.Errorf("stack pack: unexpected %T", op)
		}
		return nil
	}
	if err := walk(s); err != nil {
		return nil, err
	}
	return out, nil
}

func runGoStNBP(fc stNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	s, err := buildNestedStackDense(fc.dim, fc.nChildren, fc.dtype, fc.W)
	if err != nil {
		return z, err
	}
	// Fold seq into batch so Dense sees [batch*seq, dim].
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	xin := core.NewTensor[float32](fc.batch*seq, fc.dim)
	copy(xin.Data, fc.x)
	preT, postT, err := parallel.ForwardStack(s, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](postT.Shape...)
	copy(tgt.Data, fc.target)
	loss, err := training.MSE(postT, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(postT, tgt)
	if err != nil {
		return z, err
	}
	dX, dW, err := parallel.BackwardStack(s, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := parallel.ApplyGradSGDStack(s, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packStackWeightsF32(s)
	if err != nil {
		return z, err
	}
	z.pre = append([]float32(nil), preT.Data...)
	z.post = append([]float32(nil), postT.Data...)
	z.dX = append([]float32(nil), dX.Data...)
	z.dW = append([]float32(nil), dW.Data...)
	z.WAfter = append([]float32(nil), wa...)
	z.loss = loss
	return z, nil
}

func encodeStNBP(fc stNBPLeaf) ([]byte, error) {
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("STKB"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(fc.dim), int32(fc.nChildren),
		int32(core.ActivationLinear), int32(fc.dtype), 0,
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(1))
	for _, v := range fc.target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, fc.lr)
	return in.Bytes(), nil
}

func decodeStNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "STKO")
}

func runCppStNBP(fc stNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCppStNBPBench(fc, 0)
	return out, err
}

func runCppStNBPBench(fc stNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeStNBP(fc)
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nbpLeafOut{}, 0, fmt.Errorf("cpp stack nbp: %v: %s", err, stderr.String())
	}
	return decodeStNBP(stdout.Bytes())
}

func makeStLeafCase(dim, nChildren, batch, seq int, dt core.DType, seed uint32) (stNBPLeaf, error) {
	if seq <= 0 {
		seq = 1
	}
	fc := stNBPLeaf{
		batch: batch, seq: seq, dim: dim, nChildren: nChildren, dtype: dt, lr: 1e-2,
		W: randInit(1, stWeightCount(dim, nChildren), seed),
		x: randInit(batch*seq, dim, seed^0x9e3779b9),
	}
	fc.target = make([]float32, batch*seq*dim)
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

func caseStNBPLeafF32() error {
	const benchIters = 40
	dim, nChildren, seq := smokeStCfg()
	fc, err := makeStLeafCase(dim, nChildren, 2, seq, core.DTypeFloat32, 0x53544b42)
	if err != nil {
		return err
	}
	goO, err := runGoStNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppStNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoStNBP(fc); err != nil {
			return err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoStNBP(fc); err != nil {
			return err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppStNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%  (neg ⇒ C++ faster)",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseStNBPLeafAllDTypes() error {
	const benchIters = 20
	dim, nChildren, seq := smokeStCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — Stack NBP leaf ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeStLeafCase(dim, nChildren, 2, seq, dt, 0x53544b34)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoStNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppStNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		mW := suites.CompareF32Spectrum(goO.WAfter, cppO.WAfter)
		mDX := suites.CompareF32Spectrum(goO.dX, cppO.dX)
		mDW := suites.CompareF32Spectrum(goO.dW, cppO.dW)
		band := mW.Band
		for _, b := range []suites.DetBand{mDX.Band, mDW.Band} {
			if b.Percent() < band.Percent() {
				band = b
			}
		}
		if band.Percent() < worst.Percent() {
			worst = band
		}
		for i := 0; i < 3; i++ {
			if _, err := runGoStNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go warm: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, err := runGoStNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, cppDur, err := runCppStNBPBench(fc, benchIters)
		goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10s %9s %s\n",
				dt, band, mW.ExactPct, goUS, "?", "?", fmt.Sprintf("%.3g", mW.MaxAbs))
			continue
		}
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, band, mW.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", mW.MaxAbs))
		if band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, band))
		}
	}
	fmt.Printf("      worst_band=%s — want exact_bit on all 34\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func runGoStVolNBP(depth, rows, cols, lpc, batch int, cells []stVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstDim, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstDim < 0 {
			firstDim, firstSeq = vc.dim, vc.seq
		}
		sl, err := buildNestedStackDense(vc.dim, vc.depth, vc.dtype, vc.W)
		if err != nil {
			return z, err
		}
		if err := parallel.PlaceStack(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	if firstSeq <= 0 {
		firstSeq = 1
	}
	xin := core.NewTensor[float32](batch*firstSeq, firstDim)
	copy(xin.Data, x)
	fwd, err := forward.Forward(g, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fwd.Output.Shape...)
	copy(tgt.Data, target)
	loss, err := training.MSE(fwd.Output, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(fwd.Output, tgt)
	if err != nil {
		return z, err
	}
	bwd, err := backward.Backward(fwd, gy)
	if err != nil {
		return z, err
	}
	if err := training.SGD(fwd, bwd, lr); err != nil {
		return z, err
	}
	z.loss = loss
	z.post = append([]float32(nil), fwd.Output.Data...)
	z.gradIn = append([]float32(nil), bwd.GradIn.Data...)
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		sl, ok := cell.Op.(*parallel.Stack)
		if !ok {
			return z, fmt.Errorf("expected stack at %v", c)
		}
		wa, err := packStackWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeStVolNBP(depth, rows, cols, lpc, batch int, cells []stVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("STKV"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0, 0} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	if len(cells) != depth*rows*cols*lpc {
		return nil, fmt.Errorf("cells len %d != stack", len(cells))
	}
	for _, c := range cells {
		en := int32(0)
		if c.enabled {
			en = 1
		}
		_ = binary.Write(&in, binary.LittleEndian, en)
		if !c.enabled {
			continue
		}
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dim))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.seq))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.depth))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.W {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	for _, v := range x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return in.Bytes(), nil
}

func decodeStVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "STVO")
}

func runCppStVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nbpVolOut{}, 0, fmt.Errorf("cpp stack vol nbp: %v: %s", err, stderr.String())
	}
	return decodeStVolNBP(stdout.Bytes())
}

func benchStVolNBP(depth, rows, cols, lpc, batch int, cells []stVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppStVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func stVolPostLen(batch, dim, seq int) int {
	if seq <= 0 {
		seq = 1
	}
	return batch * seq * dim
}

func makeStVolCell(dim, nChildren, seq int, dt core.DType, seed uint32) stVolCell {
	return stVolCell{
		enabled: true,
		dim:     dim,
		seq:     seq,
		depth:   nChildren,
		dtype:   dt,
		W:       randInit(1, stWeightCount(dim, nChildren), seed),
	}
}

func caseStNBPVol1x1x1() error {
	const batch, benchIters, lr = 2, 40, 1e-2
	dim, nChildren, seq := volStCfg()
	W := randInit(1, stWeightCount(dim, nChildren), 0x53545631)
	cells := []stVolCell{{
		enabled: true, dim: dim, seq: seq, depth: nChildren,
		dtype: core.DTypeFloat32, W: W,
	}}
	x := randInit(batch*seq, dim, 0x53545632)
	target := make([]float32, stVolPostLen(batch, dim, seq))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoStVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeStVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	cppO, _, err := runCppStVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchStVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseStNBPVolChain() error {
	const batch, benchIters, lr = 1, 40, 1e-2
	dim, nChildren, seq := volStCfg()
	W0 := randInit(1, stWeightCount(dim, nChildren), 0x53545633)
	W1 := randInit(1, stWeightCount(dim, nChildren), 0x53545634)
	cells := []stVolCell{
		{enabled: true, dim: dim, seq: seq, depth: nChildren, dtype: core.DTypeFloat32, W: W0},
		{enabled: true, dim: dim, seq: seq, depth: nChildren, dtype: core.DTypeFloat32, W: W1},
	}
	x := randInit(batch*seq, dim, 0x53545635)
	target := make([]float32, stVolPostLen(batch, dim, seq))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoStVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeStVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	cppO, _, err := runCppStVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchStVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseStNBPVol111AllDTypes() error {
	const batch = 2
	dim, nChildren, seq := volStCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Stack NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []stVolCell{makeStVolCell(dim, nChildren, seq, dt, 0x53545636)}
		x := randInit(batch*seq, dim, 0x53545637)
		target := make([]float32, stVolPostLen(batch, dim, seq))
		for i := range target {
			target[i] = float32(0.03 * float64(i%4+1))
		}
		goO, err := runGoStVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeStVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppStVolNBP(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
		if m.Band.Percent() < worst.Percent() {
			worst = m.Band
		}
	}
	fmt.Printf("      worst_band=%s\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d vol dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseStNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	dim, nChildren, seq := volStCfg()
	W := randInit(1, stWeightCount(dim, nChildren), 0x53545638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, dim, 0x53545639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, stVolPostLen(batch, dim, seq))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]stVolCell, stack)
	for i := range cells {
		cells[i] = stVolCell{
			enabled: true, dim: dim, seq: seq, depth: nChildren,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	cppO, _, err := runCppStVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseStNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	stack := depth * rows * cols * lpc
	dim, nChildren, seq := volStCfg()
	W := randInit(1, stWeightCount(dim, nChildren), 0x5354563a)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, dim, 0x5354563b)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, stVolPostLen(batch, dim, seq))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Stack NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := make([]stVolCell, stack)
		for i := range cells {
			cells[i] = stVolCell{
				enabled: true, dim: dim, seq: seq, depth: nChildren,
				dtype: dt, W: append([]float32(nil), W...),
			}
		}
		goO, err := runGoStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeStVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppStVolNBP(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
		if m.Band.Percent() < worst.Percent() {
			worst = m.Band
		}
	}
	fmt.Printf("      worst_band=%s\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d vol211 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesSwiGLUTrain — NormalBP SwiGLU Go↔C++ (leaf + volumetric).
func CasesSwiGLUTrain() []Case {
	return []Case{
		{Name: "SwiGLU NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseSwiGLUNBPLeafF32},
		{Name: "SwiGLU NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseSwiGLUNBPLeafAllDTypes},
		{Name: "SwiGLU NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseSwiGLUNBPVol1x1x1},
		{Name: "SwiGLU NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseSwiGLUNBPVolChain},
		{Name: "SwiGLU NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseSwiGLUNBPVol111AllDTypes},
		{Name: "SwiGLU NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUNBPVol222F32},
		{Name: "SwiGLU NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseSwiGLUNBPVol211AllDTypes},
	}
}

func RunSwiGLUTrain() error {
	var fails []string
	cs := CasesSwiGLUTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [sgbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_swiglu_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_swiglu_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp swiglu_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type sgbpLeaf struct {
	batch, seq, in, inter int
	dtype                 core.DType
	gate, up, down        []float32
	x, target             []float32
	lr                    float64
}

type sgbpLeafOut struct {
	pre, post, dX, dW, WAfter []float32
	loss                      float64
}

func makeSwiGLUTrainInit(in, inter, batch int, seed uint32) (gate, up, down, x []float32) {
	gate = randInit(inter, in, seed)
	up = randInit(inter, in, seed+1)
	down = randInit(in, inter, seed+2)
	x = randInit(batch, in, seed+3)
	return gate, up, down, x
}

func exportSwiGLUW(l *swiglu.Layer) ([]float32, error) {
	g, err := weights.MatrixF32(l.Gate.Weights)
	if err != nil {
		return nil, err
	}
	u, err := weights.MatrixF32(l.Up.Weights)
	if err != nil {
		return nil, err
	}
	d, err := weights.MatrixF32(l.Down.Weights)
	if err != nil {
		return nil, err
	}
	out := make([]float32, 0, len(g)+len(u)+len(d))
	out = append(out, g...)
	out = append(out, u...)
	out = append(out, d...)
	return out, nil
}

func runGoSwiGLUNBP(fc sgbpLeaf) (sgbpLeafOut, error) {
	var z sgbpLeafOut
	cfg := swiglu.Config{InputDim: fc.in, IntermediateDim: fc.inter}
	l, err := swiglu.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.gate, fc.up, fc.down)
	if err != nil {
		return z, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	l.Gate.Exec.Backend = core.BackendCPUTiled
	l.Up.Exec.Backend = core.BackendCPUTiled
	l.Down.Exec.Backend = core.BackendCPUTiled
	l.Gate.Exec.MultiCore = false
	l.Up.Exec.MultiCore = false
	l.Down.Exec.MultiCore = false

	var xin *core.Tensor[float32]
	if fc.seq == 1 {
		xin = core.NewTensor[float32](fc.batch, fc.in)
	} else {
		xin = core.NewTensor[float32](fc.batch, fc.seq, fc.in)
	}
	copy(xin.Data, fc.x)
	preT, postT, err := swiglu.Forward(l, xin)
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
	dX, dW, err := swiglu.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := swiglu.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	Wa, err := exportSwiGLUW(l)
	if err != nil {
		return z, err
	}
	z.pre = append([]float32(nil), preT.Data...)
	z.post = append([]float32(nil), postT.Data...)
	z.dX = append([]float32(nil), dX.Data...)
	z.dW = append([]float32(nil), dW.Data...)
	z.WAfter = append([]float32(nil), Wa...)
	z.loss = loss
	return z, nil
}

func encodeSwiGLUNBP(fc sgbpLeaf) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WSB1"))
	for _, v := range []int32{int32(fc.batch), int32(fc.seq), int32(fc.in), int32(fc.inter), int32(fc.dtype), 0} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.gate {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.up {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.down {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.x {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(1)) // use_target
	for _, v := range fc.target {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, fc.lr)
	return in.Bytes()
}

func decodeSwiGLUNBP(raw []byte) (sgbpLeafOut, time.Duration, error) {
	var z sgbpLeafOut
	var bench time.Duration
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return z, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return z, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return z, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "WSBO" {
		return z, bench, fmt.Errorf("cpp swiglu nbp bad magic")
	}
	r := bytes.NewReader(raw[4:])
	readVec := func() ([]float32, error) {
		var n int32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		v := make([]float32, n)
		for i := range v {
			if err := binary.Read(r, binary.LittleEndian, &v[i]); err != nil {
				return nil, err
			}
		}
		return v, nil
	}
	var err error
	if z.pre, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.post, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.dX, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.dW, err = readVec(); err != nil {
		return z, bench, err
	}
	if z.WAfter, err = readVec(); err != nil {
		return z, bench, err
	}
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	return z, bench, nil
}

func runCppSwiGLUNBP(fc sgbpLeaf) (sgbpLeafOut, error) {
	out, _, err := runCppSwiGLUNBPBench(fc, 0)
	return out, err
}

func runCppSwiGLUNBPBench(fc sgbpLeaf, benchIters int) (sgbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return sgbpLeafOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(encodeSwiGLUNBP(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return sgbpLeafOut{}, 0, fmt.Errorf("cpp swiglu nbp: %v: %s", err, stderr.String())
	}
	return decodeSwiGLUNBP(stdout.Bytes())
}

func compareSwiGLUNBP(goO, cppO sgbpLeafOut, requireExact bool) error {
	checks := []struct {
		name string
		a, b []float32
	}{
		{"pre", goO.pre, cppO.pre},
		{"post", goO.post, cppO.post},
		{"dX", goO.dX, cppO.dX},
		{"dW", goO.dW, cppO.dW},
		{"W_after", goO.WAfter, cppO.WAfter},
	}
	for _, c := range checks {
		m := suites.CompareF32Spectrum(c.a, c.b)
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("%s: %s", c.name, suites.FormatSpectrum(m))
		}
		if !requireExact && (m.Band == suites.DetBroken || m.Band == suites.DetBallpark) {
			return fmt.Errorf("%s: %s", c.name, suites.FormatSpectrum(m))
		}
	}
	if !lossWithin1Ulp(goO.loss, cppO.loss) {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact || rel > 1e-5 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

func caseSwiGLUNBPLeafF32() error {
	const benchIters = defaultGoBenchIters
	in, inter, batch := 8, 16, 2
	gate, up, down, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574910)
	target := make([]float32, batch*in)
	for i := range target {
		target[i] = float32(0.1*float64(i+1) + 0.05)
	}
	fc := sgbpLeaf{
		batch: batch, seq: 1, in: in, inter: inter, dtype: core.DTypeFloat32,
		gate: gate, up: up, down: down, x: x, target: target, lr: 1e-2,
	}
	goO, err := runGoSwiGLUNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSwiGLUNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareSwiGLUNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoSwiGLUNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cppDur, err := runCppSwiGLUNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	cppUS := perIterUS(cppDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%  (neg ⇒ C++ faster)",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseSwiGLUNBPLeafAllDTypes() error {
	const in, inter, batch, seq, benchIters = 8, 16, 2, 1, defaultGoBenchIters
	gate, up, down, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574911)
	target := make([]float32, batch*in)
	for i := range target {
		target[i] = float32(0.1*float64(i+1) + 0.05)
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ", "max|Δ|W")
	var fails []string
	worst := suites.DetExactBit
	for _, dt := range core.AllDTypes {
		fc := sgbpLeaf{
			batch: batch, seq: seq, in: in, inter: inter, dtype: dt,
			gate: append([]float32(nil), gate...), up: append([]float32(nil), up...),
			down: append([]float32(nil), down...), x: append([]float32(nil), x...),
			target: append([]float32(nil), target...), lr: 1e-2,
		}
		goO, err := runGoSwiGLUNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppSwiGLUNBP(fc)
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
		goUS, err := benchGoUS(func() error {
			_, err := runGoSwiGLUNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cppDur, err := runCppSwiGLUNBPBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, band, mW.ExactPct, FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", mW.MaxAbs))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, band, mW.ExactPct, FormatUS(goUS), FormatUS(cppUS), spdPct, fmt.Sprintf("%.3g", mW.MaxAbs))
		if band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, band))
		}
	}
	fmt.Printf("      worst_band=%s — want exact_bit on all 34\n", worst)
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster); cpp via --bench=N (no spawn)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

type sgVolCell struct {
	enabled        bool
	in, inter      int
	dtype          core.DType
	gate, up, down []float32
}

func runGoSwiGLUVolNBP(depth, rows, cols, lpc, batch int, cells []sgVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		cfg := swiglu.Config{InputDim: vc.in, IntermediateDim: vc.inter}
		sl, err := swiglu.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.gate, vc.up, vc.down)
		if err != nil {
			return z, err
		}
		sl.Exec.Backend = core.BackendCPUTiled
		sl.Exec.MultiCore = false
		sl.Core.MultiCore = false
		sl.Gate.Exec.Backend = core.BackendCPUTiled
		sl.Up.Exec.Backend = core.BackendCPUTiled
		sl.Down.Exec.Backend = core.BackendCPUTiled
		sl.Gate.Exec.MultiCore = false
		sl.Up.Exec.MultiCore = false
		sl.Down.Exec.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerSwiGLU, DType: vc.dtype, Activation: core.ActivationSilu,
			InputHeight: vc.in, OutputHeight: vc.in,
		}, sl); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, len(x)/batch)
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
		sl, ok := cell.Op.(*swiglu.Layer)
		if !ok {
			return z, fmt.Errorf("expected swiglu at %v", c)
		}
		Wa, err := exportSwiGLUW(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), Wa...))
	}
	return z, nil
}

func encodeSwiGLUVolNBP(depth, rows, cols, lpc, batch int, cells []sgVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WSV1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0, 0} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	if len(cells) != depth*rows*cols*lpc {
		return nil, fmt.Errorf("cells len %d != stack %d", len(cells), depth*rows*cols*lpc)
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.in))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.inter))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.gate {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.up {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.down {
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

func decodeSwiGLUVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	var z nbpVolOut
	var bench time.Duration
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		if err := binary.Read(r, binary.LittleEndian, &ns); err != nil {
			return z, 0, err
		}
		if err := binary.Read(r, binary.LittleEndian, &iters); err != nil {
			return z, 0, err
		}
		rest, err := io.ReadAll(r)
		if err != nil {
			return z, 0, err
		}
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "WSVO" {
		return z, bench, fmt.Errorf("cpp swiglu vol nbp bad magic")
	}
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	var ginN int32
	if err := binary.Read(r, binary.LittleEndian, &ginN); err != nil {
		return z, bench, err
	}
	z.gradIn = make([]float32, ginN)
	for i := range z.gradIn {
		if err := binary.Read(r, binary.LittleEndian, &z.gradIn[i]); err != nil {
			return z, bench, err
		}
	}
	var nCells int32
	if err := binary.Read(r, binary.LittleEndian, &nCells); err != nil {
		return z, bench, err
	}
	for i := int32(0); i < nCells; i++ {
		var nw int32
		if err := binary.Read(r, binary.LittleEndian, &nw); err != nil {
			return z, bench, err
		}
		w := make([]float32, nw)
		for j := range w {
			if err := binary.Read(r, binary.LittleEndian, &w[j]); err != nil {
				return z, bench, err
			}
		}
		z.Ws = append(z.Ws, w)
	}
	var outN int32
	if err := binary.Read(r, binary.LittleEndian, &outN); err != nil {
		return z, bench, err
	}
	z.post = make([]float32, outN)
	for i := range z.post {
		if err := binary.Read(r, binary.LittleEndian, &z.post[i]); err != nil {
			return z, bench, err
		}
	}
	return z, bench, nil
}

func runCppSwiGLUVolNBP(depth, rows, cols, lpc, batch int, cells []sgVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	out, _, err := runCppSwiGLUVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, 0)
	return out, err
}

func runCppSwiGLUVolNBPBench(depth, rows, cols, lpc, batch int, cells []sgVolCell, x, target []float32, lr float64, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	payload, err := encodeSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp swiglu vol nbp: %v: %s", err, stderr.String())
	}
	return decodeSwiGLUVolNBP(stdout.Bytes())
}

func benchSwiGLUVolNBP(depth, rows, cols, lpc, batch int, cells []sgVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	goUS, err = benchGoUS(func() error {
		_, err := runGoSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cppDur, err := runCppSwiGLUVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cppUS = perIterUS(cppDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func scaleSwiGLU(gate, up, down, x []float32, s float32) {
	for i := range gate {
		gate[i] *= s
	}
	for i := range up {
		up[i] *= s
	}
	for i := range down {
		down[i] *= s
	}
	for i := range x {
		x[i] *= s
	}
}

func caseSwiGLUNBPVol1x1x1() error {
	const batch, in, inter, benchIters = 2, 4, 8, defaultGoBenchIters
	gate, up, down, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574920)
	target := make([]float32, batch*in)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := []sgVolCell{{
		enabled: true, in: in, inter: inter, dtype: core.DTypeFloat32,
		gate: gate, up: up, down: down,
	}}
	const lr = 1e-2
	goO, err := runGoSwiGLUVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSwiGLUVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchSwiGLUVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseSwiGLUNBPVolChain() error {
	const batch, in, inter, benchIters = 1, 4, 8, defaultGoBenchIters
	g0, u0, d0, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574921)
	g1, u1, d1, _ := makeSwiGLUTrainInit(in, inter, batch, 0x53574922)
	cells := []sgVolCell{
		{enabled: true, in: in, inter: inter, dtype: core.DTypeFloat32, gate: g0, up: u0, down: d0},
		{enabled: true, in: in, inter: inter, dtype: core.DTypeFloat32, gate: g1, up: u1, down: d1},
	}
	target := make([]float32, in)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	const lr = 1e-2
	goO, err := runGoSwiGLUVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSwiGLUVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchSwiGLUVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 chain loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseSwiGLUNBPVol111AllDTypes() error {
	const depth, rows, cols, lpc, batch, in, inter, benchIters = 1, 1, 1, 1, 1, 4, 8, defaultGoBenchIters
	gate, up, down, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574923)
	target := make([]float32, in)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := []sgVolCell{{
			enabled: true, in: in, inter: inter, dtype: dt,
			gate: append([]float32(nil), gate...), up: append([]float32(nil), up...),
			down: append([]float32(nil), down...),
		}}
		goO, err := runGoSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, "—", "—", "cmpFAIL", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		goUS, cppUS, spdPct, err := benchSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cppUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
	}
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d vol111 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseSwiGLUNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, in, inter, benchIters = 2, 2, 2, 1, 1, 4, 8, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	gate, up, down, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574924)
	scaleSwiGLU(gate, up, down, x, 0.05)
	target := make([]float32, in)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]sgVolCell, stack)
	for i := range cells {
		cells[i] = sgVolCell{
			enabled: true, in: in, inter: inter, dtype: core.DTypeFloat32,
			gate: append([]float32(nil), gate...), up: append([]float32(nil), up...),
			down: append([]float32(nil), down...),
		}
	}
	goO, err := runGoSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 Float32 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseSwiGLUNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch, in, inter, benchIters = 2, 1, 1, 1, 1, 4, 8, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	gate, up, down, x := makeSwiGLUTrainInit(in, inter, batch, 0x53574925)
	scaleSwiGLU(gate, up, down, x, 0.05)
	target := make([]float32, in)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := make([]sgVolCell, stack)
		for i := range cells {
			cells[i] = sgVolCell{
				enabled: true, in: in, inter: inter, dtype: dt,
				gate: append([]float32(nil), gate...), up: append([]float32(nil), up...),
				down: append([]float32(nil), down...),
			}
		}
		goO, err := runGoSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, "—", "—", "cmpFAIL", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		goUS, cppUS, spdPct, err := benchSwiGLUVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cppUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
	}
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d vol211 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

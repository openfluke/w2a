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
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesDenseTrain — NormalBP Dense Go↔C++ (leaf + volumetric). Separate from forward CasesCore.
func CasesDenseTrain() []Case {
	return []Case{
		{Name: "Dense NormalBP leaf Float32 ReLU — exact bits (dX/dW/W/loss)", Run: caseDenseNBPLeafF32},
		{Name: "Dense NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseDenseNBPLeafAllDTypes},
		{Name: "Dense NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseDenseNBPVol1x1x1},
		{Name: "Dense NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseDenseNBPVolChain},
		{Name: "Dense NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseDenseNBPVol111AllDTypes},
		{Name: "Dense NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseDenseNBPVol222F32},
		{Name: "Dense NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseDenseNBPVol211AllDTypes},
	}
}

func RunDenseTrain() error {
	var fails []string
	cs := CasesDenseTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [nbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_dense_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_dense_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp dense_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type nbpLeaf struct {
	batch, in, out int
	act            core.ActivationType
	dtype          core.DType
	W              []float32
	bias           []float64
	x              []float32
	target         []float32
	lr             float64
}

type nbpLeafOut struct {
	pre, post, dX, dW, WAfter []float32
	loss                      float64
}

func runGoDenseNBP(fc nbpLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	l, err := dense.NewConfigured(fc.in, fc.out, fc.act, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return z, err
	}
	l.Weights.Bias = append([]float64(nil), fc.bias...)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false

	xin := core.NewTensor[float32](fc.batch, fc.in)
	copy(xin.Data, fc.x)
	preT, postT, err := dense.Forward(l, xin)
	if err != nil {
		return z, err
	}
	tgt := core.NewTensor[float32](fc.batch, fc.out)
	copy(tgt.Data, fc.target)
	loss, err := training.MSE(postT, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(postT, tgt)
	if err != nil {
		return z, err
	}
	dX, dW, err := dense.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := dense.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	Wa, err := weights.MatrixF32(l.Weights)
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

func encodeDenseNBP(fc nbpLeaf) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WDB1"))
	for _, v := range []int32{int32(fc.batch), int32(fc.in), int32(fc.out), int32(fc.act), int32(fc.dtype), 0} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, uint32(len(fc.bias)))
	for _, v := range fc.bias {
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

func decodeDenseNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	var z nbpLeafOut
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
	if len(raw) < 8 || string(raw[:4]) != "WDBO" {
		return z, bench, fmt.Errorf("cpp dense nbp bad magic")
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

func runCppDenseNBP(fc nbpLeaf) (nbpLeafOut, error) {
	out, _, err := runCppDenseNBPBench(fc, 0)
	return out, err
}

// runCppDenseNBPBench runs WDB1; benchIters>0 → in-process --bench (WVB1 ns), excluding spawn.
func runCppDenseNBPBench(fc nbpLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(encodeDenseNBP(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nbpLeafOut{}, 0, fmt.Errorf("cpp dense nbp: %v: %s", err, stderr.String())
	}
	return decodeDenseNBP(stdout.Bytes())
}

func compareNBP(goO, cppO nbpLeafOut, requireExact bool) error {
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
	if goO.loss != cppO.loss {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact && rel > 0 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
		if !requireExact && rel > 1e-5 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

func caseDenseNBPLeafF32() error {
	const benchIters = 40
	fc := nbpLeaf{
		batch: 2, in: 4, out: 3,
		act:    core.ActivationReLU,
		dtype:  core.DTypeFloat32,
		W:      []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1, 0.05, 0.15, -0.25, 0.35},
		bias:   []float64{0.1, -0.05, 0.02},
		x:      []float32{1, 0.5, -0.5, 2, -1, 0.25, 0.75, -0.25},
		target: []float32{0.2, 0.4, 0.6, 0.1, 0.3, 0.5},
		lr:     1e-2,
	}
	goO, err := runGoDenseNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppDenseNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoDenseNBP(fc); err != nil {
			return err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoDenseNBP(fc); err != nil {
			return err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppDenseNBPBench(fc, benchIters)
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

func caseDenseNBPLeafAllDTypes() error {
	const in, out, batch, benchIters = 8, 4, 2, 40
	W, x, bias := makeSharedInit(out, in, batch)
	target := make([]float32, batch*out)
	for i := range target {
		target[i] = float32(0.1*float64(i+1) + 0.05)
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ", "max|Δ|W")
	var fails []string
	worst := suites.DetExactBit
	for _, dt := range core.AllDTypes {
		fc := nbpLeaf{
			batch: batch, in: in, out: out,
			act: core.ActivationLinear, dtype: dt,
			W: append([]float32(nil), W...), bias: append([]float64(nil), bias...),
			x: append([]float32(nil), x...), target: append([]float32(nil), target...),
			lr: 1e-2,
		}
		goO, err := runGoDenseNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppDenseNBP(fc)
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
			if _, err := runGoDenseNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go warm: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, err := runGoDenseNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, cppDur, err := runCppDenseNBPBench(fc, benchIters)
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
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster); cpp via --bench=N (no spawn)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

type nbpVolOut struct {
	loss   float64
	gradIn []float32
	Ws     [][]float32
	post   []float32
}

func runGoVolNBP(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		dl, err := dense.NewConfigured(vc.in, vc.out, vc.act, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		dl.Weights.Bias = append([]float64(nil), vc.bias...)
		dl.Exec.Backend = core.BackendCPUTiled
		dl.Exec.MultiCore = false
		dl.Core.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerDense, DType: vc.dtype, Activation: vc.act,
			InputHeight: vc.in, OutputHeight: vc.out,
		}, dl); err != nil {
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
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return z, fmt.Errorf("expected dense at %v", c)
		}
		Wa, err := weights.MatrixF32(dl.Weights)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), Wa...))
	}
	return z, nil
}

func encodeVolNBP(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WVN1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0 /*cpu*/, 0 /*n_links*/} {
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.out))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.act))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.W {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		_ = binary.Write(&in, binary.LittleEndian, uint32(len(c.bias)))
		for _, v := range c.bias {
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

func decodeVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WVNO" {
		return z, bench, fmt.Errorf("cpp vol nbp bad magic")
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
		var o, inn int32
		if err := binary.Read(r, binary.LittleEndian, &o); err != nil {
			return z, bench, err
		}
		if err := binary.Read(r, binary.LittleEndian, &inn); err != nil {
			return z, bench, err
		}
		w := make([]float32, o*inn)
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

func runCppVolNBP(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) (nbpVolOut, error) {
	out, _, err := runCppVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, 0)
	return out, err
}

func runCppVolNBPBench(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	payload, err := encodeVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp vol nbp: %v: %s", err, stderr.String())
	}
	return decodeVolNBP(stdout.Bytes())
}

func benchVolNBP(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	for i := 0; i < 3; i++ {
		if _, err := runGoVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func compareVolNBP(goO, cppO nbpVolOut, requireExact bool) error {
	if goO.loss != cppO.loss {
		if requireExact || math.Abs(goO.loss-cppO.loss) > 1e-12 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	mPost := suites.CompareF32Spectrum(goO.post, cppO.post)
	mGin := suites.CompareF32Spectrum(goO.gradIn, cppO.gradIn)
	if requireExact && mPost.Band != suites.DetExactBit {
		return fmt.Errorf("post: %s", suites.FormatSpectrum(mPost))
	}
	if requireExact && mGin.Band != suites.DetExactBit {
		return fmt.Errorf("gradIn: %s", suites.FormatSpectrum(mGin))
	}
	if len(goO.Ws) != len(cppO.Ws) {
		return fmt.Errorf("n_cells go=%d cpp=%d", len(goO.Ws), len(cppO.Ws))
	}
	for i := range goO.Ws {
		m := suites.CompareF32Spectrum(goO.Ws[i], cppO.Ws[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
		if !requireExact && (m.Band == suites.DetBroken || m.Band == suites.DetBallpark) {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	return nil
}

func caseDenseNBPVol1x1x1() error {
	const batch, dim, benchIters = 2, 4, 40
	W := []float32{
		0.2, -0.1, 0.05, 0.3,
		-0.15, 0.25, 0.1, -0.05,
		0.4, 0.05, -0.2, 0.15,
		-0.05, 0.1, 0.3, -0.25,
	}
	x := []float32{1, 0.5, -0.5, 0.25, -1, 0.75, 0.25, -0.5}
	target := []float32{0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45}
	cells := []volCell{{
		enabled: true, in: dim, out: dim, act: core.ActivationReLU,
		dtype: core.DTypeFloat32, W: W, bias: []float64{0.01, -0.02, 0.03, -0.01},
	}}
	const lr = 1e-2
	goO, err := runGoVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseDenseNBPVolChain() error {
	const batch, benchIters = 1, 40
	W0 := []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	W1 := []float32{1, 0, 0, 0, 1, 0}
	cells := []volCell{
		{enabled: true, in: 4, out: 3, act: core.ActivationReLU, dtype: core.DTypeFloat32, W: W0, bias: []float64{0.1, -0.2, 0.05}},
		{enabled: true, in: 3, out: 2, act: core.ActivationLinear, dtype: core.DTypeFloat32, W: W1, bias: []float64{0.5, -0.25}},
	}
	x := []float32{1, 2, 3, 4}
	target := []float32{0.5, -0.25}
	const lr = 1e-2
	goO, err := runGoVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 chain loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseDenseNBPVol111AllDTypes() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 1, 1, 1, 1, 1, 4, 40
	W, x, bias := makeSharedInit(dim, dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := []volCell{{
			enabled: true, in: dim, out: dim, act: core.ActivationLinear,
			dtype: dt, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...),
		}}
		goO, err := runGoVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% FAIL %s\n", dt, m.Band, m.ExactPct, fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		goUS, cppUS, spdPct, err := benchVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
	}
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d vol111 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseDenseNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 2, 2, 2, 1, 1, 4, 40
	stack := depth * rows * cols * lpc
	W, x, bias := makeSharedInit(dim, dim, batch)
	for i := range W {
		W[i] *= 0.05
	}
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]volCell, stack)
	for i := range cells {
		cells[i] = volCell{
			enabled: true, in: dim, out: dim, act: core.ActivationLinear,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...),
		}
	}
	goO, err := runGoVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 Float32 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

// 2×1×1×1 = two-cell depth chain (true multi-cell volume) without 8-hop blow-up on low-bit signed.
func caseDenseNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 2, 1, 1, 1, 1, 4, 40
	stack := depth * rows * cols * lpc
	W, x, bias := makeSharedInit(dim, dim, batch)
	// Scale down init for multi-hop low-bit stability (still FormatNone×34).
	for i := range W {
		W[i] *= 0.05
	}
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := make([]volCell, stack)
		for i := range cells {
			cells[i] = volCell{
				enabled: true, in: dim, out: dim, act: core.ActivationLinear,
				dtype: dt, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...),
			}
		}
		goO, err := runGoVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% FAIL %s\n", dt, m.Band, m.ExactPct, fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		goUS, cppUS, spdPct, err := benchVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
	}
	fmt.Printf("      spd%%Δ = (cpp−go)/go×100  (negative ⇒ C++ faster)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d vol211 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

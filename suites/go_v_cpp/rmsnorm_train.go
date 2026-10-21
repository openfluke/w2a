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
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesRMSNormTrain — NormalBP RMSNorm Go↔C++ (leaf + volumetric).
func CasesRMSNormTrain() []Case {
	return []Case{
		{Name: "RMSNorm NormalBP leaf Float32 — exact bits (dX/dγ/γ/loss)", Run: caseRMSNormNBPLeafF32},
		{Name: "RMSNorm NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseRMSNormNBPLeafAllDTypes},
		{Name: "RMSNorm NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseRMSNormNBPVol1x1x1},
		{Name: "RMSNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseRMSNormNBPVolChain},
		{Name: "RMSNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseRMSNormNBPVol111AllDTypes},
		{Name: "RMSNorm NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormNBPVol222F32},
		{Name: "RMSNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseRMSNormNBPVol211AllDTypes},
	}
}

func RunRMSNormTrain() error {
	var fails []string
	cs := CasesRMSNormTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [rnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_rmsnorm_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_rmsnorm_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp rmsnorm_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type rnbpLeaf struct {
	batch, seq, dim int
	eps             float64
	dtype           core.DType
	gamma           []float32
	x               []float32
	target          []float32
	lr              float64
}

type rnbpLeafOut struct {
	pre, post, dX, dW, gammaAfter []float32
	loss                          float64
}

func makeRMSGammaInit(dim, batch int) (gamma, x []float32) {
	gamma = make([]float32, dim)
	x = make([]float32, batch*dim)
	var s uint32 = 0xBADC0DE
	next := func() float32 {
		s = s*1664525 + 1013904223
		return float32(int32(s%2001)-1000) / 100.0
	}
	for i := range gamma {
		gamma[i] = 0.5 + next()*0.01
	}
	for i := range x {
		x[i] = next()
	}
	return gamma, x
}

func runGoRMSNormNBP(fc rnbpLeaf) (rnbpLeafOut, error) {
	var z rnbpLeafOut
	cfg := rmsnorm.Config{Dim: fc.dim, Eps: fc.eps}
	l, err := rmsnorm.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.gamma)
	if err != nil {
		return z, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false

	var xin *core.Tensor[float32]
	if fc.seq == 1 {
		xin = core.NewTensor[float32](fc.batch, fc.dim)
	} else {
		xin = core.NewTensor[float32](fc.batch, fc.seq, fc.dim)
	}
	copy(xin.Data, fc.x)
	preT, postT, err := rmsnorm.Forward(l, xin)
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
	dX, dW, err := rmsnorm.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := rmsnorm.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	ga, err := weights.MatrixF32(l.Gamma)
	if err != nil {
		return z, err
	}
	z.pre = append([]float32(nil), preT.Data...)
	z.post = append([]float32(nil), postT.Data...)
	z.dX = append([]float32(nil), dX.Data...)
	z.dW = append([]float32(nil), dW.Data...)
	z.gammaAfter = append([]float32(nil), ga...)
	z.loss = loss
	return z, nil
}

func encodeRMSNormNBP(fc rnbpLeaf) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WRB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.seq))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dim))
	_ = binary.Write(&in, binary.LittleEndian, fc.eps)
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(0)) // cpu
	for _, v := range fc.gamma {
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

func decodeRMSNormNBP(raw []byte) (rnbpLeafOut, time.Duration, error) {
	var z rnbpLeafOut
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
	if len(raw) < 8 || string(raw[:4]) != "WRBO" {
		return z, bench, fmt.Errorf("cpp rmsnorm nbp bad magic")
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
	if z.gammaAfter, err = readVec(); err != nil {
		return z, bench, err
	}
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	return z, bench, nil
}

func runCppRMSNormNBP(fc rnbpLeaf) (rnbpLeafOut, error) {
	out, _, err := runCppRMSNormNBPBench(fc, 0)
	return out, err
}

func runCppRMSNormNBPBench(fc rnbpLeaf, benchIters int) (rnbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return rnbpLeafOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(encodeRMSNormNBP(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return rnbpLeafOut{}, 0, fmt.Errorf("cpp rmsnorm nbp: %v: %s", err, stderr.String())
	}
	return decodeRMSNormNBP(stdout.Bytes())
}

func compareRMSNormNBP(goO, cppO rnbpLeafOut, requireExact bool) error {
	checks := []struct {
		name string
		a, b []float32
	}{
		{"pre", goO.pre, cppO.pre},
		{"post", goO.post, cppO.post},
		{"dX", goO.dX, cppO.dX},
		{"dW", goO.dW, cppO.dW},
		{"gamma_after", goO.gammaAfter, cppO.gammaAfter},
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

func caseRMSNormNBPLeafF32() error {
	const benchIters = defaultGoBenchIters
	fc := rnbpLeaf{
		batch: 2, seq: 1, dim: 4, eps: 0,
		dtype:  core.DTypeFloat32,
		gamma:  []float32{1.0, 0.9, 1.1, 0.95},
		x:      []float32{1, 0.5, -0.5, 2, -1, 0.25, 0.75, -0.25},
		target: []float32{0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, 0.15},
		lr:     1e-2,
	}
	goO, err := runGoRMSNormNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppRMSNormNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareRMSNormNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoRMSNormNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cppDur, err := runCppRMSNormNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	cppUS := perIterUS(cppDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%  (neg ⇒ C++ faster)",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseRMSNormNBPLeafAllDTypes() error {
	const dim, batch, seq, benchIters = 8, 2, 1, defaultGoBenchIters
	gamma, x := makeRMSGammaInit(dim, batch)
	target := make([]float32, batch*dim)
	for i := range target {
		target[i] = float32(0.1*float64(i+1) + 0.05)
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%γ", "goµs", "cppµs", "spd%Δ", "max|Δ|γ")
	var fails []string
	worst := suites.DetExactBit
	for _, dt := range core.AllDTypes {
		fc := rnbpLeaf{
			batch: batch, seq: seq, dim: dim, eps: 0,
			dtype: dt,
			gamma: append([]float32(nil), gamma...), x: append([]float32(nil), x...),
			target: append([]float32(nil), target...), lr: 1e-2,
		}
		goO, err := runGoRMSNormNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppRMSNormNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		mG := suites.CompareF32Spectrum(goO.gammaAfter, cppO.gammaAfter)
		mDX := suites.CompareF32Spectrum(goO.dX, cppO.dX)
		mDW := suites.CompareF32Spectrum(goO.dW, cppO.dW)
		band := mG.Band
		for _, b := range []suites.DetBand{mDX.Band, mDW.Band} {
			if b.Percent() < band.Percent() {
				band = b
			}
		}
		if band.Percent() < worst.Percent() {
			worst = band
		}
		goUS, err := benchGoUS(func() error {
			_, err := runGoRMSNormNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cppDur, err := runCppRMSNormNBPBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, band, mG.ExactPct, FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", mG.MaxAbs))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, band, mG.ExactPct, FormatUS(goUS), FormatUS(cppUS), spdPct, fmt.Sprintf("%.3g", mG.MaxAbs))
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

type rnVolCell struct {
	enabled bool
	dim     int
	dtype   core.DType
	eps     float64
	gamma   []float32
}

func runGoRMSNormVolNBP(depth, rows, cols, lpc, batch int, cells []rnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := rmsnorm.Config{Dim: vc.dim, Eps: vc.eps}
		rl, err := rmsnorm.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.gamma)
		if err != nil {
			return z, err
		}
		rl.Exec.Backend = core.BackendCPUTiled
		rl.Exec.MultiCore = false
		rl.Core.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerRMSNorm, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: vc.dim, OutputHeight: vc.dim,
		}, rl); err != nil {
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
		rl, ok := cell.Op.(*rmsnorm.Layer)
		if !ok {
			return z, fmt.Errorf("expected rmsnorm at %v", c)
		}
		ga, err := weights.MatrixF32(rl.Gamma)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), ga...))
	}
	return z, nil
}

func encodeRMSNormVolNBP(depth, rows, cols, lpc, batch int, cells []rnVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WRV1"))
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dim))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		_ = binary.Write(&in, binary.LittleEndian, c.eps)
		for _, v := range c.gamma {
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

func decodeRMSNormVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WRVO" {
		return z, bench, fmt.Errorf("cpp rmsnorm vol nbp bad magic")
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
		var d int32
		if err := binary.Read(r, binary.LittleEndian, &d); err != nil {
			return z, bench, err
		}
		w := make([]float32, d)
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

func runCppRMSNormVolNBP(depth, rows, cols, lpc, batch int, cells []rnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	out, _, err := runCppRMSNormVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, 0)
	return out, err
}

func runCppRMSNormVolNBPBench(depth, rows, cols, lpc, batch int, cells []rnVolCell, x, target []float32, lr float64, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	payload, err := encodeRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp rmsnorm vol nbp: %v: %s", err, stderr.String())
	}
	return decodeRMSNormVolNBP(stdout.Bytes())
}

func benchRMSNormVolNBP(depth, rows, cols, lpc, batch int, cells []rnVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	goUS, err = benchGoUS(func() error {
		_, err := runGoRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cppDur, err := runCppRMSNormVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cppUS = perIterUS(cppDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func caseRMSNormNBPVol1x1x1() error {
	const batch, dim, benchIters = 2, 4, defaultGoBenchIters
	gamma := []float32{1.0, 0.9, 1.1, 0.95}
	x := []float32{1, 0.5, -0.5, 0.25, -1, 0.75, 0.25, -0.5}
	target := []float32{0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45}
	cells := []rnVolCell{{
		enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0, gamma: gamma,
	}}
	const lr = 1e-2
	goO, err := runGoRMSNormVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppRMSNormVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchRMSNormVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseRMSNormNBPVolChain() error {
	const batch, dim, benchIters = 1, 4, defaultGoBenchIters
	g0 := []float32{1, 0.9, 1.1, 0.95}
	g1 := []float32{0.8, 1.2, 1.0, 0.85}
	cells := []rnVolCell{
		{enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0, gamma: g0},
		{enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0, gamma: g1},
	}
	x := []float32{1, 2, 3, 4}
	target := []float32{0.5, -0.25, 0.1, 0.2}
	const lr = 1e-2
	goO, err := runGoRMSNormVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppRMSNormVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchRMSNormVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 chain loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseRMSNormNBPVol111AllDTypes() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 1, 1, 1, 1, 1, 4, defaultGoBenchIters
	gamma, x := makeRMSGammaInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := []rnVolCell{{
			enabled: true, dim: dim, dtype: dt, eps: 0, gamma: append([]float32(nil), gamma...),
		}}
		goO, err := runGoRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
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
		goUS, cppUS, spdPct, err := benchRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
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

func caseRMSNormNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 2, 2, 2, 1, 1, 4, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	gamma, x := makeRMSGammaInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]rnVolCell, stack)
	for i := range cells {
		cells[i] = rnVolCell{
			enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0,
			gamma: append([]float32(nil), gamma...),
		}
	}
	goO, err := runGoRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 Float32 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseRMSNormNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch, dim, benchIters = 2, 1, 1, 1, 1, 4, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	gamma, x := makeRMSGammaInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := make([]rnVolCell, stack)
		for i := range cells {
			cells[i] = rnVolCell{
				enabled: true, dim: dim, dtype: dt, eps: 0,
				gamma: append([]float32(nil), gamma...),
			}
		}
		goO, err := runGoRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
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
		goUS, cppUS, spdPct, err := benchRMSNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
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

package go_v_cuda

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
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesLayerNormTrain — NormalBP LayerNorm Go↔CUDA (leaf + volumetric).
func CasesLayerNormTrain() []Case {
	return []Case{
		{Name: "LayerNorm NormalBP leaf Float32 — exact bits (dX/dγβ/γβ/loss)", Run: caseLayerNormNBPLeafF32},
		{Name: "LayerNorm NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseLayerNormNBPLeafAllDTypes},
		{Name: "LayerNorm NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseLayerNormNBPVol1x1x1},
		{Name: "LayerNorm NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseLayerNormNBPVolChain},
		{Name: "LayerNorm NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseLayerNormNBPVol111AllDTypes},
		{Name: "LayerNorm NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormNBPVol222F32},
		{Name: "LayerNorm NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseLayerNormNBPVol211AllDTypes},
	}
}

func RunLayerNormTrain() error {
	var fails []string
	cs := CasesLayerNormTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [lnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_layernorm_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_layernorm_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda layernorm_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type lnbpLeaf struct {
	batch, seq, dim int
	eps             float64
	dtype           core.DType
	gamma, beta     []float32
	x               []float32
	target          []float32
	lr              float64
}

type lnbpLeafOut struct {
	pre, post, dX, dW, gammaAfter, betaAfter []float32
	loss                                     float64
}

func makeLNAffineInit(dim, batch int) (gamma, beta, x []float32) {
	gamma = make([]float32, dim)
	beta = make([]float32, dim)
	x = make([]float32, batch*dim)
	var s uint32 = 0xC0FFEE11
	next := func() float32 {
		s = s*1664525 + 1013904223
		return float32(int32(s%2001)-1000) / 100.0
	}
	for i := range gamma {
		gamma[i] = 0.5 + next()*0.01
		beta[i] = next() * 0.01
	}
	for i := range x {
		x[i] = next()
	}
	return gamma, beta, x
}

func runGoLayerNormNBP(fc lnbpLeaf) (lnbpLeafOut, error) {
	var z lnbpLeafOut
	cfg := layernorm.Config{Dim: fc.dim, Eps: fc.eps}
	l, err := layernorm.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.gamma, fc.beta)
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
	preT, postT, err := layernorm.Forward(l, xin)
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
	dX, dW, err := layernorm.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := layernorm.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	ga, err := weights.MatrixF32(l.Gamma)
	if err != nil {
		return z, err
	}
	ba, err := weights.MatrixF32(l.Beta)
	if err != nil {
		return z, err
	}
	z.pre = append([]float32(nil), preT.Data...)
	z.post = append([]float32(nil), postT.Data...)
	z.dX = append([]float32(nil), dX.Data...)
	z.dW = append([]float32(nil), dW.Data...)
	z.gammaAfter = append([]float32(nil), ga...)
	z.betaAfter = append([]float32(nil), ba...)
	z.loss = loss
	return z, nil
}

func encodeLayerNormNBP(fc lnbpLeaf) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WLB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.seq))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dim))
	_ = binary.Write(&in, binary.LittleEndian, fc.eps)
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(0)) // cpu
	for _, v := range fc.gamma {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.beta {
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

func decodeLayerNormNBP(raw []byte) (lnbpLeafOut, time.Duration, error) {
	var z lnbpLeafOut
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
	if len(raw) < 8 || string(raw[:4]) != "WLBO" {
		return z, bench, fmt.Errorf("cuda layernorm nbp bad magic")
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
	if z.betaAfter, err = readVec(); err != nil {
		return z, bench, err
	}
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	return z, bench, nil
}

func runCudaLayerNormNBP(fc lnbpLeaf) (lnbpLeafOut, error) {
	out, _, err := runCudaLayerNormNBPBench(fc, 0)
	return out, err
}

func runCudaLayerNormNBPBench(fc lnbpLeaf, benchIters int) (lnbpLeafOut, time.Duration, error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return lnbpLeafOut{}, 0, err
	}
	var cmd *exec.Cmd
	if benchIters > 0 {
		cmd = exec.Command(tool, fmt.Sprintf("--bench=%d", benchIters))
	} else {
		cmd = exec.Command(tool)
	}
	cmd.Stdin = bytes.NewReader(encodeLayerNormNBP(fc))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return lnbpLeafOut{}, 0, fmt.Errorf("cuda layernorm nbp: %v: %s", err, stderr.String())
	}
	return decodeLayerNormNBP(stdout.Bytes())
}

func compareLayerNormNBP(goO, cudaO lnbpLeafOut, requireExact bool) error {
	checks := []struct {
		name string
		a, b []float32
	}{
		{"pre", goO.pre, cudaO.pre},
		{"post", goO.post, cudaO.post},
		{"dX", goO.dX, cudaO.dX},
		{"dW", goO.dW, cudaO.dW},
		{"gamma_after", goO.gammaAfter, cudaO.gammaAfter},
		{"beta_after", goO.betaAfter, cudaO.betaAfter},
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
	if !cudaLossWithinBand(goO.loss, cudaO.loss) {
		rel := math.Abs(goO.loss-cudaO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact || rel > 1e-5 {
			return fmt.Errorf("loss go=%g cuda=%g", goO.loss, cudaO.loss)
		}
	}
	return nil
}

func caseLayerNormNBPLeafF32() error {
	var benchIters = defaultGoBenchIters
	fc := lnbpLeaf{
		batch: 2, seq: 1, dim: 4, eps: 0,
		dtype:  core.DTypeFloat32,
		gamma:  []float32{1.0, 0.9, 1.1, 0.95},
		beta:   []float32{0.01, -0.02, 0.03, -0.01},
		x:      []float32{1, 0.5, -0.5, 2, -1, 0.25, 0.75, -0.25},
		target: []float32{0.2, 0.4, 0.6, 0.1, 0.3, 0.5, 0.7, 0.15},
		lr:     1e-2,
	}
	goO, err := runGoLayerNormNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaLayerNormNBP(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareLayerNormNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoLayerNormNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cudaDur, err := runCudaLayerNormNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cuda bench: %w", err)
	}
	cudaUS := perIterUS(cudaDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cudaUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%  (neg ⇒ CUDA faster)",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseLayerNormNBPLeafAllDTypes() error {
	const dim, batch, seq = 8, 2, 1
	var benchIters = defaultGoBenchIters
	gamma, beta, x := makeLNAffineInit(dim, batch)
	target := make([]float32, batch*dim)
	for i := range target {
		target[i] = float32(0.1*float64(i+1) + 0.05)
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%γ", "goµs", "cudaµs", "spd%Δ", "max|Δ|γ")
	var fails []string
	worst := suites.DetExactBit
	for _, dt := range core.AllDTypes {
		fc := lnbpLeaf{
			batch: batch, seq: seq, dim: dim, eps: 0,
			dtype: dt,
			gamma: append([]float32(nil), gamma...), beta: append([]float32(nil), beta...),
			x: append([]float32(nil), x...), target: append([]float32(nil), target...), lr: 1e-2,
		}
		goO, err := runGoLayerNormNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaLayerNormNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			continue
		}
		mG := suites.CompareF32Spectrum(goO.gammaAfter, cudaO.gammaAfter)
		mB := suites.CompareF32Spectrum(goO.betaAfter, cudaO.betaAfter)
		mDX := suites.CompareF32Spectrum(goO.dX, cudaO.dX)
		mDW := suites.CompareF32Spectrum(goO.dW, cudaO.dW)
		band := mG.Band
		for _, b := range []suites.DetBand{mB.Band, mDX.Band, mDW.Band} {
			if b.Percent() < band.Percent() {
				band = b
			}
		}
		if band.Percent() < worst.Percent() {
			worst = band
		}
		goUS, err := benchGoUS(func() error {
			_, err := runGoLayerNormNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cudaDur, err := runCudaLayerNormNBPBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, band, mG.ExactPct, FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", mG.MaxAbs))
			continue
		}
		cudaUS := perIterUS(cudaDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cudaUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, band, mG.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", mG.MaxAbs))
		if band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s: %s", dt, band))
		}
	}
	fmt.Printf("      worst_band=%s — want exact_bit on all 34\n", worst)
	fmt.Printf("      spd%%Δ = (mtl−go)/go×100  (negative ⇒ CUDA faster); cuda via --bench=N (no spawn)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

type lnVolCell struct {
	enabled     bool
	dim         int
	dtype       core.DType
	eps         float64
	gamma, beta []float32
}

func concatGB(g, b []float32) []float32 {
	out := make([]float32, len(g)+len(b))
	copy(out, g)
	copy(out[len(g):], b)
	return out
}

func runGoLayerNormVolNBP(depth, rows, cols, lpc, batch int, cells []lnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := layernorm.Config{Dim: vc.dim, Eps: vc.eps}
		ll, err := layernorm.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.gamma, vc.beta)
		if err != nil {
			return z, err
		}
		ll.Exec.Backend = core.BackendCPUTiled
		ll.Exec.MultiCore = false
		ll.Core.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerLayerNorm, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: vc.dim, OutputHeight: vc.dim,
		}, ll); err != nil {
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
		ll, ok := cell.Op.(*layernorm.Layer)
		if !ok {
			return z, fmt.Errorf("expected layernorm at %v", c)
		}
		ga, err := weights.MatrixF32(ll.Gamma)
		if err != nil {
			return z, err
		}
		ba, err := weights.MatrixF32(ll.Beta)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, concatGB(ga, ba))
	}
	return z, nil
}

func encodeLayerNormVolNBP(depth, rows, cols, lpc, batch int, cells []lnVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WLV1"))
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
		for _, v := range c.beta {
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

func decodeLayerNormVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WLVO" {
		return z, bench, fmt.Errorf("cuda layernorm vol nbp bad magic")
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
		w := make([]float32, 2*d)
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

func runCudaLayerNormVolNBP(depth, rows, cols, lpc, batch int, cells []lnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	out, _, err := runCudaLayerNormVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, 0)
	return out, err
}

func runCudaLayerNormVolNBPBench(depth, rows, cols, lpc, batch int, cells []lnVolCell, x, target []float32, lr float64, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	payload, err := encodeLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
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
		return nbpVolOut{}, 0, fmt.Errorf("cuda layernorm vol nbp: %v: %s", err, stderr.String())
	}
	return decodeLayerNormVolNBP(stdout.Bytes())
}

func benchLayerNormVolNBP(depth, rows, cols, lpc, batch int, cells []lnVolCell, x, target []float32, lr float64, benchIters int) (goUS, cudaUS, spdPct float64, err error) {
	goUS, err = benchGoUS(func() error {
		_, err := runGoLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cudaDur, err := runCudaLayerNormVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cudaUS = perIterUS(cudaDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cudaUS)
	return goUS, cudaUS, spdPct, nil
}

func caseLayerNormNBPVol1x1x1() error {
	const batch, dim = 2, 4
	var benchIters = defaultGoBenchIters
	gamma := []float32{1.0, 0.9, 1.1, 0.95}
	beta := []float32{0.01, -0.02, 0.03, -0.01}
	x := []float32{1, 0.5, -0.5, 0.25, -1, 0.75, 0.25, -0.5}
	target := []float32{0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45}
	cells := []lnVolCell{{
		enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0, gamma: gamma, beta: beta,
	}}
	const lr = 1e-2
	goO, err := runGoLayerNormVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaLayerNormVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchLayerNormVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseLayerNormNBPVolChain() error {
	const batch, dim = 1, 4
	var benchIters = defaultGoBenchIters
	g0 := []float32{1, 0.9, 1.1, 0.95}
	b0 := []float32{0.01, -0.02, 0.03, -0.01}
	g1 := []float32{0.8, 1.2, 1.0, 0.85}
	b1 := []float32{-0.05, 0.02, 0.0, 0.04}
	cells := []lnVolCell{
		{enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0, gamma: g0, beta: b0},
		{enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0, gamma: g1, beta: b1},
	}
	x := []float32{1, 2, 3, 4}
	target := []float32{0.5, -0.25, 0.1, 0.2}
	const lr = 1e-2
	goO, err := runGoLayerNormVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaLayerNormVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchLayerNormVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 chain loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseLayerNormNBPVol111AllDTypes() error {
	const depth, rows, cols, lpc, batch, dim = 1, 1, 1, 1, 1, 4
	var benchIters = defaultGoBenchIters
	gamma, beta, x := makeLNAffineInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cudaµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := []lnVolCell{{
			enabled: true, dim: dim, dtype: dt, eps: 0,
			gamma: append([]float32(nil), gamma...), beta: append([]float32(nil), beta...),
		}}
		goO, err := runGoLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, "—", "—", "cmpFAIL", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		goUS, cudaUS, spdPct, err := benchLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
	}
	fmt.Printf("      spd%%Δ = (mtl−go)/go×100  (negative ⇒ CUDA faster)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d vol111 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseLayerNormNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, dim = 2, 2, 2, 1, 1, 4
	var benchIters = defaultGoBenchIters
	stack := depth * rows * cols * lpc
	gamma, beta, x := makeLNAffineInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]lnVolCell, stack)
	for i := range cells {
		cells[i] = lnVolCell{
			enabled: true, dim: dim, dtype: core.DTypeFloat32, eps: 0,
			gamma: append([]float32(nil), gamma...), beta: append([]float32(nil), beta...),
		}
	}
	goO, err := runGoLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 Float32 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseLayerNormNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch, dim = 2, 1, 1, 1, 1, 4
	var benchIters = defaultGoBenchIters
	stack := depth * rows * cols * lpc
	gamma, beta, x := makeLNAffineInit(dim, batch)
	target := make([]float32, dim)
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cudaµs", "spd%Δ", "max|Δ|")
	var fails []string
	for _, dt := range core.AllDTypes {
		cells := make([]lnVolCell, stack)
		for i := range cells {
			cells[i] = lnVolCell{
				enabled: true, dim: dim, dtype: dt, eps: 0,
				gamma: append([]float32(nil), gamma...), beta: append([]float32(nil), beta...),
			}
		}
		goO, err := runGoLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, m.Band, m.ExactPct, "—", "—", "cmpFAIL", fmt.Sprintf("%.3g", m.MaxAbs))
			continue
		}
		goUS, cudaUS, spdPct, err := benchLayerNormVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
	}
	fmt.Printf("      spd%%Δ = (mtl−go)/go×100  (negative ⇒ CUDA faster)\n")
	if len(fails) > 0 {
		return fmt.Errorf("%d vol211 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

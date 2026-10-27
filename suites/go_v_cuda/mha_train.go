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
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesMHATrain — NormalBP MHA ModeSelf Go↔CUDA (leaf + volumetric).
func CasesMHATrain() []Case {
	return []Case{
		{Name: "MHA NormalBP leaf DecoderCausal Float32 — exact bits", Run: caseMHANBPLeafF32},
		{Name: "MHA NormalBP leaf GQA + Encoder Float32 — exact bits", Run: caseMHANBPLeafVariants},
		{Name: "MHA NormalBP leaf DecoderCausal × all 34 dtypes — spectrum + speed", Run: caseMHANBPLeafAllDTypes},
		{Name: "MHA NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseMHANBPVol1x1x1},
		{Name: "MHA NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseMHANBPVolChain},
		{Name: "MHA NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseMHANBPVol111AllDTypes},
		{Name: "MHA NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseMHANBPVol222F32},
		{Name: "MHA NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseMHANBPVol211AllDTypes},
	}
}

func RunMHATrain() error {
	var fails []string
	cs := CasesMHATrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [mhbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_mha_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_mha_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda mha_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type mhbpLeaf struct {
	batch, seq int
	cfg        mha.Config
	dtype      core.DType
	q, k, v, o []float32
	x          []float32
	target     []float32
	lr         float64
}

type mhbpLeafOut struct {
	pre, post, dX, dW, WAfter []float32
	loss                      float64
}

func exportMHAWeights(l *mha.Layer) ([]float32, error) {
	var out []float32
	for _, p := range []*weights.Store{l.Q.Weights, l.K.Weights, l.V.Weights, l.O.Weights} {
		w, err := weights.MatrixF32(p)
		if err != nil {
			return nil, err
		}
		out = append(out, w...)
	}
	return out, nil
}

func makeMHANBPCase(cfg mha.Config, batch, seq int, dt core.DType, seed uint32, lr float64) (mhbpLeaf, error) {
	fc, err := makeMHACase(cfg, batch, seq, dt, seed)
	if err != nil {
		return mhbpLeaf{}, err
	}
	target := make([]float32, batch*seq*cfg.DModel)
	for i := range target {
		target[i] = float32(0.02 * float64(i%7+1))
	}
	return mhbpLeaf{
		batch: batch, seq: seq, cfg: fc.cfg, dtype: dt,
		q: fc.q, k: fc.k, v: fc.v, o: fc.o, x: fc.x, target: target, lr: lr,
	}, nil
}

func runGoMHANBP(fc mhbpLeaf) (mhbpLeafOut, error) {
	var z mhbpLeafOut
	cfg := fc.cfg
	l, err := mha.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return z, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false

	xin := core.NewTensor[float32](fc.batch, fc.seq, cfg.DModel)
	copy(xin.Data, fc.x)
	preT, postT, err := mha.Forward(l, xin)
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
	dX, dW, err := mha.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := mha.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := exportMHAWeights(l)
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

func encodeMHANBP(fc mhbpLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("WHB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.batch))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.seq))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.DModel))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumHeads))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumKVHeads))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.HeadDim))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.MaxSeqLen))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Mask))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Pos))
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(0)) // cpu
	for _, v := range fc.q {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.k {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.v {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	for _, v := range fc.o {
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
	return in.Bytes(), nil
}

func decodeMHANBP(raw []byte) (mhbpLeafOut, time.Duration, error) {
	var z mhbpLeafOut
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
	if len(raw) < 8 || string(raw[:4]) != "WHBO" {
		return z, bench, fmt.Errorf("cuda mha nbp bad magic")
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

func runCudaMHANBP(fc mhbpLeaf) (mhbpLeafOut, error) {
	out, _, err := runCudaMHANBPBench(fc, 0)
	return out, err
}

func runCudaMHANBPBench(fc mhbpLeaf, benchIters int) (mhbpLeafOut, time.Duration, error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return mhbpLeafOut{}, 0, err
	}
	payload, err := encodeMHANBP(fc)
	if err != nil {
		return mhbpLeafOut{}, 0, err
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
		return mhbpLeafOut{}, 0, fmt.Errorf("cuda mha nbp: %v: %s", err, stderr.String())
	}
	return decodeMHANBP(stdout.Bytes())
}

func compareMHANBP(goO, cudaO mhbpLeafOut, requireExact bool) error {
	checks := []struct {
		name string
		a, b []float32
	}{
		{"pre", goO.pre, cudaO.pre},
		{"post", goO.post, cudaO.post},
		{"dX", goO.dX, cudaO.dX},
		{"dW", goO.dW, cudaO.dW},
		{"W_after", goO.WAfter, cudaO.WAfter},
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

func caseMHANBPLeafF32() error {
	var benchIters = defaultGoBenchIters
	cfg := mha.DecoderCausal(32, 4, 4)
	cfg.MaxSeqLen = 16
	fc, err := makeMHANBPCase(cfg, 1, 4, core.DTypeFloat32, 0x4d484250, 1e-2)
	if err != nil {
		return err
	}
	goO, err := runGoMHANBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaMHANBP(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareMHANBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoMHANBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cudaDur, err := runCudaMHANBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cuda bench: %w", err)
	}
	cudaUS := perIterUS(cudaDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cudaUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%  (neg ⇒ CUDA faster)",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseMHANBPLeafVariants() error {
	gqa := mha.DecoderCausal(32, 4, 2)
	gqa.MaxSeqLen = 16
	fc, err := makeMHANBPCase(gqa, 1, 3, core.DTypeFloat32, 0x47514142, 1e-2)
	if err != nil {
		return err
	}
	goO, err := runGoMHANBP(fc)
	if err != nil {
		return fmt.Errorf("gqa go: %w", err)
	}
	cudaO, err := runCudaMHANBP(fc)
	if err != nil {
		return fmt.Errorf("gqa cuda: %w", err)
	}
	if err := compareMHANBP(goO, cudaO, true); err != nil {
		return fmt.Errorf("gqa: %w", err)
	}
	fmt.Printf("\n      gqa exact_bit loss=%.6g", goO.loss)

	enc := mha.EncoderBidirectional(32, 4)
	enc.MaxSeqLen = 16
	fc2, err := makeMHANBPCase(enc, 2, 3, core.DTypeFloat32, 0x454e4342, 1e-2)
	if err != nil {
		return err
	}
	goO2, err := runGoMHANBP(fc2)
	if err != nil {
		return fmt.Errorf("enc go: %w", err)
	}
	cudaO2, err := runCudaMHANBP(fc2)
	if err != nil {
		return fmt.Errorf("enc cuda: %w", err)
	}
	if err := compareMHANBP(goO2, cudaO2, true); err != nil {
		return fmt.Errorf("enc: %w", err)
	}
	fmt.Printf("\n      encoder exact_bit loss=%.6g", goO2.loss)
	return nil
}

func caseMHANBPLeafAllDTypes() error {
	var benchIters = defaultGoBenchIters
	cfg := mha.DecoderCausal(32, 4, 4)
	cfg.MaxSeqLen = 16
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs CUDA — MHA NBP DecoderCausal ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cuda_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeMHANBPCase(cfg, 1, 4, dt, 0xd7a4e0b0, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoMHANBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaMHANBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			continue
		}
		mW := suites.CompareF32Spectrum(goO.WAfter, cudaO.WAfter)
		mDX := suites.CompareF32Spectrum(goO.dX, cudaO.dX)
		mDW := suites.CompareF32Spectrum(goO.dW, cudaO.dW)
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
			_, err := runGoMHANBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cudaDur, err := runCudaMHANBPBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, band, mW.ExactPct, FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", mW.MaxAbs))
			continue
		}
		cudaUS := perIterUS(cudaDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, cudaUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, band, mW.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", mW.MaxAbs))
		// W_after must be exact_bit (FormatNone truth after SGD). dX/dW may be
		// ≥strict on rare packed dtypes (soft-double fwd → host MatVecT/outer ULP).
		if mW.Band != suites.DetExactBit {
			fails = append(fails, fmt.Sprintf("%s W: %s max=%.3g", dt, mW.Band, mW.MaxAbs))
		}
		if mDX.Band.Percent() < suites.DetStrict.Percent() {
			fails = append(fails, fmt.Sprintf("%s dX: %s max=%.3g", dt, mDX.Band, mDX.MaxAbs))
		}
		if mDW.Band.Percent() < suites.DetStrict.Percent() {
			fails = append(fails, fmt.Sprintf("%s dW: %s max=%.3g", dt, mDW.Band, mDW.MaxAbs))
		}
	}
	fmt.Printf("      worst_band=%s — want W exact_bit; dX/dW ≥strict on all 34\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

type mhVolCell struct {
	enabled    bool
	cfg        mha.Config
	dtype      core.DType
	q, k, v, o []float32
}

func runGoMHAVolNBP(depth, rows, cols, lpc, batch int, cells []mhVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := vc.cfg
		ml, err := mha.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.q, vc.k, vc.v, vc.o)
		if err != nil {
			return z, err
		}
		ml.Exec.Backend = core.BackendCPUTiled
		ml.Exec.MultiCore = false
		ml.Core.MultiCore = false
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerMultiHeadAttention, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: cfg.DModel, OutputHeight: cfg.DModel,
		}, ml); err != nil {
			return z, err
		}
	}
	// x layout [batch, seq, d]
	d := cells[0].cfg.DModel
	seq := len(x) / (batch * d)
	xin := core.NewTensor[float32](batch, seq, d)
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
		ml, ok := cell.Op.(*mha.Layer)
		if !ok {
			return z, fmt.Errorf("expected mha at %v", c)
		}
		wa, err := exportMHAWeights(ml)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), wa...))
	}
	return z, nil
}

func encodeMHAVolNBP(depth, rows, cols, lpc, batch int, cells []mhVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WHV1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0, 0} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	d := 0
	seq := 0
	for _, c := range cells {
		if c.enabled {
			d = c.cfg.DModel
			break
		}
	}
	if d > 0 {
		seq = len(x) / (batch * d)
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
		cfg := c.cfg
		_ = binary.Write(&in, binary.LittleEndian, int32(seq))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.DModel))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumHeads))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumKVHeads))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.HeadDim))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.MaxSeqLen))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Mask))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Pos))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.q {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.k {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.v {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range c.o {
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

func decodeMHAVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WHVO" {
		return z, bench, fmt.Errorf("cuda mha vol nbp bad magic")
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

func runCudaMHAVolNBP(depth, rows, cols, lpc, batch int, cells []mhVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	out, _, err := runCudaMHAVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, 0)
	return out, err
}

func runCudaMHAVolNBPBench(depth, rows, cols, lpc, batch int, cells []mhVolCell, x, target []float32, lr float64, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	payload, err := encodeMHAVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
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
		return nbpVolOut{}, 0, fmt.Errorf("cuda mha vol nbp: %v: %s", err, stderr.String())
	}
	return decodeMHAVolNBP(stdout.Bytes())
}

func benchMHAVolNBP(depth, rows, cols, lpc, batch int, cells []mhVolCell, x, target []float32, lr float64, benchIters int) (goUS, cudaUS, spdPct float64, err error) {
	goUS, err = benchGoUS(func() error {
		_, err := runGoMHAVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cudaDur, err := runCudaMHAVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cudaUS = perIterUS(cudaDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cudaUS)
	return goUS, cudaUS, spdPct, nil
}

func makeMHVolCell(cfg mha.Config, dt core.DType, seed uint32) (mhVolCell, error) {
	if err := cfg.Validate(); err != nil {
		return mhVolCell{}, err
	}
	qDim, kvDim, d := cfg.QDim(), cfg.KVDim(), cfg.DModel
	return mhVolCell{
		enabled: true, cfg: cfg, dtype: dt,
		q: randInit(qDim, d, seed),
		k: randInit(kvDim, d, seed+1),
		v: randInit(kvDim, d, seed+2),
		o: randInit(d, qDim, seed+3),
	}, nil
}

func caseMHANBPVol1x1x1() error {
	const batch, seq = 1, 4
	var benchIters = defaultGoBenchIters
	cfg := mha.DecoderCausal(16, 2, 2)
	cfg.MaxSeqLen = 16
	cell, err := makeMHVolCell(cfg, core.DTypeFloat32, 0x4d485631)
	if err != nil {
		return err
	}
	x := randInit(batch*seq, cfg.DModel, 0x4d485632)
	target := make([]float32, batch*seq*cfg.DModel)
	for i := range target {
		target[i] = float32(0.03 * float64(i%5+1))
	}
	cells := []mhVolCell{cell}
	const lr = 1e-2
	goO, err := runGoMHAVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaMHAVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchMHAVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseMHANBPVolChain() error {
	const batch, seq = 1, 3
	var benchIters = defaultGoBenchIters
	cfg := mha.DecoderCausal(16, 2, 2)
	cfg.MaxSeqLen = 16
	c0, err := makeMHVolCell(cfg, core.DTypeFloat32, 0x4d485633)
	if err != nil {
		return err
	}
	c1, err := makeMHVolCell(cfg, core.DTypeFloat32, 0x4d485634)
	if err != nil {
		return err
	}
	x := randInit(batch*seq, cfg.DModel, 0x4d485635)
	target := make([]float32, batch*seq*cfg.DModel)
	for i := range target {
		target[i] = float32(0.02 * float64(i%4+1))
	}
	cells := []mhVolCell{c0, c1}
	const lr = 1e-2
	goO, err := runGoMHAVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaMHAVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchMHAVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseMHANBPVol111AllDTypes() error {
	const batch, seq = 1, 3
	cfg := mha.DecoderCausal(16, 2, 2)
	cfg.MaxSeqLen = 16
	x := randInit(batch*seq, cfg.DModel, 0x4d485636)
	target := make([]float32, batch*seq*cfg.DModel)
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      MHA NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cell, err := makeMHVolCell(cfg, dt, 0x4d485637)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		cells := []mhVolCell{cell}
		goO, err := runGoMHAVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaMHAVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
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

func scaleMHA(q, k, v, o, x []float32, s float32) {
	for i := range q {
		q[i] *= s
	}
	for i := range k {
		k[i] *= s
	}
	for i := range v {
		v[i] *= s
	}
	for i := range o {
		o[i] *= s
	}
	for i := range x {
		x[i] *= s
	}
}

func caseMHANBPVol222F32() error {
	const batch, seq = 1, 2
	var benchIters = defaultGoBenchIters
	cfg := mha.DecoderCausal(16, 2, 2)
	cfg.MaxSeqLen = 32
	x := randInit(batch*seq, cfg.DModel, 0x4d485638)
	target := make([]float32, batch*seq*cfg.DModel)
	for i := range target {
		target[i] = float32(0.01 * float64(i%3+1))
	}
	cells := make([]mhVolCell, 8)
	for i := range cells {
		c, err := makeMHVolCell(cfg, core.DTypeFloat32, 0x4d485700+uint32(i))
		if err != nil {
			return err
		}
		// Scale down for 8-hop stability (attention product grows fast).
		scaleMHA(c.q, c.k, c.v, c.o, nil, 0.05)
		cells[i] = c
	}
	scaleMHA(nil, nil, nil, nil, x, 0.05)
	const lr = 1e-2
	goO, err := runGoMHAVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaMHAVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchMHAVolNBP(2, 2, 2, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseMHANBPVol211AllDTypes() error {
	const batch, seq = 1, 2
	cfg := mha.DecoderCausal(16, 2, 2)
	cfg.MaxSeqLen = 16
	x := randInit(batch*seq, cfg.DModel, 0x4d485639)
	for i := range x {
		x[i] *= 0.5
	}
	target := make([]float32, batch*seq*cfg.DModel)
	for i := range target {
		target[i] = float32(0.02 * float64(i%4+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      MHA NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		c0, err := makeMHVolCell(cfg, dt, 0x4d48563a)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		c1, err := makeMHVolCell(cfg, dt, 0x4d48563b)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		cells := []mhVolCell{c0, c1}
		goO, err := runGoMHAVolNBP(2, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaMHAVolNBP(2, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
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

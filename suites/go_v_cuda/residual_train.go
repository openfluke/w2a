package go_v_cuda

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
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesResidualTrain — NormalBP Residual F(x)+x Go↔CUDA (leaf + volumetric).
func CasesResidualTrain() []Case {
	return []Case{
		{Name: "Residual NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseReNBPLeafF32},
		{Name: "Residual NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseReNBPLeafAllDTypes},
		{Name: "Residual NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseReNBPVol1x1x1},
		{Name: "Residual NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseReNBPVolChain},
		{Name: "Residual NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseReNBPVol111AllDTypes},
		{Name: "Residual NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseReNBPVol222F32},
		{Name: "Residual NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseReNBPVol211AllDTypes},
	}
}

func RunResidualTrain() error {
	var fails []string
	cs := CasesResidualTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [renbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_residual_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_residual_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda residual_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type reNBPLeaf struct {
	batch  int
	seq    int // 1 → [B,D]
	cfg    residual.Config
	dtype  core.DType
	W      []float32
	x      []float32
	target []float32
	lr     float64
}

type reVolCell struct {
	enabled         bool
	dim, seq, depth int
	dtype           core.DType
	W               []float32
}

func smokeReCfg() residual.Config {
	return residual.Config{Dim: 6, SeqLen: 3, Depth: 3}
}

func volReCfg() residual.Config {
	return residual.Config{Dim: 4, SeqLen: 2, Depth: 2}
}

func disableResidualTrain(l *residual.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	for _, ch := range l.Children {
		if ch == nil {
			continue
		}
		ch.Exec.MultiCore = false
		ch.Core.MultiCore = false
		ch.Exec.Backend = be
	}
}

func packResidualWeightsF32(l *residual.Layer) ([]float32, error) {
	if l == nil {
		return nil, fmt.Errorf("residual: nil layer")
	}
	out := make([]float32, 0, l.Cfg.WeightCount())
	for i, ch := range l.Children {
		if ch == nil {
			return nil, fmt.Errorf("residual: nil child %d", i)
		}
		w, err := weights.MatrixF32(ch.Weights)
		if err != nil {
			return nil, err
		}
		out = append(out, w...)
	}
	return out, nil
}

func runGoReNBP(fc reNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	cfg := fc.cfg
	l, err := residual.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return z, err
	}
	disableResidualTrain(l)
	var xin *core.Tensor[float32]
	if fc.seq <= 1 {
		xin = core.NewTensor[float32](fc.batch, cfg.Dim)
	} else {
		xin = core.NewTensor[float32](fc.batch, fc.seq, cfg.Dim)
	}
	copy(xin.Data, fc.x)
	preT, postT, err := residual.Forward(l, xin)
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
	dX, dW, err := residual.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := residual.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packResidualWeightsF32(l)
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

func encodeReNBP(fc reNBPLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("REB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.Dim), int32(cfg.Depth),
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

func decodeReNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "REBO")
}

func runCudaReNBP(fc reNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCudaReNBPBench(fc, 0)
	return out, err
}

func runCudaReNBPBench(fc reNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeReNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("cuda residual nbp: %v: %s", err, stderr.String())
	}
	return decodeReNBP(stdout.Bytes())
}

func makeReLeafCase(cfg residual.Config, batch, seq int, dt core.DType, seed uint32) (reNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return reNBPLeaf{}, err
	}
	if seq <= 0 {
		seq = 1
	}
	fc := reNBPLeaf{
		batch: batch, seq: seq, cfg: cfg, dtype: dt, lr: 1e-2,
		W: randInit(1, cfg.WeightCount(), seed),
		x: randInit(batch*seq, cfg.Dim, seed^0x9e3779b9),
	}
	fc.target = make([]float32, batch*seq*cfg.Dim)
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

func caseReNBPLeafF32() error {
	var benchIters = defaultGoBenchIters
	cfg := smokeReCfg()
	fc, err := makeReLeafCase(cfg, 2, cfg.SeqLen, core.DTypeFloat32, 0x52454250)
	if err != nil {
		return err
	}
	goO, err := runGoReNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaReNBP(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoReNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cudaDur, err := runCudaReNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cuda bench: %w", err)
	}
	cudaUS := perIterUS(cudaDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cudaUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%  (neg ⇒ CUDA faster)",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseReNBPLeafAllDTypes() error {
	var benchIters = defaultGoBenchIters
	cfg := smokeReCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs CUDA — Residual NBP leaf ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cuda_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeReLeafCase(cfg, 2, cfg.SeqLen, dt, 0x52454434)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoReNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaReNBP(fc)
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
			_, err := runGoReNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cudaDur, err := runCudaReNBPBench(fc, benchIters)
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

func runGoReVolNBP(depth, rows, cols, lpc, batch int, cells []reVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := residual.Config{Dim: vc.dim, SeqLen: vc.seq, Depth: vc.depth}
		sl, err := residual.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableResidualTrain(sl)
		if err := residual.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	var xin *core.Tensor[float32]
	if firstSeq <= 1 {
		xin = core.NewTensor[float32](batch, firstDim)
	} else {
		xin = core.NewTensor[float32](batch, firstSeq, firstDim)
	}
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
		sl, ok := cell.Op.(*residual.Layer)
		if !ok {
			return z, fmt.Errorf("expected residual at %v", c)
		}
		wa, err := packResidualWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeReVolNBP(depth, rows, cols, lpc, batch int, cells []reVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("REV1"))
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

func decodeReVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "REVO")
}

func runCudaReVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findCUDAFwd()
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
		return nbpVolOut{}, 0, fmt.Errorf("cuda residual vol nbp: %v: %s", err, stderr.String())
	}
	return decodeReVolNBP(stdout.Bytes())
}

func benchReVolNBP(depth, rows, cols, lpc, batch int, cells []reVolCell, x, target []float32, lr float64, benchIters int) (goUS, cudaUS, spdPct float64, err error) {
	payload, err := encodeReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cudaDur, err := runCudaReVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cudaUS = perIterUS(cudaDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cudaUS)
	return goUS, cudaUS, spdPct, nil
}

func reVolPostLen(batch int, cfg residual.Config) int {
	seq := cfg.SeqLen
	if seq <= 0 {
		seq = 1
	}
	return batch * seq * cfg.Dim
}

func makeReVolCell(cfg residual.Config, dt core.DType, seed uint32) reVolCell {
	return reVolCell{
		enabled: true,
		dim:     cfg.Dim,
		seq:     cfg.SeqLen,
		depth:   cfg.Depth,
		dtype:   dt,
		W:       randInit(1, cfg.WeightCount(), seed),
	}
}

func caseReNBPVol1x1x1() error {
	const batch, lr = 2, 1e-2
	var benchIters = defaultGoBenchIters
	cfg := volReCfg()
	W := randInit(1, cfg.WeightCount(), 0x52455631)
	cells := []reVolCell{{
		enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, depth: cfg.Depth,
		dtype: core.DTypeFloat32, W: W,
	}}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x52455632)
	target := make([]float32, reVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoReVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeReVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	cudaO, _, err := runCudaReVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchReVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseReNBPVolChain() error {
	const batch, lr = 1, 1e-2
	var benchIters = defaultGoBenchIters
	cfg := volReCfg()
	W0 := randInit(1, cfg.WeightCount(), 0x52455633)
	W1 := randInit(1, cfg.WeightCount(), 0x52455634)
	cells := []reVolCell{
		{enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, depth: cfg.Depth, dtype: core.DTypeFloat32, W: W0},
		{enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, depth: cfg.Depth, dtype: core.DTypeFloat32, W: W1},
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x52455635)
	target := make([]float32, reVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoReVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeReVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	cudaO, _, err := runCudaReVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchReVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseReNBPVol111AllDTypes() error {
	const batch = 2
	cfg := volReCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Residual NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []reVolCell{makeReVolCell(cfg, dt, 0x52455636)}
		x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x52455637)
		target := make([]float32, reVolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%4+1))
		}
		goO, err := runGoReVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeReVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		cudaO, _, err := runCudaReVolNBP(payload, 0)
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

func caseReNBPVol222F32() error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	var benchIters = defaultGoBenchIters
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volReCfg()
	W := randInit(1, cfg.WeightCount(), 0x52455638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x52455639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, reVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]reVolCell, stack)
	for i := range cells {
		cells[i] = reVolCell{
			enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, depth: cfg.Depth,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	cudaO, _, err := runCudaReVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareVolNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, cudaUS, spdPct, err := benchReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseReNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	stack := depth * rows * cols * lpc
	cfg := volReCfg()
	W := randInit(1, cfg.WeightCount(), 0x5245563a)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x5245563b)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, reVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Residual NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := make([]reVolCell, stack)
		for i := range cells {
			cells[i] = reVolCell{
				enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, depth: cfg.Depth,
				dtype: dt, W: append([]float32(nil), W...),
			}
		}
		goO, err := runGoReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeReVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		cudaO, _, err := runCudaReVolNBP(payload, 0)
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

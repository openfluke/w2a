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
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesMetacognitionTrain — NormalBP Metacognition Observed Dense Go↔C++ (leaf + volumetric).
func CasesMetacognitionTrain() []Case {
	return []Case{
		{Name: "Metacognition NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseMcNBPLeafF32},
		{Name: "Metacognition NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseMcNBPLeafAllDTypes},
		{Name: "Metacognition NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseMcNBPVol1x1x1},
		{Name: "Metacognition NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseMcNBPVolChain},
		{Name: "Metacognition NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseMcNBPVol111AllDTypes},
		{Name: "Metacognition NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseMcNBPVol222F32},
		{Name: "Metacognition NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseMcNBPVol211AllDTypes},
	}
}

func RunMetacognitionTrain() error {
	var fails []string
	cs := CasesMetacognitionTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [mcnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_metacognition_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_metacognition_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp metacognition_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type mcNBPLeaf struct {
	batch  int
	seq    int // 1 → [B,D]
	cfg    metacognition.Config
	dtype  core.DType
	W      []float32
	x      []float32
	target []float32
	lr     float64
}

type mcVolCell struct {
	enabled  bool
	dim, seq int
	dtype    core.DType
	W        []float32
}

func smokeMcCfg() metacognition.Config {
	return metacognition.Config{Dim: 6, SeqLen: 3}
}

func volMcCfg() metacognition.Config {
	return metacognition.Config{Dim: 4, SeqLen: 2}
}

func mcWeightCount(cfg metacognition.Config) int {
	return cfg.Dim * cfg.Dim
}

func disableMetacognitionTrain(l *metacognition.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	if l.Observed != nil {
		l.Observed.Exec.MultiCore = false
		l.Observed.Core.MultiCore = false
		l.Observed.Exec.Backend = be
	}
}

func packMetacognitionWeightsF32(l *metacognition.Layer) ([]float32, error) {
	if l == nil || l.Observed == nil {
		return nil, fmt.Errorf("metacognition: nil layer")
	}
	return weights.MatrixF32(l.Observed.Weights)
}

func runGoMcNBP(fc mcNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	cfg := fc.cfg
	l, err := metacognition.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return z, err
	}
	disableMetacognitionTrain(l)
	var xin *core.Tensor[float32]
	if fc.seq <= 1 {
		xin = core.NewTensor[float32](fc.batch, cfg.Dim)
	} else {
		xin = core.NewTensor[float32](fc.batch, fc.seq, cfg.Dim)
	}
	copy(xin.Data, fc.x)
	preT, postT, err := metacognition.Forward(l, xin)
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
	dX, dW, err := metacognition.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := metacognition.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packMetacognitionWeightsF32(l)
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

func encodeMcNBP(fc mcNBPLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("MCB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.Dim), int32(fc.dtype), 0,
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

func decodeMcNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "MCBO")
}

func runCppMcNBP(fc mcNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCppMcNBPBench(fc, 0)
	return out, err
}

func runCppMcNBPBench(fc mcNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeMcNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("cpp metacognition nbp: %v: %s", err, stderr.String())
	}
	return decodeMcNBP(stdout.Bytes())
}

func makeMcLeafCase(cfg metacognition.Config, batch, seq int, dt core.DType, seed uint32) (mcNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return mcNBPLeaf{}, err
	}
	if seq <= 0 {
		seq = 1
	}
	fc := mcNBPLeaf{
		batch: batch, seq: seq, cfg: cfg, dtype: dt, lr: 1e-2,
		W: randInit(1, mcWeightCount(cfg), seed),
		x: randInit(batch*seq, cfg.Dim, seed^0x9e3779b9),
	}
	fc.target = make([]float32, batch*seq*cfg.Dim)
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

func caseMcNBPLeafF32() error {
	const benchIters = defaultGoBenchIters
	cfg := smokeMcCfg()
	fc, err := makeMcLeafCase(cfg, 2, cfg.SeqLen, core.DTypeFloat32, 0x4d434250)
	if err != nil {
		return err
	}
	goO, err := runGoMcNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppMcNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoMcNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cppDur, err := runCppMcNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	cppUS := perIterUS(cppDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%  (neg ⇒ C++ faster)",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMcNBPLeafAllDTypes() error {
	const benchIters = defaultGoBenchIters
	cfg := smokeMcCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — Metacognition NBP leaf ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeMcLeafCase(cfg, 2, cfg.SeqLen, dt, 0x4d434434)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoMcNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppMcNBP(fc)
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
			_, err := runGoMcNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cppDur, err := runCppMcNBPBench(fc, benchIters)
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
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func runGoMcVolNBP(depth, rows, cols, lpc, batch int, cells []mcVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := metacognition.Config{Dim: vc.dim, SeqLen: vc.seq}
		sl, err := metacognition.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableMetacognitionTrain(sl)
		if err := metacognition.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
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
		sl, ok := cell.Op.(*metacognition.Layer)
		if !ok {
			return z, fmt.Errorf("expected metacognition at %v", c)
		}
		wa, err := packMetacognitionWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeMcVolNBP(depth, rows, cols, lpc, batch int, cells []mcVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("MCV1"))
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

func decodeMcVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "MCVO")
}

func runCppMcVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp metacognition vol nbp: %v: %s", err, stderr.String())
	}
	return decodeMcVolNBP(stdout.Bytes())
}

func benchMcVolNBP(depth, rows, cols, lpc, batch int, cells []mcVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cppDur, err := runCppMcVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cppUS = perIterUS(cppDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func mcVolPostLen(batch int, cfg metacognition.Config) int {
	seq := cfg.SeqLen
	if seq <= 0 {
		seq = 1
	}
	return batch * seq * cfg.Dim
}

func makeMcVolCell(cfg metacognition.Config, dt core.DType, seed uint32) mcVolCell {
	return mcVolCell{
		enabled: true,
		dim:     cfg.Dim,
		seq:     cfg.SeqLen,
		dtype:   dt,
		W:       randInit(1, mcWeightCount(cfg), seed),
	}
}

func caseMcNBPVol1x1x1() error {
	const batch, benchIters, lr = 2, defaultGoBenchIters, 1e-2
	cfg := volMcCfg()
	W := randInit(1, mcWeightCount(cfg), 0x4d435631)
	cells := []mcVolCell{{
		enabled: true, dim: cfg.Dim, seq: cfg.SeqLen,
		dtype: core.DTypeFloat32, W: W,
	}}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x4d435632)
	target := make([]float32, mcVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoMcVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeMcVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	cppO, _, err := runCppMcVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchMcVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMcNBPVolChain() error {
	const batch, benchIters, lr = 1, defaultGoBenchIters, 1e-2
	cfg := volMcCfg()
	W0 := randInit(1, mcWeightCount(cfg), 0x4d435633)
	W1 := randInit(1, mcWeightCount(cfg), 0x4d435634)
	cells := []mcVolCell{
		{enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, dtype: core.DTypeFloat32, W: W0},
		{enabled: true, dim: cfg.Dim, seq: cfg.SeqLen, dtype: core.DTypeFloat32, W: W1},
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x4d435635)
	target := make([]float32, mcVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoMcVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeMcVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	cppO, _, err := runCppMcVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchMcVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMcNBPVol111AllDTypes() error {
	const batch = 2
	cfg := volMcCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Metacognition NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []mcVolCell{makeMcVolCell(cfg, dt, 0x4d435636)}
		x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x4d435637)
		target := make([]float32, mcVolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%4+1))
		}
		goO, err := runGoMcVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeMcVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppMcVolNBP(payload, 0)
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

func caseMcNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volMcCfg()
	W := randInit(1, mcWeightCount(cfg), 0x4d435638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x4d435639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, mcVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]mcVolCell, stack)
	for i := range cells {
		cells[i] = mcVolCell{
			enabled: true, dim: cfg.Dim, seq: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	cppO, _, err := runCppMcVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMcNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	stack := depth * rows * cols * lpc
	cfg := volMcCfg()
	W := randInit(1, mcWeightCount(cfg), 0x4d43563a)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.Dim, 0x4d43563b)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, mcVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Metacognition NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := make([]mcVolCell, stack)
		for i := range cells {
			cells[i] = mcVolCell{
				enabled: true, dim: cfg.Dim, seq: cfg.SeqLen,
				dtype: dt, W: append([]float32(nil), W...),
			}
		}
		goO, err := runGoMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeMcVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppMcVolNBP(payload, 0)
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

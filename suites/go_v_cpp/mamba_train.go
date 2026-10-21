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
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesMambaTrain — NormalBP Mamba Go↔C++ (leaf + volumetric).
func CasesMambaTrain() []Case {
	return []Case{
		{Name: "Mamba NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseMbNBPLeafF32},
		{Name: "Mamba NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseMbNBPLeafAllDTypes},
		{Name: "Mamba NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseMbNBPVol1x1x1},
		{Name: "Mamba NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseMbNBPVolChain},
		{Name: "Mamba NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseMbNBPVol111AllDTypes},
		{Name: "Mamba NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseMbNBPVol222F32},
		{Name: "Mamba NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseMbNBPVol211AllDTypes},
	}
}

func RunMambaTrain() error {
	var fails []string
	cs := CasesMambaTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [mbnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_mamba_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_mamba_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp mamba_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type mbNBPLeaf struct {
	batch  int
	cfg    mamba.Config
	dtype  core.DType
	W      []float32
	x      []float32
	target []float32
	lr     float64
}

type mbVolCell struct {
	enabled                        bool
	dModel, dState, expand, seqLen int
	dtype                          core.DType
	W                              []float32
}

func smokeMbCfg() mamba.Config {
	return mamba.Config{DModel: 6, DState: 4, Expand: 2, SeqLen: 3}
}

func volMbCfg() mamba.Config {
	return mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
}

func disableMambaTrain(l *mamba.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	if l.InProj != nil {
		l.InProj.Exec.MultiCore = false
		l.InProj.Core.MultiCore = false
		l.InProj.Exec.Backend = be
	}
	if l.OutProj != nil {
		l.OutProj.Exec.MultiCore = false
		l.OutProj.Core.MultiCore = false
		l.OutProj.Exec.Backend = be
	}
}

func packMambaWeightsF32(l *mamba.Layer) ([]float32, error) {
	if l == nil || l.InProj == nil || l.OutProj == nil {
		return nil, fmt.Errorf("mamba: nil layer")
	}
	inW, err := weights.MatrixF32(l.InProj.Weights)
	if err != nil {
		return nil, err
	}
	outW, err := weights.MatrixF32(l.OutProj.Weights)
	if err != nil {
		return nil, err
	}
	out := make([]float32, 0, len(inW)+len(outW)+len(l.ALog)+len(l.DSkip))
	out = append(out, inW...)
	out = append(out, outW...)
	out = append(out, l.ALog...)
	out = append(out, l.DSkip...)
	return out, nil
}

func runGoMbNBP(fc mbNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	cfg := fc.cfg
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, fc.W)
	if err != nil {
		return z, err
	}
	l, err := mamba.NewConfigured(cfg, fc.dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return z, err
	}
	disableMambaTrain(l)
	xin := core.NewTensor[float32](fc.batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, fc.x)
	preT, postT, err := mamba.Forward(l, xin)
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
	dX, dW, err := mamba.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := mamba.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packMambaWeightsF32(l)
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

func encodeMbNBP(fc mbNBPLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("MBB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.SeqLen), int32(cfg.DModel), int32(cfg.DState),
		int32(cfg.Expand), int32(fc.dtype), 0,
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

func decodeMbNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "MBBO")
}

func runCppMbNBP(fc mbNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCppMbNBPBench(fc, 0)
	return out, err
}

func runCppMbNBPBench(fc mbNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeMbNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("cpp mamba nbp: %v: %s", err, stderr.String())
	}
	return decodeMbNBP(stdout.Bytes())
}

func makeMbLeafCase(cfg mamba.Config, batch int, dt core.DType, seed uint32) (mbNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return mbNBPLeaf{}, err
	}
	fc := mbNBPLeaf{
		batch: batch, cfg: cfg, dtype: dt, lr: 1e-2,
		W: randInit(1, mambaWeightCount(cfg), seed),
		x: randInit(batch*cfg.SeqLen, cfg.DModel, seed^0x9e3779b9),
	}
	fc.target = make([]float32, batch*cfg.SeqLen*cfg.DModel)
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

func caseMbNBPLeafF32() error {
	const benchIters = defaultGoBenchIters
	cfg := smokeMbCfg()
	fc, err := makeMbLeafCase(cfg, 2, core.DTypeFloat32, 0x4d424250)
	if err != nil {
		return err
	}
	goO, err := runGoMbNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppMbNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoMbNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cppDur, err := runCppMbNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	cppUS := perIterUS(cppDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%  (neg ⇒ C++ faster)",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMbNBPLeafAllDTypes() error {
	const benchIters = defaultGoBenchIters
	cfg := smokeMbCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — Mamba NBP leaf ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeMbLeafCase(cfg, 2, dt, 0x4d424434)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoMbNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppMbNBP(fc)
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
			_, err := runGoMbNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cppDur, err := runCppMbNBPBench(fc, benchIters)
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

func runGoMbVolNBP(depth, rows, cols, lpc, batch int, cells []mbVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstD, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstD < 0 {
			firstD, firstSeq = vc.dModel, vc.seqLen
		}
		cfg := mamba.Config{DModel: vc.dModel, DState: vc.dState, Expand: vc.expand, SeqLen: vc.seqLen}
		inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, vc.W)
		if err != nil {
			return z, err
		}
		ml, err := mamba.NewConfigured(cfg, vc.dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
		if err != nil {
			return z, err
		}
		disableMambaTrain(ml)
		if err := mamba.Place(g, c.Z, c.Y, c.X, c.L, ml); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstD)
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
		ml, ok := cell.Op.(*mamba.Layer)
		if !ok {
			return z, fmt.Errorf("expected mamba at %v", c)
		}
		wa, err := packMambaWeightsF32(ml)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeMbVolNBP(depth, rows, cols, lpc, batch int, cells []mbVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("MBV1"))
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dModel))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dState))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.expand))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.seqLen))
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

func decodeMbVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "MBVO")
}

func runCppMbVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp mamba vol nbp: %v: %s", err, stderr.String())
	}
	return decodeMbVolNBP(stdout.Bytes())
}

func benchMbVolNBP(depth, rows, cols, lpc, batch int, cells []mbVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cppDur, err := runCppMbVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cppUS = perIterUS(cppDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func mbVolPostLen(batch int, cfg mamba.Config) int {
	return batch * cfg.SeqLen * cfg.DModel
}

func makeMbVolCell(cfg mamba.Config, dt core.DType, seed uint32) mbVolCell {
	return mbVolCell{
		enabled: true,
		dModel:  cfg.DModel,
		dState:  cfg.DState,
		expand:  cfg.Expand,
		seqLen:  cfg.SeqLen,
		dtype:   dt,
		W:       randInit(1, mambaWeightCount(cfg), seed),
	}
}

func caseMbNBPVol1x1x1() error {
	const batch, benchIters, lr = 2, defaultGoBenchIters, 1e-2
	cfg := volMbCfg()
	W := randInit(1, mambaWeightCount(cfg), 0x4d425631)
	cells := []mbVolCell{{
		enabled: true, dModel: cfg.DModel, dState: cfg.DState, expand: cfg.Expand, seqLen: cfg.SeqLen,
		dtype: core.DTypeFloat32, W: W,
	}}
	x := randInit(batch*cfg.SeqLen, cfg.DModel, 0x4d425632)
	target := make([]float32, mbVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoMbVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeMbVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	cppO, _, err := runCppMbVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchMbVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMbNBPVolChain() error {
	const batch, benchIters, lr = 1, defaultGoBenchIters, 1e-2
	cfg := volMbCfg()
	W0 := randInit(1, mambaWeightCount(cfg), 0x4d425633)
	W1 := randInit(1, mambaWeightCount(cfg), 0x4d425634)
	cells := []mbVolCell{
		{enabled: true, dModel: cfg.DModel, dState: cfg.DState, expand: cfg.Expand, seqLen: cfg.SeqLen, dtype: core.DTypeFloat32, W: W0},
		{enabled: true, dModel: cfg.DModel, dState: cfg.DState, expand: cfg.Expand, seqLen: cfg.SeqLen, dtype: core.DTypeFloat32, W: W1},
	}
	x := randInit(batch*cfg.SeqLen, cfg.DModel, 0x4d425635)
	target := make([]float32, mbVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoMbVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeMbVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	cppO, _, err := runCppMbVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchMbVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMbNBPVol111AllDTypes() error {
	const batch = 2
	cfg := volMbCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Mamba NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []mbVolCell{makeMbVolCell(cfg, dt, 0x4d425636)}
		x := randInit(batch*cfg.SeqLen, cfg.DModel, 0x4d425637)
		target := make([]float32, mbVolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%4+1))
		}
		goO, err := runGoMbVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeMbVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppMbVolNBP(payload, 0)
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

func caseMbNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volMbCfg()
	W := randInit(1, mambaWeightCount(cfg), 0x4d425638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.DModel, 0x4d425639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, mbVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]mbVolCell, stack)
	for i := range cells {
		cells[i] = mbVolCell{
			enabled: true, dModel: cfg.DModel, dState: cfg.DState, expand: cfg.Expand, seqLen: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	cppO, _, err := runCppMbVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseMbNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	stack := depth * rows * cols * lpc
	cfg := volMbCfg()
	W := randInit(1, mambaWeightCount(cfg), 0x4d42563a)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.DModel, 0x4d42563b)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, mbVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Mamba NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := make([]mbVolCell, stack)
		for i := range cells {
			cells[i] = mbVolCell{
				enabled: true, dModel: cfg.DModel, dState: cfg.DState, expand: cfg.Expand, seqLen: cfg.SeqLen,
				dtype: dt, W: append([]float32(nil), W...),
			}
		}
		goO, err := runGoMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeMbVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppMbVolNBP(payload, 0)
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

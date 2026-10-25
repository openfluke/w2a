package go_v_metal

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
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesLSTMTrain — NormalBP LSTM Go↔Metal (leaf + volumetric).
func CasesLSTMTrain() []Case {
	return []Case{
		{Name: "LSTM NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseLSTMNBPLeafF32},
		{Name: "LSTM NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseLSTMNBPLeafAllDTypes},
		{Name: "LSTM NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseLSTMNBPVol1x1x1},
		{Name: "LSTM NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseLSTMNBPVolChain},
		{Name: "LSTM NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseLSTMNBPVol111AllDTypes},
		{Name: "LSTM NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMNBPVol222F32},
		{Name: "LSTM NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseLSTMNBPVol211AllDTypes},
	}
}

func RunLSTMTrain() error {
	var fails []string
	cs := CasesLSTMTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [lsnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_lstm_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_lstm_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal lstm_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type lstmNBPLeaf struct {
	batch  int
	cfg    lstm.Config
	dtype  core.DType
	W      []float32
	x      []float32
	target []float32
	lr     float64
}

type lstmVolCell struct {
	enabled      bool
	in, hid, seq int
	dtype        core.DType
	W            []float32
}

func smokeLSTMCfg() lstm.Config {
	return lstm.Config{InputSize: 4, HiddenSize: 4, SeqLen: 3}
}

func volLSTMCfg() lstm.Config {
	return lstm.Config{InputSize: 3, HiddenSize: 3, SeqLen: 2}
}

func disableLSTMTrainGates(l *lstm.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	for _, g := range []*lstm.Gate{l.I, l.F, l.G, l.O} {
		if g == nil {
			continue
		}
		if g.IH != nil {
			g.IH.Exec.MultiCore = false
			g.IH.Core.MultiCore = false
			g.IH.Exec.Backend = be
		}
		if g.HH != nil {
			g.HH.Exec.MultiCore = false
			g.HH.Core.MultiCore = false
			g.HH.Exec.Backend = be
		}
	}
}

func packLSTMGateF32(g *lstm.Gate, hid, in int) ([]float32, error) {
	if g == nil || g.IH == nil || g.HH == nil {
		return nil, fmt.Errorf("lstm: nil gate")
	}
	ihW, err := weights.MatrixF32(g.IH.Weights)
	if err != nil {
		return nil, err
	}
	hhW, err := weights.MatrixF32(g.HH.Weights)
	if err != nil {
		return nil, err
	}
	ihN, hhN := hid*in, hid*hid
	out := make([]float32, ihN+hhN+hid)
	copy(out[:ihN], ihW)
	copy(out[ihN:ihN+hhN], hhW)
	for i := 0; i < hid; i++ {
		out[ihN+hhN+i] = float32(g.IH.Weights.Bias[i])
	}
	return out, nil
}

func packLSTMWeightsF32(l *lstm.Layer) ([]float32, error) {
	if l == nil {
		return nil, fmt.Errorf("lstm: nil layer")
	}
	h, in := l.Cfg.HiddenSize, l.Cfg.InputSize
	gateN := l.Cfg.GateSize()
	out := make([]float32, l.Cfg.WeightCount())
	for i, g := range []*lstm.Gate{l.I, l.F, l.G, l.O} {
		pack, err := packLSTMGateF32(g, h, in)
		if err != nil {
			return nil, fmt.Errorf("lstm gate %d: %w", i, err)
		}
		copy(out[i*gateN:(i+1)*gateN], pack)
	}
	return out, nil
}

func runGoLSTMNBP(fc lstmNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	cfg := fc.cfg
	l, err := lstm.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return z, err
	}
	disableLSTMTrainGates(l)
	xin := core.NewTensor[float32](fc.batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, fc.x)
	preT, postT, err := lstm.Forward(l, xin)
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
	dX, dW, err := lstm.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := lstm.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packLSTMWeightsF32(l)
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

func encodeLSTMNBP(fc lstmNBPLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("LSB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(cfg.InputSize), int32(cfg.HiddenSize), int32(cfg.SeqLen),
		int32(fc.dtype), 0,
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

func decodeLSTMNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "LSBO")
}

func runMetalLSTMNBP(fc lstmNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runMetalLSTMNBPBench(fc, 0)
	return out, err
}

func runMetalLSTMNBPBench(fc lstmNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findMetalFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeLSTMNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("metal lstm nbp: %v: %s", err, stderr.String())
	}
	return decodeLSTMNBP(stdout.Bytes())
}

func makeLSTMLeafCase(cfg lstm.Config, batch int, dt core.DType, seed uint32) (lstmNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return lstmNBPLeaf{}, err
	}
	fc := lstmNBPLeaf{
		batch: batch, cfg: cfg, dtype: dt, lr: 1e-2,
		W: randInit(1, cfg.WeightCount(), seed),
		x: randInit(batch*cfg.SeqLen, cfg.InputSize, seed^0x9e3779b9),
	}
	fc.target = make([]float32, batch*cfg.SeqLen*cfg.HiddenSize)
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

func caseLSTMNBPLeafF32() error {
	const benchIters = defaultGoBenchIters
	cfg := smokeLSTMCfg()
	fc, err := makeLSTMLeafCase(cfg, 2, core.DTypeFloat32, 0x4c534250)
	if err != nil {
		return err
	}
	goO, err := runGoLSTMNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	mtlO, err := runMetalLSTMNBP(fc)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoLSTMNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, mtlDur, err := runMetalLSTMNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("metal bench: %w", err)
	}
	mtlUS := perIterUS(mtlDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, mtlUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%  (neg ⇒ Metal faster)",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseLSTMNBPLeafAllDTypes() error {
	const benchIters = defaultGoBenchIters
	cfg := smokeLSTMCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs Metal — LSTM NBP leaf ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "mtl_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeLSTMLeafCase(cfg, 2, dt, 0x4c534434)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoLSTMNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		mtlO, err := runMetalLSTMNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			continue
		}
		mW := suites.CompareF32Spectrum(goO.WAfter, mtlO.WAfter)
		mDX := suites.CompareF32Spectrum(goO.dX, mtlO.dX)
		mDW := suites.CompareF32Spectrum(goO.dW, mtlO.dW)
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
			_, err := runGoLSTMNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, mtlDur, err := runMetalLSTMNBPBench(fc, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %9s %s\n",
				dt, band, mW.ExactPct, FormatUS(goUS), "?", "?", fmt.Sprintf("%.3g", mW.MaxAbs))
			continue
		}
		mtlUS := perIterUS(mtlDur, benchIters)
		spdPct := suites.SpeedPctDiff(goUS, mtlUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
			dt, band, mW.ExactPct, FormatUS(goUS), FormatUS(mtlUS), spdPct, fmt.Sprintf("%.3g", mW.MaxAbs))
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

func runGoLSTMVolNBP(depth, rows, cols, lpc, batch int, cells []lstmVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	var z nbpVolOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	hops := g.HopOrder()
	if len(cells) != len(hops) {
		return z, fmt.Errorf("cells %d != hops %d", len(cells), len(hops))
	}
	firstIn, firstSeq := -1, -1
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			cell := g.At(c.Z, c.Y, c.X, c.L)
			if cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		if firstIn < 0 {
			firstIn, firstSeq = vc.in, vc.seq
		}
		cfg := lstm.Config{InputSize: vc.in, HiddenSize: vc.hid, SeqLen: vc.seq}
		ll, err := lstm.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableLSTMTrainGates(ll)
		if err := lstm.Place(g, c.Z, c.Y, c.X, c.L, ll); err != nil {
			return z, err
		}
	}
	xin := core.NewTensor[float32](batch, firstSeq, firstIn)
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
		ll, ok := cell.Op.(*lstm.Layer)
		if !ok {
			return z, fmt.Errorf("expected lstm at %v", c)
		}
		wa, err := packLSTMWeightsF32(ll)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeLSTMVolNBP(depth, rows, cols, lpc, batch int, cells []lstmVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("LSV1"))
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.in))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.hid))
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

func decodeLSTMVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "LSVO")
}

func runMetalLSTMVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findMetalFwd()
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
		return nbpVolOut{}, 0, fmt.Errorf("metal lstm vol nbp: %v: %s", err, stderr.String())
	}
	return decodeLSTMVolNBP(stdout.Bytes())
}

func benchLSTMVolNBP(depth, rows, cols, lpc, batch int, cells []lstmVolCell, x, target []float32, lr float64, benchIters int) (goUS, mtlUS, spdPct float64, err error) {
	payload, err := encodeLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, mtlDur, err := runMetalLSTMVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	mtlUS = perIterUS(mtlDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, mtlUS)
	return goUS, mtlUS, spdPct, nil
}

func lstmVolPostLen(batch int, cfg lstm.Config) int {
	return batch * cfg.SeqLen * cfg.HiddenSize
}

func makeLSTMVolCell(cfg lstm.Config, dt core.DType, seed uint32) lstmVolCell {
	return lstmVolCell{
		enabled: true,
		in:      cfg.InputSize,
		hid:     cfg.HiddenSize,
		seq:     cfg.SeqLen,
		dtype:   dt,
		W:       randInit(1, cfg.WeightCount(), seed),
	}
}

func caseLSTMNBPVol1x1x1() error {
	const batch, benchIters, lr = 2, defaultGoBenchIters, 1e-2
	cfg := volLSTMCfg()
	W := randInit(1, cfg.WeightCount(), 0x4c535631)
	cells := []lstmVolCell{{
		enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
		dtype: core.DTypeFloat32, W: W,
	}}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x4c535632)
	target := make([]float32, lstmVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoLSTMVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeLSTMVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	mtlO, _, err := runMetalLSTMVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareVolNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, mtlUS, spdPct, err := benchLSTMVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseLSTMNBPVolChain() error {
	const batch, benchIters, lr = 1, defaultGoBenchIters, 1e-2
	cfg := volLSTMCfg()
	W0 := randInit(1, cfg.WeightCount(), 0x4c535633)
	W1 := randInit(1, cfg.WeightCount(), 0x4c535634)
	cells := []lstmVolCell{
		{enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen, dtype: core.DTypeFloat32, W: W0},
		{enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen, dtype: core.DTypeFloat32, W: W1},
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x4c535635)
	target := make([]float32, lstmVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoLSTMVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeLSTMVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	mtlO, _, err := runMetalLSTMVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareVolNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, mtlUS, spdPct, err := benchLSTMVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseLSTMNBPVol111AllDTypes() error {
	const batch = 2
	cfg := volLSTMCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      LSTM NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []lstmVolCell{makeLSTMVolCell(cfg, dt, 0x4c535636)}
		x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x4c535637)
		target := make([]float32, lstmVolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%4+1))
		}
		goO, err := runGoLSTMVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeLSTMVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		mtlO, _, err := runMetalLSTMVolNBP(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, mtlO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], mtlO.Ws[0])
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

func caseLSTMNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volLSTMCfg()
	W := randInit(1, cfg.WeightCount(), 0x4c535638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x4c535639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, lstmVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]lstmVolCell, stack)
	for i := range cells {
		cells[i] = lstmVolCell{
			enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	mtlO, _, err := runMetalLSTMVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("metal: %w", err)
	}
	if err := compareVolNBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, mtlUS, spdPct, err := benchLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(mtlUS), spdPct)
	return nil
}

func caseLSTMNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	stack := depth * rows * cols * lpc
	cfg := volLSTMCfg()
	W := randInit(1, cfg.WeightCount(), 0x4c53563a)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x4c53563b)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, lstmVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      LSTM NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := make([]lstmVolCell, stack)
		for i := range cells {
			cells[i] = lstmVolCell{
				enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
				dtype: dt, W: append([]float32(nil), W...),
			}
		}
		goO, err := runGoLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeLSTMVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		mtlO, _, err := runMetalLSTMVolNBP(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s metal: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, mtlO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.Ws[0], mtlO.Ws[0])
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

package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesRNNTrain — NormalBP RNN Go↔C++ (leaf + volumetric).
func CasesRNNTrain() []Case {
	return []Case{
		{Name: "RNN NormalBP leaf Float32 — exact bits (dX/dW/W/loss)", Run: caseRNNNBPLeafF32},
		{Name: "RNN NormalBP leaf × all 34 dtypes — spectrum + speed", Run: caseRNNNBPLeafAllDTypes},
		{Name: "RNN NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseRNNNBPVol1x1x1},
		{Name: "RNN NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseRNNNBPVolChain},
		{Name: "RNN NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseRNNNBPVol111AllDTypes},
		{Name: "RNN NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseRNNNBPVol222F32},
		{Name: "RNN NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseRNNNBPVol211AllDTypes},
	}
}

func RunRNNTrain() error {
	var fails []string
	cs := CasesRNNTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [rnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_rnn_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_rnn_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp rnn_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type rnnNBPLeaf struct {
	batch int
	cfg   rnn.Config
	dtype core.DType
	W     []float32
	x     []float32
	target []float32
	lr    float64
}

type rnnVolCell struct {
	enabled bool
	in, hid, seq int
	dtype   core.DType
	W       []float32
}

func smokeRNNCfg() rnn.Config {
	return rnn.Config{InputSize: 4, HiddenSize: 4, SeqLen: 3}
}

func volRNNCfg() rnn.Config {
	return rnn.Config{InputSize: 3, HiddenSize: 3, SeqLen: 2}
}

func disableRNN(l *rnn.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	if l.IH != nil {
		l.IH.Exec.MultiCore = false
		l.IH.Core.MultiCore = false
		l.IH.Exec.Backend = be
	}
	if l.HH != nil {
		l.HH.Exec.MultiCore = false
		l.HH.Core.MultiCore = false
		l.HH.Exec.Backend = be
	}
}

func packRNNWeightsF32(l *rnn.Layer) ([]float32, error) {
	if l == nil || l.IH == nil || l.HH == nil {
		return nil, fmt.Errorf("rnn: nil layer")
	}
	ihW, err := weights.MatrixF32(l.IH.Weights)
	if err != nil {
		return nil, err
	}
	hhW, err := weights.MatrixF32(l.HH.Weights)
	if err != nil {
		return nil, err
	}
	h := l.Cfg.HiddenSize
	out := make([]float32, l.Cfg.WeightCount())
	copy(out[:len(ihW)], ihW)
	copy(out[len(ihW):len(ihW)+len(hhW)], hhW)
	for i := 0; i < h; i++ {
		out[len(ihW)+len(hhW)+i] = float32(l.IH.Weights.Bias[i])
	}
	return out, nil
}

func runGoRNNNBP(fc rnnNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	cfg := fc.cfg
	l, err := rnn.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return z, err
	}
	disableRNN(l)
	xin := core.NewTensor[float32](fc.batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, fc.x)
	preT, postT, err := rnn.Forward(l, xin)
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
	dX, dW, err := rnn.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := rnn.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packRNNWeightsF32(l)
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

func encodeRNNNBP(fc rnnNBPLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("RNB1"))
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

func decodeRecNBPLeaf(raw []byte, respMagic string) (nbpLeafOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != respMagic {
		return z, bench, fmt.Errorf("cpp rnn/lstm nbp bad magic %q", string(raw[:min(4, len(raw))]))
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

func decodeRNNNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "RNBO")
}

func runCppRNNNBP(fc rnnNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCppRNNNBPBench(fc, 0)
	return out, err
}

func runCppRNNNBPBench(fc rnnNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeRNNNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("cpp rnn nbp: %v: %s", err, stderr.String())
	}
	return decodeRNNNBP(stdout.Bytes())
}

func makeRNNLeafCase(cfg rnn.Config, batch int, dt core.DType, seed uint32) (rnnNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return rnnNBPLeaf{}, err
	}
	fc := rnnNBPLeaf{
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

func caseRNNNBPLeafF32() error {
	const benchIters = 40
	cfg := smokeRNNCfg()
	fc, err := makeRNNLeafCase(cfg, 2, core.DTypeFloat32, 0x524e4250)
	if err != nil {
		return err
	}
	goO, err := runGoRNNNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppRNNNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoRNNNBP(fc); err != nil {
			return err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoRNNNBP(fc); err != nil {
			return err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppRNNNBPBench(fc, benchIters)
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

func caseRNNNBPLeafAllDTypes() error {
	const benchIters = 20
	cfg := smokeRNNCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — RNN NBP leaf ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeRNNLeafCase(cfg, 2, dt, 0x524e4434)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoRNNNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppRNNNBP(fc)
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
			if _, err := runGoRNNNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go warm: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, err := runGoRNNNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, cppDur, err := runCppRNNNBPBench(fc, benchIters)
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

func runGoRNNVolNBP(depth, rows, cols, lpc, batch int, cells []rnnVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cfg := rnn.Config{InputSize: vc.in, HiddenSize: vc.hid, SeqLen: vc.seq}
		rl, err := rnn.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableRNN(rl)
		if err := rnn.Place(g, c.Z, c.Y, c.X, c.L, rl); err != nil {
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
		rl, ok := cell.Op.(*rnn.Layer)
		if !ok {
			return z, fmt.Errorf("expected rnn at %v", c)
		}
		wa, err := packRNNWeightsF32(rl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeRNNVolNBP(depth, rows, cols, lpc, batch int, cells []rnnVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("RNV1"))
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

func decodeRecVolNBP(raw []byte, respMagic string) (nbpVolOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != respMagic {
		return z, bench, fmt.Errorf("cpp rnn/lstm vol nbp bad magic")
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

func decodeRNNVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "RNVO")
}

func runCppRNNVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp rnn vol nbp: %v: %s", err, stderr.String())
	}
	return decodeRNNVolNBP(stdout.Bytes())
}

func benchRNNVolNBP(depth, rows, cols, lpc, batch int, cells []rnnVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppRNNVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func rnnVolPostLen(batch int, cfg rnn.Config) int {
	return batch * cfg.SeqLen * cfg.HiddenSize
}

func makeRNNVolCell(cfg rnn.Config, dt core.DType, seed uint32) rnnVolCell {
	return rnnVolCell{
		enabled: true,
		in:      cfg.InputSize,
		hid:     cfg.HiddenSize,
		seq:     cfg.SeqLen,
		dtype:   dt,
		W:       randInit(1, cfg.WeightCount(), seed),
	}
}

func caseRNNNBPVol1x1x1() error {
	const batch, benchIters, lr = 2, 40, 1e-2
	cfg := volRNNCfg()
	W := randInit(1, cfg.WeightCount(), 0x524e5631)
	cells := []rnnVolCell{{
		enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
		dtype: core.DTypeFloat32, W: W,
	}}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x524e5632)
	target := make([]float32, rnnVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoRNNVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeRNNVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	cppO, _, err := runCppRNNVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchRNNVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseRNNNBPVolChain() error {
	const batch, benchIters, lr = 1, 40, 1e-2
	cfg := volRNNCfg()
	W0 := randInit(1, cfg.WeightCount(), 0x524e5633)
	W1 := randInit(1, cfg.WeightCount(), 0x524e5634)
	cells := []rnnVolCell{
		{enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen, dtype: core.DTypeFloat32, W: W0},
		{enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen, dtype: core.DTypeFloat32, W: W1},
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x524e5635)
	target := make([]float32, rnnVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoRNNVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeRNNVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	cppO, _, err := runCppRNNVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchRNNVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseRNNNBPVol111AllDTypes() error {
	const batch = 2
	cfg := volRNNCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      RNN NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []rnnVolCell{makeRNNVolCell(cfg, dt, 0x524e5636)}
		x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x524e5637)
		target := make([]float32, rnnVolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%4+1))
		}
		goO, err := runGoRNNVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeRNNVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppRNNVolNBP(payload, 0)
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

func caseRNNNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 20
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volRNNCfg()
	W := randInit(1, cfg.WeightCount(), 0x524e5638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x524e5639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, rnnVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]rnnVolCell, stack)
	for i := range cells {
		cells[i] = rnnVolCell{
			enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	cppO, _, err := runCppRNNVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseRNNNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	stack := depth * rows * cols * lpc
	cfg := volRNNCfg()
	W := randInit(1, cfg.WeightCount(), 0x524e563a)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x524e563b)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, rnnVolPostLen(batch, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      RNN NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := make([]rnnVolCell, stack)
		for i := range cells {
			cells[i] = rnnVolCell{
				enabled: true, in: cfg.InputSize, hid: cfg.HiddenSize, seq: cfg.SeqLen,
				dtype: dt, W: append([]float32(nil), W...),
			}
		}
		goO, err := runGoRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeRNNVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppRNNVolNBP(payload, 0)
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

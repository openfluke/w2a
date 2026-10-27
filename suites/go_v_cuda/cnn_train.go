package go_v_cuda

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
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesCNNTrain — NormalBP CNN1/2/3 Go↔CUDA (leaf + volumetric).
func CasesCNNTrain() []Case {
	var cs []Case
	add := func(tier int, name string, run func() error) {
		prefix := fmt.Sprintf("CNN%d", tier)
		cs = append(cs, Case{Name: prefix + " NormalBP " + name, Run: run})
	}
	for _, tier := range []int{1, 2, 3} {
		t := tier
		add(t, "leaf Float32 Linear — exact bits (dX/dW/W/loss)", func() error { return caseCNNNBPLeafF32(t) })
		add(t, "leaf × all 34 dtypes — spectrum + speed", func() error { return caseCNNNBPLeafAllDTypes(t) })
		add(t, "volume 1×1×1×1 Float32 — exact bits", func() error { return caseCNNNBPVol111F32(t) })
		add(t, "volume 1×1×1×2 chain Float32 — exact bits", func() error { return caseCNNNBPVolChain(t) })
		add(t, "volume 1×1×1×1 × all 34 dtypes — spectrum", func() error { return caseCNNNBPVol111AllDTypes(t) })
		add(t, "volume 2×2×2×1 Float32 — exact bits", func() error { return caseCNNNBPVol222F32(t) })
		add(t, "volume 2×1×1×1 × all 34 dtypes — spectrum", func() error { return caseCNNNBPVol211AllDTypes(t) })
	}
	return cs
}

func RunCNNTrain() error {
	var fails []string
	cs := CasesCNNTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [cnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_cnn_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_cnn_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda cnn_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type cnnNBPLeaf struct {
	tier           int
	batch          int
	c1             cnn1.Config
	c2             cnn2.Config
	c3             cnn3.Config
	dtype          core.DType
	W              []float32
	bias           []float64
	x              []float32
	target         []float32
	lr             float64
}

func preservingCNN1() cnn1.Config {
	return cnn1.Config{
		InChannels: 2, Filters: 2, SeqLen: 8, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
}

func preservingCNN2() cnn2.Config {
	return cnn2.Config{
		InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
}

func preservingCNN3() cnn3.Config {
	return cnn3.Config{
		InChannels: 1, Filters: 1, Depth: 3, Height: 3, Width: 3, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
}

func smokeCNN1() cnn1.Config {
	return cnn1.Config{
		InChannels: 2, Filters: 3, SeqLen: 8, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
}

func smokeCNN2() cnn2.Config {
	return cnn2.Config{
		InChannels: 2, Filters: 3, Height: 6, Width: 6, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationLinear,
	}
}

func smokeCNN3() cnn3.Config {
	return cnn3.Config{
		InChannels: 1, Filters: 2, Depth: 4, Height: 4, Width: 4, Kernel: 2, Stride: 1, Padding: 0,
		Activation: core.ActivationLinear,
	}
}

func cnnLeafPostLen(fc cnnNBPLeaf) (int, error) {
	switch fc.tier {
	case 1:
		if err := fc.c1.Validate(); err != nil {
			return 0, err
		}
		return fc.batch * fc.c1.Filters * fc.c1.OutLen(), nil
	case 2:
		if err := fc.c2.Validate(); err != nil {
			return 0, err
		}
		return fc.batch * fc.c2.Filters * fc.c2.OutH() * fc.c2.OutW(), nil
	case 3:
		if err := fc.c3.Validate(); err != nil {
			return 0, err
		}
		return fc.batch * fc.c3.Filters * fc.c3.OutD() * fc.c3.OutH() * fc.c3.OutW(), nil
	default:
		return 0, fmt.Errorf("cnn nbp: bad tier %d", fc.tier)
	}
}

func runGoCNNNBP(fc cnnNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	switch fc.tier {
	case 1:
		cfg := fc.c1
		l, err := cnn1.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
		if err != nil {
			return z, err
		}
		disableCNN1(l)
		if len(fc.bias) > 0 {
			l.Proj.Weights.Bias = append([]float64(nil), fc.bias...)
		}
		xin := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.SeqLen)
		copy(xin.Data, fc.x)
		preT, postT, err := cnn1.Forward(l, xin)
		if err != nil {
			return z, err
		}
		return finishCNNLeafNBP(l.Proj.Weights, preT, postT, xin, fc,
			func(gy *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
				return cnn1.Backward(l, gy, xin, preT)
			},
			func(dW *core.Tensor[float32], lr float64) error { return cnn1.ApplyGradSGD(l, dW, lr) },
		)
	case 2:
		cfg := fc.c2
		l, err := cnn2.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
		if err != nil {
			return z, err
		}
		disableCNN2(l)
		if len(fc.bias) > 0 {
			l.Proj.Weights.Bias = append([]float64(nil), fc.bias...)
		}
		xin := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Height, cfg.Width)
		copy(xin.Data, fc.x)
		preT, postT, err := cnn2.Forward(l, xin)
		if err != nil {
			return z, err
		}
		return finishCNNLeafNBP(l.Proj.Weights, preT, postT, xin, fc,
			func(gy *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
				return cnn2.Backward(l, gy, xin, preT)
			},
			func(dW *core.Tensor[float32], lr float64) error { return cnn2.ApplyGradSGD(l, dW, lr) },
		)
	case 3:
		cfg := fc.c3
		l, err := cnn3.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
		if err != nil {
			return z, err
		}
		disableCNN3(l)
		if len(fc.bias) > 0 {
			l.Proj.Weights.Bias = append([]float64(nil), fc.bias...)
		}
		xin := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
		copy(xin.Data, fc.x)
		preT, postT, err := cnn3.Forward(l, xin)
		if err != nil {
			return z, err
		}
		return finishCNNLeafNBP(l.Proj.Weights, preT, postT, xin, fc,
			func(gy *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
				return cnn3.Backward(l, gy, xin, preT)
			},
			func(dW *core.Tensor[float32], lr float64) error { return cnn3.ApplyGradSGD(l, dW, lr) },
		)
	default:
		return z, fmt.Errorf("cnn nbp: bad tier %d", fc.tier)
	}
}

func finishCNNLeafNBP(
	ws *weights.Store,
	preT, postT, xin *core.Tensor[float32],
	fc cnnNBPLeaf,
	bwd func(*core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error),
	apply func(*core.Tensor[float32], float64) error,
) (nbpLeafOut, error) {
	var z nbpLeafOut
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
	dX, dW, err := bwd(gy)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := apply(dW, fc.lr); err != nil {
			return z, err
		}
	}
	Wa, err := weights.MatrixF32(ws)
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

func cnnLeafReqMagic(tier int) (req, resp string) {
	switch tier {
	case 1:
		return "C1B1", "C1BO"
	case 2:
		return "C2B1", "C2BO"
	case 3:
		return "C3B1", "C3BO"
	default:
		return "", ""
	}
}

func cnnVolReqMagic(tier int) (req, resp string) {
	switch tier {
	case 1:
		return "C1V1", "C1VO"
	case 2:
		return "C2V1", "C2VO"
	case 3:
		return "C3V1", "C3VO"
	default:
		return "", ""
	}
}

func encodeCNNNBP(fc cnnNBPLeaf) ([]byte, error) {
	reqMagic, _ := cnnLeafReqMagic(fc.tier)
	var in bytes.Buffer
	_, _ = in.Write([]byte(reqMagic))
	switch fc.tier {
	case 1:
		cfg := fc.c1
		if err := cfg.Validate(); err != nil {
			return nil, err
		}
		for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.SeqLen),
			int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.Activation), int32(fc.dtype), 0} {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	case 2:
		cfg := fc.c2
		if err := cfg.Validate(); err != nil {
			return nil, err
		}
		for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height),
			int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.Activation),
			int32(fc.dtype), 0} {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	case 3:
		cfg := fc.c3
		if err := cfg.Validate(); err != nil {
			return nil, err
		}
		for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth),
			int32(cfg.Height), int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding),
			int32(cfg.Activation), int32(fc.dtype), 0} {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	default:
		return nil, fmt.Errorf("encode cnn nbp tier %d", fc.tier)
	}
	for _, v := range fc.W {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	writeBias(&in, fc.bias)
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

func decodeCNNNBP(raw []byte, tier int) (nbpLeafOut, time.Duration, error) {
	var z nbpLeafOut
	var bench time.Duration
	_, respMagic := cnnLeafReqMagic(tier)
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
		return z, bench, fmt.Errorf("cuda cnn%d nbp bad magic %q", tier, string(raw[:min(4, len(raw))]))
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

func runCudaCNNNBP(fc cnnNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCudaCNNNBPBench(fc, 0)
	return out, err
}

func runCudaCNNNBPBench(fc cnnNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findCUDAFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeCNNNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("cuda cnn%d nbp: %v: %s", fc.tier, err, stderr.String())
	}
	return decodeCNNNBP(stdout.Bytes(), fc.tier)
}

func makeCNNLeafCase(tier, batch int, dt core.DType, seed uint32, smoke bool) (cnnNBPLeaf, error) {
	var fc cnnNBPLeaf
	fc.tier, fc.batch, fc.dtype, fc.lr = tier, batch, dt, 1e-2
	switch tier {
	case 1:
		if smoke {
			fc.c1 = smokeCNN1()
		} else {
			fc.c1 = preservingCNN1()
		}
		cfg := fc.c1
		fc.W = randInit(cfg.Filters, cfg.PatchDim(), seed)
		if smoke {
			fc.bias = []float64{0.1, -0.2, 0.05}
		}
		fc.x = randInit(batch*cfg.InChannels, cfg.SeqLen, seed^0x9e3779b9)
	case 2:
		if smoke {
			fc.c2 = smokeCNN2()
		} else {
			fc.c2 = preservingCNN2()
		}
		cfg := fc.c2
		fc.W = randInit(cfg.Filters, cfg.PatchDim(), seed)
		if smoke {
			fc.bias = []float64{0.1, -0.2, 0.05}
		}
		fc.x = randInit(cfg.InChannels*cfg.Height, cfg.Width, seed^0x9e3779b9)
	case 3:
		if smoke {
			fc.c3 = smokeCNN3()
		} else {
			fc.c3 = preservingCNN3()
		}
		cfg := fc.c3
		fc.W = randInit(cfg.Filters, cfg.PatchDim(), seed)
		if smoke {
			fc.bias = []float64{0.25, -0.1}
		}
		fc.x = randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, seed^0x9e3779b9)
	default:
		return fc, fmt.Errorf("makeCNNLeafCase tier %d", tier)
	}
	n, err := cnnLeafPostLen(fc)
	if err != nil {
		return fc, err
	}
	fc.target = make([]float32, n)
	for i := range fc.target {
		fc.target[i] = float32(0.02 * float64(i%7+1))
	}
	return fc, nil
}

func caseCNNNBPLeafF32(tier int) error {
	benchIters := defaultGoBenchIters
	batch := 1
	if tier == 1 {
		batch = 2
	}
	fc, err := makeCNNLeafCase(tier, batch, core.DTypeFloat32, uint32(0x434e0000|uint32(tier)), true)
	if err != nil {
		return err
	}
	goO, err := runGoCNNNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cudaO, err := runCudaCNNNBP(fc)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	if err := compareNBP(goO, cudaO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoCNNNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cudaDur, err := runCudaCNNNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cuda bench: %w", err)
	}
	cudaUS := perIterUS(cudaDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cudaUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
	return nil
}

func caseCNNNBPLeafAllDTypes(tier int) error {
	benchIters := defaultGoBenchIters
	batch := 1
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — CNN%d NBP leaf ×34\n", tier)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cuda_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeCNNLeafCase(tier, batch, dt, 0xd7a4e100+uint32(tier), false)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoCNNNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cudaO, err := runCudaCNNNBP(fc)
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
			_, err := runGoCNNNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cudaDur, err := runCudaCNNNBPBench(fc, benchIters)
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

type cnn1VolCell struct {
	enabled bool
	cfg     cnn1.Config
	dtype   core.DType
	W       []float32
	bias    []float64
}

type cnn2VolCell struct {
	enabled bool
	cfg     cnn2.Config
	dtype   core.DType
	W       []float32
	bias    []float64
}

type cnn3VolCell struct {
	enabled bool
	cfg     cnn3.Config
	dtype   core.DType
	W       []float32
	bias    []float64
}

func makeCNN1VolCell(cfg cnn1.Config, dt core.DType, seed uint32) (cnn1VolCell, error) {
	if err := cfg.Validate(); err != nil {
		return cnn1VolCell{}, err
	}
	return cnn1VolCell{
		enabled: true, cfg: cfg, dtype: dt,
		W: randInit(cfg.Filters, cfg.PatchDim(), seed),
	}, nil
}

func makeCNN2VolCell(cfg cnn2.Config, dt core.DType, seed uint32) (cnn2VolCell, error) {
	if err := cfg.Validate(); err != nil {
		return cnn2VolCell{}, err
	}
	return cnn2VolCell{
		enabled: true, cfg: cfg, dtype: dt,
		W: randInit(cfg.Filters, cfg.PatchDim(), seed),
	}, nil
}

func makeCNN3VolCell(cfg cnn3.Config, dt core.DType, seed uint32) (cnn3VolCell, error) {
	if err := cfg.Validate(); err != nil {
		return cnn3VolCell{}, err
	}
	return cnn3VolCell{
		enabled: true, cfg: cfg, dtype: dt,
		W: randInit(cfg.Filters, cfg.PatchDim(), seed),
	}, nil
}

func runGoCNN1VolNBP(depth, rows, cols, lpc, batch int, cells []cnn1VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cl, err := cnn1.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableCNN1(cl)
		if len(vc.bias) > 0 {
			cl.Proj.Weights.Bias = append([]float64(nil), vc.bias...)
		}
		if err := cnn1.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	return finishCNNVolNBP(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*cnn1.Layer)
		if !ok {
			return nil, fmt.Errorf("expected cnn1 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoCNN2VolNBP(depth, rows, cols, lpc, batch int, cells []cnn2VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cl, err := cnn2.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableCNN2(cl)
		if len(vc.bias) > 0 {
			cl.Proj.Weights.Bias = append([]float64(nil), vc.bias...)
		}
		if err := cnn2.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishCNNVolNBP(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*cnn2.Layer)
		if !ok {
			return nil, fmt.Errorf("expected cnn2 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoCNN3VolNBP(depth, rows, cols, lpc, batch int, cells []cnn3VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cl, err := cnn3.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableCNN3(cl)
		if len(vc.bias) > 0 {
			cl.Proj.Weights.Bias = append([]float64(nil), vc.bias...)
		}
		if err := cnn3.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishCNNVolNBP(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*cnn3.Layer)
		if !ok {
			return nil, fmt.Errorf("expected cnn3 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func finishCNNVolNBP[T any](
	g *architecture.Grid,
	hops []architecture.Coord,
	cells []T,
	xin *core.Tensor[float32],
	target []float32,
	lr float64,
	exportW func(int, *architecture.Cell) ([]float32, error),
) (nbpVolOut, error) {
	var z nbpVolOut
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
		_ = c
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil {
			continue
		}
		wa, err := exportW(i, cell)
		if err != nil {
			return z, err
		}
		if wa != nil {
			z.Ws = append(z.Ws, wa)
		}
	}
	return z, nil
}

func writeCNN1VolCell(buf *bytes.Buffer, c cnn1VolCell) error {
	cfg := c.cfg
	if err := cfg.Validate(); err != nil {
		return err
	}
	for _, v := range []int32{
		int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.SeqLen),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.Activation), int32(c.dtype),
	} {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	for _, v := range c.W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeBias(buf, c.bias)
	return nil
}

func writeCNN2VolCell(buf *bytes.Buffer, c cnn2VolCell) error {
	cfg := c.cfg
	if err := cfg.Validate(); err != nil {
		return err
	}
	for _, v := range []int32{
		int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height), int32(cfg.Width),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.Activation), int32(c.dtype),
	} {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	for _, v := range c.W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeBias(buf, c.bias)
	return nil
}

func writeCNN3VolCell(buf *bytes.Buffer, c cnn3VolCell) error {
	cfg := c.cfg
	if err := cfg.Validate(); err != nil {
		return err
	}
	for _, v := range []int32{
		int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth), int32(cfg.Height), int32(cfg.Width),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.Activation), int32(c.dtype),
	} {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	for _, v := range c.W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeBias(buf, c.bias)
	return nil
}

func encodeCNN1VolNBP(depth, rows, cols, lpc, batch int, cells []cnn1VolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("C1V1"))
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
		if err := writeCNN1VolCell(&in, c); err != nil {
			return nil, err
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

func encodeCNN2VolNBP(depth, rows, cols, lpc, batch int, cells []cnn2VolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("C2V1"))
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
		if err := writeCNN2VolCell(&in, c); err != nil {
			return nil, err
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

func encodeCNN3VolNBP(depth, rows, cols, lpc, batch int, cells []cnn3VolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("C3V1"))
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
		if err := writeCNN3VolCell(&in, c); err != nil {
			return nil, err
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

func decodeCNNVolNBP(raw []byte, tier int) (nbpVolOut, time.Duration, error) {
	var z nbpVolOut
	var bench time.Duration
	_, respMagic := cnnVolReqMagic(tier)
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
		return z, bench, fmt.Errorf("cuda cnn%d vol nbp bad magic", tier)
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

func runCudaCNNVolNBP(tier, depth, rows, cols, lpc, batch int, payload []byte) (nbpVolOut, error) {
	out, _, err := runCudaCNNVolNBPBench(tier, payload, 0)
	return out, err
}

func runCudaCNNVolNBPBench(tier int, payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
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
		return nbpVolOut{}, 0, fmt.Errorf("cuda cnn%d vol nbp: %v: %s", tier, err, stderr.String())
	}
	return decodeCNNVolNBP(stdout.Bytes(), tier)
}

func cnn1VolPostLen(batch int, cfg cnn1.Config) int {
	return batch * cfg.Filters * cfg.OutLen()
}

func cnn2VolPostLen(batch int, cfg cnn2.Config) int {
	return batch * cfg.Filters * cfg.OutH() * cfg.OutW()
}

func cnn3VolPostLen(batch int, cfg cnn3.Config) int {
	return batch * cfg.Filters * cfg.OutD() * cfg.OutH() * cfg.OutW()
}

func benchCNN1VolNBP(depth, rows, cols, lpc, batch int, cells []cnn1VolCell, x, target []float32, lr float64, benchIters int) (goUS, cudaUS, spdPct float64, err error) {
	payload, err := encodeCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cudaDur, err := runCudaCNNVolNBPBench(1, payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cudaUS = perIterUS(cudaDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cudaUS)
	return goUS, cudaUS, spdPct, nil
}

func benchCNN2VolNBP(depth, rows, cols, lpc, batch int, cells []cnn2VolCell, x, target []float32, lr float64, benchIters int) (goUS, cudaUS, spdPct float64, err error) {
	payload, err := encodeCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cudaDur, err := runCudaCNNVolNBPBench(2, payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cudaUS = perIterUS(cudaDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cudaUS)
	return goUS, cudaUS, spdPct, nil
}

func benchCNN3VolNBP(depth, rows, cols, lpc, batch int, cells []cnn3VolCell, x, target []float32, lr float64, benchIters int) (goUS, cudaUS, spdPct float64, err error) {
	payload, err := encodeCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cudaDur, err := runCudaCNNVolNBPBench(3, payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cudaUS = perIterUS(cudaDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cudaUS)
	return goUS, cudaUS, spdPct, nil
}

func caseCNNNBPVol111F32(tier int) error {
	const batch = 1
	benchIters := defaultGoBenchIters
	const lr = 1e-2
	switch tier {
	case 1:
		cfg := preservingCNN1()
		cell, err := makeCNN1VolCell(cfg, core.DTypeFloat32, 0x43311100|uint32(tier))
		if err != nil {
			return err
		}
		cells := []cnn1VolCell{cell}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43311101)
		target := make([]float32, cnn1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%5+1))
		}
		goO, err := runGoCNN1VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeCNN1VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 1, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN1VolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	case 2:
		cfg := preservingCNN2()
		cell, err := makeCNN2VolCell(cfg, core.DTypeFloat32, 0x43321100)
		if err != nil {
			return err
		}
		cells := []cnn2VolCell{cell}
		x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43321101)
		target := make([]float32, cnn2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%5+1))
		}
		goO, err := runGoCNN2VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeCNN2VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 1, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN2VolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	case 3:
		cfg := preservingCNN3()
		cell, err := makeCNN3VolCell(cfg, core.DTypeFloat32, 0x43331100)
		if err != nil {
			return err
		}
		cells := []cnn3VolCell{cell}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43331101)
		target := make([]float32, cnn3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%5+1))
		}
		goO, err := runGoCNN3VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeCNN3VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 1, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN3VolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	default:
		return fmt.Errorf("vol111 tier %d", tier)
	}
}

func caseCNNNBPVolChain(tier int) error {
	const batch = 1
	benchIters := defaultGoBenchIters
	const lr = 1e-2
	switch tier {
	case 1:
		cfg := preservingCNN1()
		c0, err := makeCNN1VolCell(cfg, core.DTypeFloat32, 0x43311200)
		if err != nil {
			return err
		}
		c1, err := makeCNN1VolCell(cfg, core.DTypeFloat32, 0x43311201)
		if err != nil {
			return err
		}
		cells := []cnn1VolCell{c0, c1}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43311202)
		target := make([]float32, cnn1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.02 * float64(i%4+1))
		}
		goO, err := runGoCNN1VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeCNN1VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 2, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN1VolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	case 2:
		cfg := preservingCNN2()
		c0, _ := makeCNN2VolCell(cfg, core.DTypeFloat32, 0x43321200)
		c1, _ := makeCNN2VolCell(cfg, core.DTypeFloat32, 0x43321201)
		cells := []cnn2VolCell{c0, c1}
		x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43321202)
		target := make([]float32, cnn2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.02 * float64(i%4+1))
		}
		goO, err := runGoCNN2VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeCNN2VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 2, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN2VolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	case 3:
		cfg := preservingCNN3()
		c0, _ := makeCNN3VolCell(cfg, core.DTypeFloat32, 0x43331200)
		c1, _ := makeCNN3VolCell(cfg, core.DTypeFloat32, 0x43331201)
		cells := []cnn3VolCell{c0, c1}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43331202)
		target := make([]float32, cnn3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.02 * float64(i%4+1))
		}
		goO, err := runGoCNN3VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeCNN3VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 2, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN3VolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	default:
		return fmt.Errorf("vol chain tier %d", tier)
	}
}

func caseCNNNBPVol111AllDTypes(tier int) error {
	const batch = 1
	benchIters := defaultGoBenchIters
	var fails []string
	fmt.Printf("\n      CNN%d NBP vol 1×1×1×1 ×34\n", tier)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cudaµs", "spd%Δ", "max|Δ|")
	for _, dt := range core.AllDTypes {
		switch tier {
		case 1:
			cfg := preservingCNN1()
			cell, err := makeCNN1VolCell(cfg, dt, 0x43311300)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			cells := []cnn1VolCell{cell}
			x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43311301)
			target := make([]float32, cnn1VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.03 * float64(i%4+1))
			}
			goO, err := runGoCNN1VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeCNN1VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 1, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cudaO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cudaUS, spdPct, err := benchCNN1VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 2:
			cfg := preservingCNN2()
			cell, err := makeCNN2VolCell(cfg, dt, 0x43321300)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			cells := []cnn2VolCell{cell}
			x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43321301)
			target := make([]float32, cnn2VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.03 * float64(i%4+1))
			}
			goO, err := runGoCNN2VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeCNN2VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 1, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cudaO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cudaUS, spdPct, err := benchCNN2VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 3:
			cfg := preservingCNN3()
			cell, err := makeCNN3VolCell(cfg, dt, 0x43331300)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			cells := []cnn3VolCell{cell}
			x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43331301)
			target := make([]float32, cnn3VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.03 * float64(i%4+1))
			}
			goO, err := runGoCNN3VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeCNN3VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			cudaO, err := runCudaCNNVolNBP(tier, 1, 1, 1, 1, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cudaO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cudaUS, spdPct, err := benchCNN3VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d vol111 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseCNNNBPVol222F32(tier int) error {
	const depth, rows, cols, lpc, batch = 2, 2, 2, 1, 1
	benchIters := defaultGoBenchIters
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	switch tier {
	case 1:
		cfg := preservingCNN1()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x43312200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43312201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, cnn1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]cnn1VolCell, stack)
		for i := range cells {
			cells[i] = cnn1VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		cudaO, err := runCudaCNNVolNBP(tier, depth, rows, cols, lpc, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	case 2:
		cfg := preservingCNN2()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x43322200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43322201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, cnn2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]cnn2VolCell, stack)
		for i := range cells {
			cells[i] = cnn2VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		cudaO, err := runCudaCNNVolNBP(tier, depth, rows, cols, lpc, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	case 3:
		cfg := preservingCNN3()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x43332200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43332201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, cnn3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]cnn3VolCell, stack)
		for i := range cells {
			cells[i] = cnn3VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		cudaO, err := runCudaCNNVolNBP(tier, depth, rows, cols, lpc, batch, payload)
		if err != nil {
			return fmt.Errorf("cuda: %w", err)
		}
		if err := compareVolNBP(goO, cudaO, true); err != nil {
			return err
		}
		goUS, cudaUS, spdPct, err := benchCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cuda=%s  spd%%Δ=%+.1f%%",
			goO.loss, FormatUS(goUS), FormatUS(cudaUS), spdPct)
		return nil
	default:
		return fmt.Errorf("vol222 tier %d", tier)
	}
}

func caseCNNNBPVol211AllDTypes(tier int) error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	benchIters := defaultGoBenchIters
	stack := depth * rows * cols * lpc
	var fails []string
	fmt.Printf("\n      CNN%d NBP vol 2×1×1×1 ×34\n", tier)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cudaµs", "spd%Δ", "max|Δ|")
	for _, dt := range core.AllDTypes {
		switch tier {
		case 1:
			cfg := preservingCNN1()
			W := randInit(cfg.Filters, cfg.PatchDim(), 0x43321100)
			for i := range W {
				W[i] *= 0.05
			}
			x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43321101)
			for i := range x {
				x[i] *= 0.05
			}
			target := make([]float32, cnn1VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.05 * float64(i+1))
			}
			cells := make([]cnn1VolCell, stack)
			for i := range cells {
				cells[i] = cnn1VolCell{enabled: true, cfg: cfg, dtype: dt, W: append([]float32(nil), W...)}
			}
			goO, err := runGoCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			cudaO, err := runCudaCNNVolNBP(tier, depth, rows, cols, lpc, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cudaO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cudaUS, spdPct, err := benchCNN1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 2:
			cfg := preservingCNN2()
			W := randInit(cfg.Filters, cfg.PatchDim(), 0x43322100)
			for i := range W {
				W[i] *= 0.05
			}
			x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43322101)
			for i := range x {
				x[i] *= 0.05
			}
			target := make([]float32, cnn2VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.05 * float64(i+1))
			}
			cells := make([]cnn2VolCell, stack)
			for i := range cells {
				cells[i] = cnn2VolCell{enabled: true, cfg: cfg, dtype: dt, W: append([]float32(nil), W...)}
			}
			goO, err := runGoCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			cudaO, err := runCudaCNNVolNBP(tier, depth, rows, cols, lpc, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cudaO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cudaUS, spdPct, err := benchCNN2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 3:
			cfg := preservingCNN3()
			W := randInit(cfg.Filters, cfg.PatchDim(), 0x43332100)
			for i := range W {
				W[i] *= 0.05
			}
			x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43332101)
			for i := range x {
				x[i] *= 0.05
			}
			target := make([]float32, cnn3VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.05 * float64(i+1))
			}
			cells := make([]cnn3VolCell, stack)
			for i := range cells {
				cells[i] = cnn3VolCell{enabled: true, cfg: cfg, dtype: dt, W: append([]float32(nil), W...)}
			}
			goO, err := runGoCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			cudaO, err := runCudaCNNVolNBP(tier, depth, rows, cols, lpc, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cuda: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cudaO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cudaUS, spdPct, err := benchCNN3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cudaO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d vol211 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

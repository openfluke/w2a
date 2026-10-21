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
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesConvTTrain — NormalBP ConvT1/2/3 Go↔C++ (leaf + volumetric).
func CasesConvTTrain() []Case {
	var cs []Case
	add := func(tier int, name string, run func() error) {
		prefix := fmt.Sprintf("ConvT%d", tier)
		cs = append(cs, Case{Name: prefix + " NormalBP " + name, Run: run})
	}
	for _, tier := range []int{1, 2, 3} {
		t := tier
		add(t, "leaf Float32 Linear — exact bits (dX/dW/W/loss)", func() error { return caseConvTNBPLeafF32(t) })
		add(t, "leaf × all 34 dtypes — spectrum + speed", func() error { return caseConvTNBPLeafAllDTypes(t) })
		add(t, "volume 1×1×1×1 Float32 — exact bits", func() error { return caseConvTNBPVol111F32(t) })
		add(t, "volume 1×1×1×2 chain Float32 — exact bits", func() error { return caseConvTNBPVolChain(t) })
		add(t, "volume 1×1×1×1 × all 34 dtypes — spectrum", func() error { return caseConvTNBPVol111AllDTypes(t) })
		add(t, "volume 2×2×2×1 Float32 — exact bits", func() error { return caseConvTNBPVol222F32(t) })
		add(t, "volume 2×1×1×1 × all 34 dtypes — spectrum", func() error { return caseConvTNBPVol211AllDTypes(t) })
	}
	return cs
}

func RunConvTTrain() error {
	var fails []string
	cs := CasesConvTTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [ctnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_convt_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_convt_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp convt_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type convtNBPLeaf struct {
	tier           int
	batch          int
	c1             convt1.Config
	c2             convt2.Config
	c3             convt3.Config
	dtype          core.DType
	W              []float32
	x              []float32
	target         []float32
	lr             float64
}

func preservingConvT1() convt1.Config {
	return convt1.Config{
		InChannels: 2, Filters: 2, SeqLen: 8, Kernel: 3, Stride: 1, Padding: 1, OutputPadding: 0,
		Activation: core.ActivationLinear,
	}
}

func preservingConvT2() convt2.Config {
	return convt2.Config{
		InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Stride: 1, Padding: 1, OutputPadding: 0,
		Activation: core.ActivationLinear,
	}
}

func preservingConvT3() convt3.Config {
	return convt3.Config{
		InChannels: 1, Filters: 1, Depth: 3, Height: 3, Width: 3, Kernel: 3, Stride: 1, Padding: 1, OutputPadding: 0,
		Activation: core.ActivationLinear,
	}
}

func smokeConvT1() convt1.Config {
	return convt1.Config{
		InChannels: 2, Filters: 3, SeqLen: 5, Kernel: 3, Stride: 2, Padding: 1, OutputPadding: 1,
		Activation: core.ActivationLinear,
	}
}

func smokeConvT2() convt2.Config {
	return convt2.Config{
		InChannels: 2, Filters: 2, Height: 4, Width: 4, Kernel: 3, Stride: 1, Padding: 1, OutputPadding: 0,
		Activation: core.ActivationReLU,
	}
}

func smokeConvT3() convt3.Config {
	return convt3.Config{
		InChannels: 1, Filters: 2, Depth: 3, Height: 3, Width: 3, Kernel: 2, Stride: 2, Padding: 0,
		OutputPadding: 1, Activation: core.ActivationLinear,
	}
}

func convtLeafPostLen(fc convtNBPLeaf) (int, error) {
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
		return 0, fmt.Errorf("convt nbp: bad tier %d", fc.tier)
	}
}

func runGoConvTNBP(fc convtNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	switch fc.tier {
	case 1:
		cfg := fc.c1
		l, err := convt1.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
		if err != nil {
			return z, err
		}
		disableConvT1(l)
		xin := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.SeqLen)
		copy(xin.Data, fc.x)
		preT, postT, err := convt1.Forward(l, xin)
		if err != nil {
			return z, err
		}
		return finishConvTLeafNBP(l.Proj.Weights, preT, postT, xin, fc,
			func(gy *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
				return convt1.Backward(l, gy, xin, preT)
			},
			func(dW *core.Tensor[float32], lr float64) error { return convt1.ApplyGradSGD(l, dW, lr) },
		)
	case 2:
		cfg := fc.c2
		l, err := convt2.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
		if err != nil {
			return z, err
		}
		disableConvT2(l)
		xin := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Height, cfg.Width)
		copy(xin.Data, fc.x)
		preT, postT, err := convt2.Forward(l, xin)
		if err != nil {
			return z, err
		}
		return finishConvTLeafNBP(l.Proj.Weights, preT, postT, xin, fc,
			func(gy *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
				return convt2.Backward(l, gy, xin, preT)
			},
			func(dW *core.Tensor[float32], lr float64) error { return convt2.ApplyGradSGD(l, dW, lr) },
		)
	case 3:
		cfg := fc.c3
		l, err := convt3.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
		if err != nil {
			return z, err
		}
		disableConvT3(l)
		xin := core.NewTensor[float32](fc.batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
		copy(xin.Data, fc.x)
		preT, postT, err := convt3.Forward(l, xin)
		if err != nil {
			return z, err
		}
		return finishConvTLeafNBP(l.Proj.Weights, preT, postT, xin, fc,
			func(gy *core.Tensor[float32]) (*core.Tensor[float32], *core.Tensor[float32], error) {
				return convt3.Backward(l, gy, xin, preT)
			},
			func(dW *core.Tensor[float32], lr float64) error { return convt3.ApplyGradSGD(l, dW, lr) },
		)
	default:
		return z, fmt.Errorf("convt nbp: bad tier %d", fc.tier)
	}
}

func finishConvTLeafNBP(
	ws *weights.Store,
	preT, postT, xin *core.Tensor[float32],
	fc convtNBPLeaf,
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

func convtLeafReqMagic(tier int) (req, resp string) {
	switch tier {
	case 1:
		return "T1B1", "T1BO"
	case 2:
		return "T2B1", "T2BO"
	case 3:
		return "T3B1", "T3BO"
	default:
		return "", ""
	}
}

func convtVolReqMagic(tier int) (req, resp string) {
	switch tier {
	case 1:
		return "T1V1", "T1VO"
	case 2:
		return "T2V1", "T2VO"
	case 3:
		return "T3V1", "T3VO"
	default:
		return "", ""
	}
}

func encodeConvTNBP(fc convtNBPLeaf) ([]byte, error) {
	reqMagic, _ := convtLeafReqMagic(fc.tier)
	var in bytes.Buffer
	_, _ = in.Write([]byte(reqMagic))
	switch fc.tier {
	case 1:
		cfg := fc.c1
		if err := cfg.Validate(); err != nil {
			return nil, err
		}
		for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.SeqLen),
			int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding),
			int32(cfg.Activation), int32(fc.dtype), 0} {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	case 2:
		cfg := fc.c2
		if err := cfg.Validate(); err != nil {
			return nil, err
		}
		for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height),
			int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding),
			int32(cfg.Activation), int32(fc.dtype), 0} {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	case 3:
		cfg := fc.c3
		if err := cfg.Validate(); err != nil {
			return nil, err
		}
		for _, v := range []int32{int32(fc.batch), int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth),
			int32(cfg.Height), int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding),
			int32(cfg.OutputPadding), int32(cfg.Activation), int32(fc.dtype), 0} {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	default:
		return nil, fmt.Errorf("encode convt nbp tier %d", fc.tier)
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

func decodeConvTNBP(raw []byte, tier int) (nbpLeafOut, time.Duration, error) {
	var z nbpLeafOut
	var bench time.Duration
	_, respMagic := convtLeafReqMagic(tier)
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
		return z, bench, fmt.Errorf("cpp convt%d nbp bad magic %q", tier, string(raw[:min(4, len(raw))]))
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

func runCppConvTNBP(fc convtNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCppConvTNBPBench(fc, 0)
	return out, err
}

func runCppConvTNBPBench(fc convtNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeConvTNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("cpp convt%d nbp: %v: %s", fc.tier, err, stderr.String())
	}
	return decodeConvTNBP(stdout.Bytes(), fc.tier)
}

func makeConvTLeafCase(tier, batch int, dt core.DType, seed uint32, smoke bool) (convtNBPLeaf, error) {
	var fc convtNBPLeaf
	fc.tier, fc.batch, fc.dtype, fc.lr = tier, batch, dt, 1e-2
	switch tier {
	case 1:
		if smoke {
			fc.c1 = smokeConvT1()
		} else {
			fc.c1 = preservingConvT1()
		}
		cfg := fc.c1
		fc.W = randInit(cfg.Filters, cfg.PatchDim(), seed)
		fc.x = randInit(max(batch,1)*cfg.InChannels, cfg.SeqLen, seed^0x9e3779b9)
	case 2:
		if smoke {
			fc.c2 = smokeConvT2()
		} else {
			fc.c2 = preservingConvT2()
		}
		cfg := fc.c2
		fc.W = randInit(cfg.Filters, cfg.PatchDim(), seed)
		fc.x = randInit(cfg.InChannels*cfg.Height, cfg.Width, seed^0x9e3779b9)
	case 3:
		if smoke {
			fc.c3 = smokeConvT3()
		} else {
			fc.c3 = preservingConvT3()
		}
		cfg := fc.c3
		fc.W = randInit(cfg.Filters, cfg.PatchDim(), seed)
		fc.x = randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, seed^0x9e3779b9)
	default:
		return fc, fmt.Errorf("makeConvTLeafCase tier %d", tier)
	}
	n, err := convtLeafPostLen(fc)
	if err != nil {
		return fc, err
	}
	fc.target = make([]float32, n)
	for i := range fc.target {
		fc.target[i] = float32(0.02 * float64(i%7+1))
	}
	return fc, nil
}

func caseConvTNBPLeafF32(tier int) error {
	const benchIters = 40
	batch := 1
	if tier == 1 {
		batch = 2
	}
	fc, err := makeConvTLeafCase(tier, batch, core.DTypeFloat32, uint32(0x43540000|uint32(tier)), true)
	if err != nil {
		return err
	}
	goO, err := runGoConvTNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppConvTNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoConvTNBP(fc); err != nil {
			return err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoConvTNBP(fc); err != nil {
			return err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppConvTNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseConvTNBPLeafAllDTypes(tier int) error {
	const benchIters = 20
	batch := 1
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — ConvT%d NBP leaf ×34\n", tier)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeConvTLeafCase(tier, batch, dt, 0xd7a4e100+uint32(tier), false)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoConvTNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppConvTNBP(fc)
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
			if _, err := runGoConvTNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go warm: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, err := runGoConvTNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, cppDur, err := runCppConvTNBPBench(fc, benchIters)
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

type convt1VolCell struct {
	enabled bool
	cfg     convt1.Config
	dtype   core.DType
	W       []float32
}

type convt2VolCell struct {
	enabled bool
	cfg     convt2.Config
	dtype   core.DType
	W       []float32
}

type convt3VolCell struct {
	enabled bool
	cfg     convt3.Config
	dtype   core.DType
	W       []float32
}

func makeConvT1VolCell(cfg convt1.Config, dt core.DType, seed uint32) (convt1VolCell, error) {
	if err := cfg.Validate(); err != nil {
		return convt1VolCell{}, err
	}
	return convt1VolCell{
		enabled: true, cfg: cfg, dtype: dt,
		W: randInit(cfg.Filters, cfg.PatchDim(), seed),
	}, nil
}

func makeConvT2VolCell(cfg convt2.Config, dt core.DType, seed uint32) (convt2VolCell, error) {
	if err := cfg.Validate(); err != nil {
		return convt2VolCell{}, err
	}
	return convt2VolCell{
		enabled: true, cfg: cfg, dtype: dt,
		W: randInit(cfg.Filters, cfg.PatchDim(), seed),
	}, nil
}

func makeConvT3VolCell(cfg convt3.Config, dt core.DType, seed uint32) (convt3VolCell, error) {
	if err := cfg.Validate(); err != nil {
		return convt3VolCell{}, err
	}
	return convt3VolCell{
		enabled: true, cfg: cfg, dtype: dt,
		W: randInit(cfg.Filters, cfg.PatchDim(), seed),
	}, nil
}

func runGoConvT1VolNBP(depth, rows, cols, lpc, batch int, cells []convt1VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cl, err := convt1.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableConvT1(cl)
		if err := convt1.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	return finishConvTVolNBP(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*convt1.Layer)
		if !ok {
			return nil, fmt.Errorf("expected convt1 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoConvT2VolNBP(depth, rows, cols, lpc, batch int, cells []convt2VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cl, err := convt2.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableConvT2(cl)
		if err := convt2.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishConvTVolNBP(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*convt2.Layer)
		if !ok {
			return nil, fmt.Errorf("expected convt2 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func runGoConvT3VolNBP(depth, rows, cols, lpc, batch int, cells []convt3VolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		cl, err := convt3.NewConfigured(cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableConvT3(cl)
		if err := convt3.Place(g, c.Z, c.Y, c.X, c.L, cl); err != nil {
			return z, err
		}
	}
	cfg := cells[0].cfg
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	return finishConvTVolNBP(g, hops, cells, xin, target, lr, func(i int, cell *architecture.Cell) ([]float32, error) {
		if !cells[i].enabled {
			return nil, nil
		}
		cl, ok := cell.Op.(*convt3.Layer)
		if !ok {
			return nil, fmt.Errorf("expected convt3 at hop %d", i)
		}
		return weights.MatrixF32(cl.Proj.Weights)
	})
}

func finishConvTVolNBP[T any](
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

func writeConvT1VolCell(buf *bytes.Buffer, c convt1VolCell) error {
	cfg := c.cfg
	if err := cfg.Validate(); err != nil {
		return err
	}
	for _, v := range []int32{
		int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.SeqLen),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding),
		int32(cfg.Activation), int32(c.dtype),
	} {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	for _, v := range c.W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	return nil
}

func writeConvT2VolCell(buf *bytes.Buffer, c convt2VolCell) error {
	cfg := c.cfg
	if err := cfg.Validate(); err != nil {
		return err
	}
	for _, v := range []int32{
		int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height), int32(cfg.Width),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding),
		int32(cfg.Activation), int32(c.dtype),
	} {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	for _, v := range c.W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	return nil
}

func writeConvT3VolCell(buf *bytes.Buffer, c convt3VolCell) error {
	cfg := c.cfg
	if err := cfg.Validate(); err != nil {
		return err
	}
	for _, v := range []int32{
		int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth), int32(cfg.Height), int32(cfg.Width),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding),
		int32(cfg.Activation), int32(c.dtype),
	} {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	for _, v := range c.W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	return nil
}

func encodeConvT1VolNBP(depth, rows, cols, lpc, batch int, cells []convt1VolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("T1V1"))
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
		if err := writeConvT1VolCell(&in, c); err != nil {
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

func encodeConvT2VolNBP(depth, rows, cols, lpc, batch int, cells []convt2VolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("T2V1"))
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
		if err := writeConvT2VolCell(&in, c); err != nil {
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

func encodeConvT3VolNBP(depth, rows, cols, lpc, batch int, cells []convt3VolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("T3V1"))
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
		if err := writeConvT3VolCell(&in, c); err != nil {
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

func decodeConvTVolNBP(raw []byte, tier int) (nbpVolOut, time.Duration, error) {
	var z nbpVolOut
	var bench time.Duration
	_, respMagic := convtVolReqMagic(tier)
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
		return z, bench, fmt.Errorf("cpp convt%d vol nbp bad magic", tier)
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

func runCppConvTVolNBP(tier, depth, rows, cols, lpc, batch int, payload []byte) (nbpVolOut, error) {
	out, _, err := runCppConvTVolNBPBench(tier, payload, 0)
	return out, err
}

func runCppConvTVolNBPBench(tier int, payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp convt%d vol nbp: %v: %s", tier, err, stderr.String())
	}
	return decodeConvTVolNBP(stdout.Bytes(), tier)
}

func convt1VolPostLen(batch int, cfg convt1.Config) int {
	return batch * cfg.Filters * cfg.OutLen()
}

func convt2VolPostLen(batch int, cfg convt2.Config) int {
	return batch * cfg.Filters * cfg.OutH() * cfg.OutW()
}

func convt3VolPostLen(batch int, cfg convt3.Config) int {
	return batch * cfg.Filters * cfg.OutD() * cfg.OutH() * cfg.OutW()
}

func benchConvT1VolNBP(depth, rows, cols, lpc, batch int, cells []convt1VolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppConvTVolNBPBench(1, payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func benchConvT2VolNBP(depth, rows, cols, lpc, batch int, cells []convt2VolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppConvTVolNBPBench(2, payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func benchConvT3VolNBP(depth, rows, cols, lpc, batch int, cells []convt3VolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppConvTVolNBPBench(3, payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func caseConvTNBPVol111F32(tier int) error {
	const batch, benchIters = 1, 40
	const lr = 1e-2
	switch tier {
	case 1:
		cfg := preservingConvT1()
		cell, err := makeConvT1VolCell(cfg, core.DTypeFloat32, 0x43311100|uint32(tier))
		if err != nil {
			return err
		}
		cells := []convt1VolCell{cell}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43311101)
		target := make([]float32, convt1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%5+1))
		}
		goO, err := runGoConvT1VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeConvT1VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 1, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT1VolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	case 2:
		cfg := preservingConvT2()
		cell, err := makeConvT2VolCell(cfg, core.DTypeFloat32, 0x43321100)
		if err != nil {
			return err
		}
		cells := []convt2VolCell{cell}
		x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43321101)
		target := make([]float32, convt2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%5+1))
		}
		goO, err := runGoConvT2VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeConvT2VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 1, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT2VolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	case 3:
		cfg := preservingConvT3()
		cell, err := makeConvT3VolCell(cfg, core.DTypeFloat32, 0x43331100)
		if err != nil {
			return err
		}
		cells := []convt3VolCell{cell}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43331101)
		target := make([]float32, convt3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%5+1))
		}
		goO, err := runGoConvT3VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, err := encodeConvT3VolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
		if err != nil {
			return err
		}
		cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 1, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT3VolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	default:
		return fmt.Errorf("vol111 tier %d", tier)
	}
}

func caseConvTNBPVolChain(tier int) error {
	const batch, benchIters = 1, 40
	const lr = 1e-2
	switch tier {
	case 1:
		cfg := preservingConvT1()
		c0, err := makeConvT1VolCell(cfg, core.DTypeFloat32, 0x43311200)
		if err != nil {
			return err
		}
		c1, err := makeConvT1VolCell(cfg, core.DTypeFloat32, 0x43311201)
		if err != nil {
			return err
		}
		cells := []convt1VolCell{c0, c1}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43311202)
		target := make([]float32, convt1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.02 * float64(i%4+1))
		}
		goO, err := runGoConvT1VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeConvT1VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 2, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT1VolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	case 2:
		cfg := preservingConvT2()
		c0, _ := makeConvT2VolCell(cfg, core.DTypeFloat32, 0x43321200)
		c1, _ := makeConvT2VolCell(cfg, core.DTypeFloat32, 0x43321201)
		cells := []convt2VolCell{c0, c1}
		x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43321202)
		target := make([]float32, convt2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.02 * float64(i%4+1))
		}
		goO, err := runGoConvT2VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeConvT2VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 2, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT2VolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	case 3:
		cfg := preservingConvT3()
		c0, _ := makeConvT3VolCell(cfg, core.DTypeFloat32, 0x43331200)
		c1, _ := makeConvT3VolCell(cfg, core.DTypeFloat32, 0x43331201)
		cells := []convt3VolCell{c0, c1}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43331202)
		target := make([]float32, convt3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.02 * float64(i%4+1))
		}
		goO, err := runGoConvT3VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeConvT3VolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
		cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 2, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT3VolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	default:
		return fmt.Errorf("vol chain tier %d", tier)
	}
}

func caseConvTNBPVol111AllDTypes(tier int) error {
	const batch, benchIters = 1, 40
	var fails []string
	fmt.Printf("\n      ConvT%d NBP vol 1×1×1×1 ×34\n", tier)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	for _, dt := range core.AllDTypes {
		switch tier {
		case 1:
			cfg := preservingConvT1()
			cell, err := makeConvT1VolCell(cfg, dt, 0x43311300)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			cells := []convt1VolCell{cell}
			x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43311301)
			target := make([]float32, convt1VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.03 * float64(i%4+1))
			}
			goO, err := runGoConvT1VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeConvT1VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 1, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cppO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cppUS, spdPct, err := benchConvT1VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 2:
			cfg := preservingConvT2()
			cell, err := makeConvT2VolCell(cfg, dt, 0x43321300)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			cells := []convt2VolCell{cell}
			x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43321301)
			target := make([]float32, convt2VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.03 * float64(i%4+1))
			}
			goO, err := runGoConvT2VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeConvT2VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 1, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cppO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cppUS, spdPct, err := benchConvT2VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 3:
			cfg := preservingConvT3()
			cell, err := makeConvT3VolCell(cfg, dt, 0x43331300)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			cells := []convt3VolCell{cell}
			x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43331301)
			target := make([]float32, convt3VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.03 * float64(i%4+1))
			}
			goO, err := runGoConvT3VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeConvT3VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
			cppO, err := runCppConvTVolNBP(tier, 1, 1, 1, 1, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cppO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cppUS, spdPct, err := benchConvT3VolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d vol111 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func caseConvTNBPVol222F32(tier int) error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, 40
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	switch tier {
	case 1:
		cfg := preservingConvT1()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x43312200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43312201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, convt1VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]convt1VolCell, stack)
		for i := range cells {
			cells[i] = convt1VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		cppO, err := runCppConvTVolNBP(tier, depth, rows, cols, lpc, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	case 2:
		cfg := preservingConvT2()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x43322200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43322201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, convt2VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]convt2VolCell, stack)
		for i := range cells {
			cells[i] = convt2VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		cppO, err := runCppConvTVolNBP(tier, depth, rows, cols, lpc, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	case 3:
		cfg := preservingConvT3()
		W := randInit(cfg.Filters, cfg.PatchDim(), 0x43332200)
		for i := range W {
			W[i] *= 0.05
		}
		x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43332201)
		for i := range x {
			x[i] *= 0.05
		}
		target := make([]float32, convt3VolPostLen(batch, cfg))
		for i := range target {
			target[i] = float32(0.05 * float64(i+1))
		}
		cells := make([]convt3VolCell, stack)
		for i := range cells {
			cells[i] = convt3VolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32, W: append([]float32(nil), W...)}
		}
		goO, err := runGoConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		if err != nil {
			return fmt.Errorf("go: %w", err)
		}
		payload, _ := encodeConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		cppO, err := runCppConvTVolNBP(tier, depth, rows, cols, lpc, batch, payload)
		if err != nil {
			return fmt.Errorf("cpp: %w", err)
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			return err
		}
		goUS, cppUS, spdPct, err := benchConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
		if err != nil {
			return fmt.Errorf("bench: %w", err)
		}
		fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
			goO.loss, goUS, cppUS, spdPct)
		return nil
	default:
		return fmt.Errorf("vol222 tier %d", tier)
	}
}

func caseConvTNBPVol211AllDTypes(tier int) error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 1, 1, 1, 1, 40
	stack := depth * rows * cols * lpc
	var fails []string
	fmt.Printf("\n      ConvT%d NBP vol 2×1×1×1 ×34\n", tier)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "goµs", "cppµs", "spd%Δ", "max|Δ|")
	for _, dt := range core.AllDTypes {
		switch tier {
		case 1:
			cfg := preservingConvT1()
			W := randInit(cfg.Filters, cfg.PatchDim(), 0x43321100)
			for i := range W {
				W[i] *= 0.05
			}
			x := randInit(batch*cfg.InChannels, cfg.SeqLen, 0x43321101)
			for i := range x {
				x[i] *= 0.05
			}
			target := make([]float32, convt1VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.05 * float64(i+1))
			}
			cells := make([]convt1VolCell, stack)
			for i := range cells {
				cells[i] = convt1VolCell{enabled: true, cfg: cfg, dtype: dt, W: append([]float32(nil), W...)}
			}
			goO, err := runGoConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			cppO, err := runCppConvTVolNBP(tier, depth, rows, cols, lpc, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cppO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cppUS, spdPct, err := benchConvT1VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 2:
			cfg := preservingConvT2()
			W := randInit(cfg.Filters, cfg.PatchDim(), 0x43322100)
			for i := range W {
				W[i] *= 0.05
			}
			x := randInit(cfg.InChannels*cfg.Height, cfg.Width, 0x43322101)
			for i := range x {
				x[i] *= 0.05
			}
			target := make([]float32, convt2VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.05 * float64(i+1))
			}
			cells := make([]convt2VolCell, stack)
			for i := range cells {
				cells[i] = convt2VolCell{enabled: true, cfg: cfg, dtype: dt, W: append([]float32(nil), W...)}
			}
			goO, err := runGoConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			cppO, err := runCppConvTVolNBP(tier, depth, rows, cols, lpc, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cppO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cppUS, spdPct, err := benchConvT2VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		case 3:
			cfg := preservingConvT3()
			W := randInit(cfg.Filters, cfg.PatchDim(), 0x43332100)
			for i := range W {
				W[i] *= 0.05
			}
			x := randInit(cfg.InChannels*cfg.Depth*cfg.Height, cfg.Width, 0x43332101)
			for i := range x {
				x[i] *= 0.05
			}
			target := make([]float32, convt3VolPostLen(batch, cfg))
			for i := range target {
				target[i] = float32(0.05 * float64(i+1))
			}
			cells := make([]convt3VolCell, stack)
			for i := range cells {
				cells[i] = convt3VolCell{enabled: true, cfg: cfg, dtype: dt, W: append([]float32(nil), W...)}
			}
			goO, err := runGoConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
				continue
			}
			payload, _ := encodeConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
			cppO, err := runCppConvTVolNBP(tier, depth, rows, cols, lpc, batch, payload)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
				continue
			}
			if err := compareVolNBP(goO, cppO, true); err != nil {
				fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
				continue
			}
			goUS, cppUS, spdPct, err := benchConvT3VolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2, benchIters)
			if err != nil {
				fails = append(fails, fmt.Sprintf("%s bench: %v", dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.Ws[0], cppO.Ws[0])
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
				dt, m.Band, m.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", m.MaxAbs))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("%d vol211 dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

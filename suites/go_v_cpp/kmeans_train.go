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
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesKMeansTrain — NormalBP KMeans soft-assignment Go↔C++ (leaf + volumetric).
func CasesKMeansTrain() []Case {
	return []Case{
		{Name: "KMeans NormalBP leaf probs Float32 — exact bits (dX/dW/W/loss)", Run: caseKmNBPLeafProbsF32},
		{Name: "KMeans NormalBP leaf features Float32 — exact bits (dX/dW/W/loss)", Run: caseKmNBPLeafFeaturesF32},
		{Name: "KMeans NormalBP leaf probs × all 34 dtypes — spectrum + speed", Run: caseKmNBPLeafProbsAllDTypes},
		{Name: "KMeans NormalBP volume 1×1×1×1 Float32 features — exact bits", Run: caseKmNBPVol1x1x1},
		{Name: "KMeans NormalBP volume 1×1×1×2 chain Float32 features — exact bits", Run: caseKmNBPVolChain},
		{Name: "KMeans NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseKmNBPVol111AllDTypes},
		{Name: "KMeans NormalBP volume 2×2×2×1 Float32 features — exact bits", Run: caseKmNBPVol222F32},
		{Name: "KMeans NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseKmNBPVol211AllDTypes},
	}
}

func RunKMeansTrain() error {
	var fails []string
	cs := CasesKMeansTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [kmnbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_kmeans_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_kmeans_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp kmeans_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type kmNBPLeaf struct {
	batch  int
	seq    int // 1 → [B,D]
	cfg    kmeans.Config
	dtype  core.DType
	W      []float32
	x      []float32
	target []float32
	lr     float64
}

type kmVolCell struct {
	enabled bool
	cfg     kmeans.Config
	seq     int
	dtype   core.DType
	W       []float32
}

func smokeKmProbsCfg() kmeans.Config {
	return kmeans.Config{
		NumClusters: 4, FeatureDim: 6, Temperature: 1,
		OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
	}
}

func smokeKmFeaturesCfg() kmeans.Config {
	return kmeans.Config{
		NumClusters: 4, FeatureDim: 6, Temperature: 0.8,
		OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear,
	}
}

func volKmCfg() kmeans.Config {
	return kmeans.Config{
		NumClusters: 3, FeatureDim: 4, Temperature: 1,
		OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear,
	}
}

func kmOutLen(batch, seq int, cfg kmeans.Config) int {
	if seq <= 0 {
		seq = 1
	}
	return batch * seq * cfg.OutDim()
}

func disableKMeansTrain(l *kmeans.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	if l.Centers != nil {
		l.Centers.Exec.MultiCore = false
		l.Centers.Core.MultiCore = false
		l.Centers.Exec.Backend = be
	}
}

func packKMeansWeightsF32(l *kmeans.Layer) ([]float32, error) {
	if l == nil || l.Centers == nil {
		return nil, fmt.Errorf("kmeans: nil layer")
	}
	return weights.MatrixF32(l.Centers.Weights)
}

func runGoKmNBP(fc kmNBPLeaf) (nbpLeafOut, error) {
	var z nbpLeafOut
	cfg := fc.cfg
	l, err := kmeans.NewConfigured(cfg, fc.dtype, quant.FormatNone, fc.W)
	if err != nil {
		return z, err
	}
	disableKMeansTrain(l)
	var xin *core.Tensor[float32]
	if fc.seq <= 1 {
		xin = core.NewTensor[float32](fc.batch, cfg.FeatureDim)
	} else {
		xin = core.NewTensor[float32](fc.batch, fc.seq, cfg.FeatureDim)
	}
	copy(xin.Data, fc.x)
	preT, postT, err := kmeans.Forward(l, xin)
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
	dX, dW, err := kmeans.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := kmeans.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	wa, err := packKMeansWeightsF32(l)
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

func encodeKmNBP(fc kmNBPLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	seq := fc.seq
	if seq <= 0 {
		seq = 1
	}
	var in bytes.Buffer
	_, _ = in.Write([]byte("KMB1"))
	for _, v := range []int32{
		int32(fc.batch), int32(seq), int32(cfg.NumClusters), int32(cfg.FeatureDim),
		outputModeInt(cfg.OutputMode), int32(cfg.Activation),
	} {
		_ = binary.Write(&in, binary.LittleEndian, v)
	}
	_ = binary.Write(&in, binary.LittleEndian, cfg.Temperature)
	for _, v := range []int32{int32(fc.dtype), 0} {
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

func decodeKmNBP(raw []byte) (nbpLeafOut, time.Duration, error) {
	return decodeRecNBPLeaf(raw, "KMBO")
}

func runCppKmNBP(fc kmNBPLeaf) (nbpLeafOut, error) {
	out, _, err := runCppKmNBPBench(fc, 0)
	return out, err
}

func runCppKmNBPBench(fc kmNBPLeaf, benchIters int) (nbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpLeafOut{}, 0, err
	}
	payload, err := encodeKmNBP(fc)
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
		return nbpLeafOut{}, 0, fmt.Errorf("cpp kmeans nbp: %v: %s", err, stderr.String())
	}
	return decodeKmNBP(stdout.Bytes())
}

func makeKmLeafCase(cfg kmeans.Config, batch, seq int, dt core.DType, seed uint32) (kmNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return kmNBPLeaf{}, err
	}
	if seq <= 0 {
		seq = 1
	}
	fc := kmNBPLeaf{
		batch: batch, seq: seq, cfg: cfg, dtype: dt, lr: 1e-2,
		W: randInit(cfg.NumClusters, cfg.FeatureDim, seed),
		x: randInit(batch*seq, cfg.FeatureDim, seed^0x9e3779b9),
	}
	fc.target = make([]float32, kmOutLen(batch, seq, cfg))
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

func caseKmNBPLeafProbsF32() error {
	return caseKmNBPLeafExact(smokeKmProbsCfg(), 0x4b4d4250, 40)
}

func caseKmNBPLeafFeaturesF32() error {
	return caseKmNBPLeafExact(smokeKmFeaturesCfg(), 0x4b4d4650, 40)
}

func caseKmNBPLeafExact(cfg kmeans.Config, seed uint32, benchIters int) error {
	fc, err := makeKmLeafCase(cfg, 2, 3, core.DTypeFloat32, seed)
	if err != nil {
		return err
	}
	goO, err := runGoKmNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppKmNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, err := runGoKmNBP(fc)
		return err
	}, benchIters)
	if err != nil {
		return fmt.Errorf("go bench: %w", err)
	}
	_, cppDur, err := runCppKmNBPBench(fc, benchIters)
	if err != nil {
		return fmt.Errorf("cpp bench: %w", err)
	}
	cppUS := perIterUS(cppDur, benchIters)
	spdPct := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%  (neg ⇒ C++ faster)",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseKmNBPLeafProbsAllDTypes() error {
	const benchIters = defaultGoBenchIters
	cfg := smokeKmProbsCfg()
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — KMeans NBP leaf probs ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc, err := makeKmLeafCase(cfg, 2, 3, dt, 0x4b4d4434)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		goO, err := runGoKmNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppKmNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		mW := suites.CompareF32Spectrum(goO.WAfter, cppO.WAfter)
		mDX := suites.CompareF32Spectrum(goO.dX, cppO.dX)
		mDW := suites.CompareF32Spectrum(goO.dW, cppO.dW)
		// W/dW must be exact_bit. dX may differ by ~1 ulp in float64 softMax
		// (Go math.Exp vs libm) when assignments are extremely peaked on discrete
		// dtypes — abs error stays at denormal scale (~1e-18).
		dxOK := mDX.Band == suites.DetExactBit || (mDX.MaxAbs < 1e-15 && mDX.ExactPct >= 85)
		band := mW.Band
		if mDW.Band.Percent() < band.Percent() {
			band = mDW.Band
		}
		if !dxOK && mDX.Band.Percent() < band.Percent() {
			band = mDX.Band
		}
		if band.Percent() < worst.Percent() {
			worst = band
		}
		goUS, err := benchGoUS(func() error {
			_, err := runGoKmNBP(fc)
			return err
		}, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
			continue
		}
		_, cppDur, err := runCppKmNBPBench(fc, benchIters)
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
		if mW.Band != suites.DetExactBit || mDW.Band != suites.DetExactBit || !dxOK {
			fails = append(fails, fmt.Sprintf("%s: W=%s dW=%s dX=%s", dt, mW.Band, mDW.Band, mDX.Band))
		}
	}
	fmt.Printf("      worst_band=%s — want exact W/dW; dX exact or |Δ|<1e-15\n", worst)
	if len(fails) > 0 {
		return fmt.Errorf("%d dtype issues: %s", len(fails), strings.Join(fails, " | "))
	}
	return nil
}

func runGoKmVolNBP(depth, rows, cols, lpc, batch int, cells []kmVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
			firstD, firstSeq = vc.cfg.FeatureDim, vc.seq
		}
		sl, err := kmeans.NewConfigured(vc.cfg, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		disableKMeansTrain(sl)
		if err := kmeans.Place(g, c.Z, c.Y, c.X, c.L, sl); err != nil {
			return z, err
		}
	}
	var xin *core.Tensor[float32]
	if firstSeq <= 1 {
		xin = core.NewTensor[float32](batch, firstD)
	} else {
		xin = core.NewTensor[float32](batch, firstSeq, firstD)
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
		sl, ok := cell.Op.(*kmeans.Layer)
		if !ok {
			return z, fmt.Errorf("expected kmeans at %v", c)
		}
		wa, err := packKMeansWeightsF32(sl)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, wa)
	}
	return z, nil
}

func encodeKmVolNBP(depth, rows, cols, lpc, batch int, cells []kmVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("KMV1"))
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
		cfg := c.cfg
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.NumClusters))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.FeatureDim))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.seq))
		_ = binary.Write(&in, binary.LittleEndian, outputModeInt(cfg.OutputMode))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Activation))
		_ = binary.Write(&in, binary.LittleEndian, cfg.Temperature)
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

func decodeKmVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
	return decodeRecVolNBP(raw, "KMVO")
}

func runCppKmVolNBP(payload []byte, benchIters int) (nbpVolOut, time.Duration, error) {
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp kmeans vol nbp: %v: %s", err, stderr.String())
	}
	return decodeKmVolNBP(stdout.Bytes())
}

func benchKmVolNBP(depth, rows, cols, lpc, batch int, cells []kmVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	payload, err := encodeKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS, err = benchGoUS(func() error {
		_, err := runGoKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
		return err
	}, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	_, cppDur, err := runCppKmVolNBP(payload, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	cppUS = perIterUS(cppDur, benchIters)
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func makeKmVolCell(cfg kmeans.Config, seq int, dt core.DType, seed uint32) kmVolCell {
	return kmVolCell{
		enabled: true,
		cfg:     cfg,
		seq:     seq,
		dtype:   dt,
		W:       randInit(cfg.NumClusters, cfg.FeatureDim, seed),
	}
}

func caseKmNBPVol1x1x1() error {
	const batch, benchIters, lr = 2, defaultGoBenchIters, 1e-2
	cfg := volKmCfg()
	seq := 2
	W := randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d5631)
	cells := []kmVolCell{{
		enabled: true, cfg: cfg, seq: seq,
		dtype: core.DTypeFloat32, W: W,
	}}
	x := randInit(batch*seq, cfg.FeatureDim, 0x4b4d5632)
	target := make([]float32, kmOutLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.03 * float64(i%4+1))
	}
	goO, err := runGoKmVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeKmVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	cppO, _, err := runCppKmVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchKmVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseKmNBPVolChain() error {
	const batch, benchIters, lr = 1, defaultGoBenchIters, 1e-2
	cfg := volKmCfg()
	seq := 2
	W0 := randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d5633)
	W1 := randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d5634)
	cells := []kmVolCell{
		{enabled: true, cfg: cfg, seq: seq, dtype: core.DTypeFloat32, W: W0},
		{enabled: true, cfg: cfg, seq: seq, dtype: core.DTypeFloat32, W: W1},
	}
	x := randInit(batch*seq, cfg.FeatureDim, 0x4b4d5635)
	target := make([]float32, kmOutLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.02 * float64(i%5+1))
	}
	goO, err := runGoKmVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeKmVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	cppO, _, err := runCppKmVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchKmVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseKmNBPVol111AllDTypes() error {
	const batch = 2
	cfg := volKmCfg()
	seq := 2
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      KMeans NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []kmVolCell{makeKmVolCell(cfg, seq, dt, 0x4b4d5636)}
		x := randInit(batch*seq, cfg.FeatureDim, 0x4b4d5637)
		target := make([]float32, kmOutLen(batch, seq, cfg))
		for i := range target {
			target[i] = float32(0.03 * float64(i%4+1))
		}
		goO, err := runGoKmVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeKmVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppKmVolNBP(payload, 0)
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

func caseKmNBPVol222F32() error {
	const depth, rows, cols, lpc, batch, benchIters = 2, 2, 2, 1, 1, defaultGoBenchIters
	stack := depth * rows * cols * lpc
	const lr = 1e-2
	cfg := volKmCfg()
	seq := 2
	W := randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d5638)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, cfg.FeatureDim, 0x4b4d5639)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, kmOutLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	cells := make([]kmVolCell, stack)
	for i := range cells {
		cells[i] = kmVolCell{
			enabled: true, cfg: cfg, seq: seq,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...),
		}
	}
	goO, err := runGoKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, _ := encodeKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
	cppO, _, err := runCppKmVolNBP(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		goO.loss, FormatUS(goUS), FormatUS(cppUS), spdPct)
	return nil
}

func caseKmNBPVol211AllDTypes() error {
	const depth, rows, cols, lpc, batch = 2, 1, 1, 1, 1
	stack := depth * rows * cols * lpc
	cfg := volKmCfg()
	seq := 2
	W := randInit(cfg.NumClusters, cfg.FeatureDim, 0x4b4d563a)
	for i := range W {
		W[i] *= 0.05
	}
	x := randInit(batch*seq, cfg.FeatureDim, 0x4b4d563b)
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, kmOutLen(batch, seq, cfg))
	for i := range target {
		target[i] = float32(0.05 * float64(i+1))
	}
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      KMeans NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := make([]kmVolCell, stack)
		for i := range cells {
			cells[i] = kmVolCell{
				enabled: true, cfg: cfg, seq: seq,
				dtype: dt, W: append([]float32(nil), W...),
			}
		}
		goO, err := runGoKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		payload, _ := encodeKmVolNBP(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		cppO, _, err := runCppKmVolNBP(payload, 0)
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


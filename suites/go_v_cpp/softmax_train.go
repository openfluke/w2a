package go_v_cpp

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
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/runtime/backward"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/training"
)

// CasesSoftmaxTrain — NormalBP Softmax Go↔C++ (leaf all Kind + volumetric).
func CasesSoftmaxTrain() []Case {
	return []Case{
		{Name: "Softmax NormalBP leaf Standard Float32 — exact bits (dX/loss)", Run: caseSoftmaxNBPLeafStandard},
		{Name: "Softmax NormalBP leaf all Kind Float32 — exact bits (+ gumbel smoke)", Run: caseSoftmaxNBPLeafAllKinds},
		{Name: "Softmax NormalBP leaf Standard × all 34 dtypes — spectrum + speed", Run: caseSoftmaxNBPLeafAllDTypes},
		{Name: "Softmax NormalBP leaf all Kind × all 34 dtypes — spectrum", Run: caseSoftmaxNBPLeafAllKindsAllDTypes},
		{Name: "Softmax NormalBP volume 1×1×1×1 Float32 — exact bits", Run: caseSoftmaxNBPVol1x1x1},
		{Name: "Softmax NormalBP volume 1×1×1×2 chain Float32 — exact bits", Run: caseSoftmaxNBPVolChain},
		{Name: "Softmax NormalBP volume 1×1×1×1 × all 34 dtypes — spectrum", Run: caseSoftmaxNBPVol111AllDTypes},
		{Name: "Softmax NormalBP volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxNBPVol222F32},
		{Name: "Softmax NormalBP volume 2×1×1×1 × all 34 dtypes — spectrum", Run: caseSoftmaxNBPVol211AllDTypes},
	}
}

func RunSoftmaxTrain() error {
	var fails []string
	cs := CasesSoftmaxTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [smbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_softmax_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_softmax_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp softmax_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type smbpLeaf struct {
	cfg    softmax.Config
	dtype  core.DType
	shape  []int
	x      []float32
	target []float32
	lr     float64
}

type smbpLeafOut struct {
	pre, post, dX, dW, WAfter []float32
	loss                      float64
}

func smbpRandInit(rows, cols int, seed uint32) []float32 {
	return randInit(rows, cols, seed)
}

func runGoSoftmaxNBP(fc smbpLeaf) (smbpLeafOut, error) {
	var z smbpLeafOut
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return z, err
	}
	l, err := softmax.New(cfg)
	if err != nil {
		return z, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	_ = l.SetDType(fc.dtype)

	xin := core.NewTensor[float32](fc.shape...)
	copy(xin.Data, fc.x)
	preT, postT, err := softmax.Forward(l, xin)
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
	dX, dW, err := softmax.Backward(l, gy, xin, preT)
	if err != nil {
		return z, err
	}
	if fc.lr != 0 {
		if err := softmax.ApplyGradSGD(l, dW, fc.lr); err != nil {
			return z, err
		}
	}
	z.pre = append([]float32(nil), preT.Data...)
	z.post = append([]float32(nil), postT.Data...)
	z.dX = append([]float32(nil), dX.Data...)
	if dW != nil {
		z.dW = append([]float32(nil), dW.Data...)
	}
	z.WAfter = nil
	z.loss = loss
	return z, nil
}

func encodeSoftmaxNBP(fc smbpLeaf) ([]byte, error) {
	cfg := fc.cfg
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	n := len(fc.x)
	var in bytes.Buffer
	_, _ = in.Write([]byte("WXB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(n))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Dim))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Kind))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Rows))
	_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Cols))
	_ = binary.Write(&in, binary.LittleEndian, cfg.Temperature)
	_ = binary.Write(&in, binary.LittleEndian, cfg.EntmaxAlpha)
	_ = binary.Write(&in, binary.LittleEndian, int32(0)) // gumbel_seed
	_ = binary.Write(&in, binary.LittleEndian, int32(fc.dtype))
	_ = binary.Write(&in, binary.LittleEndian, int32(0)) // cpu
	_ = binary.Write(&in, binary.LittleEndian, int32(len(fc.shape)))
	for _, d := range fc.shape {
		_ = binary.Write(&in, binary.LittleEndian, int32(d))
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(len(cfg.Mask)))
	for _, m := range cfg.Mask {
		var b uint8
		if m {
			b = 1
		}
		_ = binary.Write(&in, binary.LittleEndian, b)
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(len(cfg.HierarchyLevels)))
	for _, h := range cfg.HierarchyLevels {
		_ = binary.Write(&in, binary.LittleEndian, int32(h))
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

func decodeSoftmaxNBP(raw []byte) (smbpLeafOut, time.Duration, error) {
	var z smbpLeafOut
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
	if len(raw) < 8 || string(raw[:4]) != "WXBO" {
		return z, bench, fmt.Errorf("cpp softmax nbp bad magic")
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

func runCppSoftmaxNBP(fc smbpLeaf) (smbpLeafOut, error) {
	out, _, err := runCppSoftmaxNBPBench(fc, 0)
	return out, err
}

func runCppSoftmaxNBPBench(fc smbpLeaf, benchIters int) (smbpLeafOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return smbpLeafOut{}, 0, err
	}
	payload, err := encodeSoftmaxNBP(fc)
	if err != nil {
		return smbpLeafOut{}, 0, err
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
		return smbpLeafOut{}, 0, fmt.Errorf("cpp softmax nbp: %v: %s", err, stderr.String())
	}
	return decodeSoftmaxNBP(stdout.Bytes())
}

func compareSoftmaxNBP(goO, cppO smbpLeafOut, requireExact bool) error {
	checks := []struct {
		name string
		a, b []float32
	}{
		{"pre", goO.pre, cppO.pre},
		{"post", goO.post, cppO.post},
		{"dX", goO.dX, cppO.dX},
		{"dW", goO.dW, cppO.dW},
		{"W_after", goO.WAfter, cppO.WAfter},
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
	if goO.loss != cppO.loss {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact && rel > 0 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
		if !requireExact && rel > 1e-5 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

func makeSoftmaxNBPTarget(n int, seed uint32) []float32 {
	t := make([]float32, n)
	var s = seed
	for i := range t {
		s = s*1664525 + 1013904223
		t[i] = float32(int32(s%1001)) / 1000.0
	}
	return t
}

func caseSoftmaxNBPLeafStandard() error {
	const benchIters = 40
	cfg := softmax.Config{Dim: 8, SeqLen: 2, Kind: softmax.KindStandard, Temperature: 1}
	x := smbpRandInit(2, 8, 0x534d4250)
	fc := smbpLeaf{
		cfg: cfg, dtype: core.DTypeFloat32, shape: []int{2, 8}, x: x,
		target: makeSoftmaxNBPTarget(16, 0x534d4251), lr: 1e-2,
	}
	goO, err := runGoSoftmaxNBP(fc)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSoftmaxNBP(fc)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareSoftmaxNBP(goO, cppO, true); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if _, err := runGoSoftmaxNBP(fc); err != nil {
			return err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoSoftmaxNBP(fc); err != nil {
			return err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppSoftmaxNBPBench(fc, benchIters)
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

func softmaxNBPKindFixtures() []smKindFixture {
	return softmaxKindFixtures()
}

func caseSoftmaxNBPLeafAllKinds() error {
	for _, fx := range softmaxNBPKindFixtures() {
		n := len(fx.x)
		fc := smbpLeaf{
			cfg: fx.cfg, dtype: core.DTypeFloat32, shape: fx.shape, x: fx.x,
			target: makeSoftmaxNBPTarget(n, 0x534d4b00), lr: 1e-2,
		}
		goO, err := runGoSoftmaxNBP(fc)
		if err != nil {
			return fmt.Errorf("%s go: %w", fx.name, err)
		}
		cppO, err := runCppSoftmaxNBP(fc)
		if err != nil {
			return fmt.Errorf("%s cpp: %w", fx.name, err)
		}
		if err := compareSoftmaxNBP(goO, cppO, true); err != nil {
			return fmt.Errorf("%s: %w", fx.name, err)
		}
		fmt.Printf("\n      %s exact_bit loss=%.6g", fx.name, goO.loss)
	}
	// Gumbel: smoke only (non-deterministic vs Go math/rand)
	cfg := softmax.Config{Dim: 6, SeqLen: 2, Kind: softmax.KindGumbel, Temperature: 1}
	x := smbpRandInit(2, 6, 0x534d4755)
	fc := smbpLeaf{
		cfg: cfg, dtype: core.DTypeFloat32, shape: []int{2, 6}, x: x,
		target: makeSoftmaxNBPTarget(12, 0x534d4756), lr: 1e-2,
	}
	goO, err := runGoSoftmaxNBP(fc)
	if err != nil {
		return fmt.Errorf("gumbel go: %w", err)
	}
	if err := checkProbRow(goO.post, cfg.Dim); err != nil {
		return fmt.Errorf("gumbel go probs: %w", err)
	}
	for _, v := range goO.dX {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("gumbel go dX non-finite")
		}
	}
	cppO, err := runCppSoftmaxNBP(fc)
	if err != nil {
		return fmt.Errorf("gumbel cpp: %w", err)
	}
	if err := checkProbRow(cppO.post, cfg.Dim); err != nil {
		return fmt.Errorf("gumbel cpp probs: %w", err)
	}
	for _, v := range cppO.dX {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("gumbel cpp dX non-finite")
		}
	}
	fmt.Printf("\n      gumbel smoke: go+cpp finite probs+dX (no exact_bit)")
	return nil
}

func caseSoftmaxNBPLeafAllDTypes() error {
	const benchIters = 20
	cfg := softmax.Config{Dim: 8, SeqLen: 2, Kind: softmax.KindStandard, Temperature: 1}
	x := smbpRandInit(2, 8, 0x534d4434)
	target := makeSoftmaxNBPTarget(16, 0x534d4435)
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Go CPU vs C++ CPU — Softmax NBP Standard ×34\n")
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s %s\n",
		"dtype", "band", "exact%", "go_µs", "cpp_µs", "spd%Δ", "max|Δ|")
	fmt.Printf("      %s\n", strings.Repeat("-", 80))
	for _, dt := range core.AllDTypes {
		fc := smbpLeaf{
			cfg: cfg, dtype: dt, shape: []int{2, 8}, x: x, target: target, lr: 1e-2,
		}
		goO, err := runGoSoftmaxNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppSoftmaxNBP(fc)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		mDX := suites.CompareF32Spectrum(goO.dX, cppO.dX)
		mPost := suites.CompareF32Spectrum(goO.post, cppO.post)
		band := mDX.Band
		if mPost.Band.Percent() < band.Percent() {
			band = mPost.Band
		}
		if band.Percent() < worst.Percent() {
			worst = band
		}
		for i := 0; i < 3; i++ {
			if _, err := runGoSoftmaxNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go warm: %v", dt, err))
				break
			}
		}
		t0 := time.Now()
		for i := 0; i < benchIters; i++ {
			if _, err := runGoSoftmaxNBP(fc); err != nil {
				fails = append(fails, fmt.Sprintf("%s go bench: %v", dt, err))
				break
			}
		}
		goDur := time.Since(t0)
		_, cppDur, err := runCppSoftmaxNBPBench(fc, benchIters)
		goUS := float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp bench: %v", dt, err))
			fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10s %9s %s\n",
				dt, band, mDX.ExactPct, goUS, "?", "?", fmt.Sprintf("%.3g", mDX.MaxAbs))
			continue
		}
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spdPct := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %+8.1f%% %s\n",
			dt, band, mDX.ExactPct, goUS, cppUS, spdPct, fmt.Sprintf("%.3g", mDX.MaxAbs))
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

func caseSoftmaxNBPLeafAllKindsAllDTypes() error {
	const benchIters = 10
	var allFails []string
	worstBand := suites.DetExactBit
	for _, fx := range softmaxNBPKindFixtures() {
		n := len(fx.x)
		target := makeSoftmaxNBPTarget(n, 0x534d4b34)
		fmt.Printf("\n      Softmax NBP %s ×34\n", fx.name)
		for _, dt := range core.AllDTypes {
			fc := smbpLeaf{
				cfg: fx.cfg, dtype: dt, shape: fx.shape, x: fx.x, target: target, lr: 1e-2,
			}
			goO, err := runGoSoftmaxNBP(fc)
			if err != nil {
				allFails = append(allFails, fmt.Sprintf("%s/%s go: %v", fx.name, dt, err))
				continue
			}
			cppO, err := runCppSoftmaxNBP(fc)
			if err != nil {
				allFails = append(allFails, fmt.Sprintf("%s/%s cpp: %v", fx.name, dt, err))
				continue
			}
			m := suites.CompareF32Spectrum(goO.dX, cppO.dX)
			mP := suites.CompareF32Spectrum(goO.post, cppO.post)
			if mP.Band.Percent() < m.Band.Percent() {
				m = mP
			}
			if m.Band.Percent() < worstBand.Percent() {
				worstBand = m.Band
			}
			if m.Band != suites.DetExactBit {
				allFails = append(allFails, fmt.Sprintf("%s/%s: %s", fx.name, dt, suites.FormatSpectrum(m)))
			}
			_ = benchIters
		}
	}
	fmt.Printf("\n      Kind×34 worst_band=%s\n", worstBand)
	if len(allFails) > 0 {
		return fmt.Errorf("%d kind×dtype issues: %s", len(allFails), strings.Join(allFails, " | "))
	}
	return nil
}

type smVolCell struct {
	enabled bool
	cfg     softmax.Config
	dtype   core.DType
}

func runGoSoftmaxVolNBP(depth, rows, cols, lpc, batch int, cells []smVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
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
		if err := cfg.Validate(); err != nil {
			return z, err
		}
		sl, err := softmax.New(cfg)
		if err != nil {
			return z, err
		}
		sl.Exec.Backend = core.BackendCPUTiled
		sl.Exec.MultiCore = false
		sl.Core.MultiCore = false
		_ = sl.SetDType(vc.dtype)
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerSoftmax, DType: vc.dtype, Activation: core.ActivationLinear,
			InputHeight: cfg.Dim, OutputHeight: cfg.Dim,
		}, sl); err != nil {
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
	for i := range hops {
		if !cells[i].enabled {
			continue
		}
		z.Ws = append(z.Ws, []float32{}) // weightless
	}
	return z, nil
}

func encodeSoftmaxVolNBP(depth, rows, cols, lpc, batch int, cells []smVolCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("WXV1"))
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
		cfg := c.cfg
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Dim))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.SeqLen))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Kind))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Rows))
		_ = binary.Write(&in, binary.LittleEndian, int32(cfg.Cols))
		_ = binary.Write(&in, binary.LittleEndian, cfg.Temperature)
		_ = binary.Write(&in, binary.LittleEndian, cfg.EntmaxAlpha)
		_ = binary.Write(&in, binary.LittleEndian, int32(0)) // gumbel_seed
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		_ = binary.Write(&in, binary.LittleEndian, int32(len(cfg.Mask)))
		for _, m := range cfg.Mask {
			var b uint8
			if m {
				b = 1
			}
			_ = binary.Write(&in, binary.LittleEndian, b)
		}
		_ = binary.Write(&in, binary.LittleEndian, int32(len(cfg.HierarchyLevels)))
		for _, h := range cfg.HierarchyLevels {
			_ = binary.Write(&in, binary.LittleEndian, int32(h))
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

func decodeSoftmaxVolNBP(raw []byte) (nbpVolOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != "WXVO" {
		return z, bench, fmt.Errorf("cpp softmax vol nbp bad magic")
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

func runCppSoftmaxVolNBP(depth, rows, cols, lpc, batch int, cells []smVolCell, x, target []float32, lr float64) (nbpVolOut, error) {
	out, _, err := runCppSoftmaxVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, 0)
	return out, err
}

func runCppSoftmaxVolNBPBench(depth, rows, cols, lpc, batch int, cells []smVolCell, x, target []float32, lr float64, benchIters int) (nbpVolOut, time.Duration, error) {
	tool, err := findDenseFwd()
	if err != nil {
		return nbpVolOut{}, 0, err
	}
	payload, err := encodeSoftmaxVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr)
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
		return nbpVolOut{}, 0, fmt.Errorf("cpp softmax vol nbp: %v: %s", err, stderr.String())
	}
	return decodeSoftmaxVolNBP(stdout.Bytes())
}

func benchSoftmaxVolNBP(depth, rows, cols, lpc, batch int, cells []smVolCell, x, target []float32, lr float64, benchIters int) (goUS, cppUS, spdPct float64, err error) {
	for i := 0; i < 3; i++ {
		if _, err := runGoSoftmaxVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	t0 := time.Now()
	for i := 0; i < benchIters; i++ {
		if _, err := runGoSoftmaxVolNBP(depth, rows, cols, lpc, batch, cells, x, target, lr); err != nil {
			return 0, 0, 0, err
		}
	}
	goDur := time.Since(t0)
	_, cppDur, err := runCppSoftmaxVolNBPBench(depth, rows, cols, lpc, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return 0, 0, 0, err
	}
	goUS = float64(goDur.Nanoseconds()) / float64(benchIters) / 1000
	cppUS = float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
	spdPct = suites.SpeedPctDiff(goUS, cppUS)
	return goUS, cppUS, spdPct, nil
}

func caseSoftmaxNBPVol1x1x1() error {
	const batch, dim, benchIters = 2, 4, 40
	cfg := softmax.Config{Dim: dim, SeqLen: 1, Kind: softmax.KindStandard, Temperature: 1}
	x := smbpRandInit(batch, dim, 0x534d5631)
	target := makeSoftmaxNBPTarget(batch*dim, 0x534d5632)
	cells := []smVolCell{{
		enabled: true, cfg: cfg, dtype: core.DTypeFloat32,
	}}
	const lr = 1e-2
	goO, err := runGoSoftmaxVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSoftmaxVolNBP(1, 1, 1, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchSoftmaxVolNBP(1, 1, 1, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseSoftmaxNBPVolChain() error {
	const batch, dim, benchIters = 1, 4, 40
	cfg := softmax.Config{Dim: dim, SeqLen: 1, Kind: softmax.KindStandard, Temperature: 1}
	x := smbpRandInit(batch, dim, 0x534d5633)
	target := makeSoftmaxNBPTarget(batch*dim, 0x534d5634)
	cells := []smVolCell{
		{enabled: true, cfg: cfg, dtype: core.DTypeFloat32},
		{enabled: true, cfg: cfg, dtype: core.DTypeFloat32},
	}
	const lr = 1e-2
	goO, err := runGoSoftmaxVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSoftmaxVolNBP(1, 1, 1, 2, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchSoftmaxVolNBP(1, 1, 1, 2, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      1×1×1×2 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseSoftmaxNBPVol111AllDTypes() error {
	const batch, dim = 2, 4
	cfg := softmax.Config{Dim: dim, SeqLen: 1, Kind: softmax.KindStandard, Temperature: 1}
	x := smbpRandInit(batch, dim, 0x534d5635)
	target := makeSoftmaxNBPTarget(batch*dim, 0x534d5636)
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Softmax NBP vol 1×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []smVolCell{{enabled: true, cfg: cfg, dtype: dt}}
		goO, err := runGoSoftmaxVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppSoftmaxVolNBP(1, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.gradIn, cppO.gradIn)
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

func caseSoftmaxNBPVol222F32() error {
	const batch, dim, benchIters = 1, 4, 20
	// scale logits so chained Softmax stays numerically calm
	x := smbpRandInit(batch, dim, 0x534d5637)
	for i := range x {
		x[i] *= 0.25
	}
	target := makeSoftmaxNBPTarget(batch*dim, 0x534d5638)
	cfg := softmax.Config{Dim: dim, SeqLen: 1, Kind: softmax.KindStandard, Temperature: 1}
	cells := make([]smVolCell, 8)
	for i := range cells {
		cells[i] = smVolCell{enabled: true, cfg: cfg, dtype: core.DTypeFloat32}
	}
	const lr = 1e-2
	goO, err := runGoSoftmaxVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	cppO, err := runCppSoftmaxVolNBP(2, 2, 2, 1, batch, cells, x, target, lr)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	if err := compareVolNBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, cppUS, spdPct, err := benchSoftmaxVolNBP(2, 2, 2, 1, batch, cells, x, target, lr, benchIters)
	if err != nil {
		return fmt.Errorf("bench: %w", err)
	}
	fmt.Printf("\n      2×2×2×1 loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		goO.loss, goUS, cppUS, spdPct)
	return nil
}

func caseSoftmaxNBPVol211AllDTypes() error {
	const batch, dim = 1, 4
	cfg := softmax.Config{Dim: dim, SeqLen: 1, Kind: softmax.KindStandard, Temperature: 1}
	x := smbpRandInit(batch, dim, 0x534d5639)
	for i := range x {
		x[i] *= 0.25
	}
	target := makeSoftmaxNBPTarget(batch*dim, 0x534d563a)
	var fails []string
	worst := suites.DetExactBit
	fmt.Printf("\n      Softmax NBP vol 2×1×1×1 ×34\n")
	for _, dt := range core.AllDTypes {
		cells := []smVolCell{
			{enabled: true, cfg: cfg, dtype: dt},
			{enabled: true, cfg: cfg, dtype: dt},
		}
		goO, err := runGoSoftmaxVolNBP(2, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go: %v", dt, err))
			continue
		}
		cppO, err := runCppSoftmaxVolNBP(2, 1, 1, 1, batch, cells, x, target, 1e-2)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp: %v", dt, err))
			continue
		}
		if err := compareVolNBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s: %v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.gradIn, cppO.gradIn)
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

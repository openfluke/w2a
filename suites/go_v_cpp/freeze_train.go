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
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/weights"
)

// CasesFreezeTrain — ModeFreeze (fwd + mse; no weight update) Go↔C++ for NormalBP-signed kinds.
// Parallel BranchMode Freeze (FZB1) lives here — same Shadow pattern (Parallel host under mode suite).
func CasesFreezeTrain() []Case {
	return []Case{
		// Parallel Freeze BranchMode (FZB1) — Dense Add Inside+Fully
		{Name: "Parallel Freeze∥NormalBP Float32 — exact bits", Run: caseParallelFreezeMixedF32},
		{Name: "Parallel parent ModeFreeze Float32 — exact bits (all W frozen)", Run: caseParallelFreezeParentF32},
		{Name: "Parallel Freeze∥NormalBP × all 34 dtypes — spectrum", Run: caseParallelFreezeMixedAllDTypes},
		{Name: "Parallel Freeze volume 2×2×2×1 Float32 — exact bits", Run: casePnFreezeVol222F32},

		{Name: "Dense Freeze multi-hop Float32 — exact bits (W/loss)", Run: caseDenseFreezeMultiHopF32},
		{Name: "Dense Freeze 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseFreeze1HopAllDTypes},
		{Name: "RMSNorm Freeze 1-hop Float32 — exact bits", Run: caseRMSNormFreezeF32},
		{Name: "RMSNorm Freeze 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormFreezeAllDTypes},
		{Name: "LayerNorm Freeze 1-hop Float32 — exact bits", Run: caseLayerNormFreezeF32},
		{Name: "LayerNorm Freeze 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormFreezeAllDTypes},
		{Name: "Softmax Freeze 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxFreezeF32},
		{Name: "Softmax Freeze 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxFreezeAllDTypes},
		{Name: "SwiGLU Freeze 1-hop Float32 — exact bits", Run: caseSwiGLUFreezeF32},
		{Name: "SwiGLU Freeze 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUFreezeAllDTypes},
		{Name: "Sequential Freeze 1-hop Float32 — exact bits", Run: caseSequentialFreezeF32},
		{Name: "Sequential Freeze 1-hop × all 34 dtypes — spectrum", Run: caseSequentialFreezeAllDTypes},
		{Name: "Residual Freeze 1-hop Float32 — exact bits", Run: caseResidualFreezeF32},
		{Name: "Residual Freeze 1-hop × all 34 dtypes — spectrum", Run: caseResidualFreezeAllDTypes},
		{Name: "Mamba Freeze 1-hop Float32 — exact bits", Run: caseMambaFreezeF32},
		{Name: "Mamba Freeze 1-hop × all 34 dtypes — spectrum", Run: caseMambaFreezeAllDTypes},
		{Name: "Metacognition Freeze 1-hop Float32 — exact bits", Run: caseMetacognitionFreezeF32},
		{Name: "Metacognition Freeze 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionFreezeAllDTypes},
		{Name: "KMeans Freeze 1-hop Float32 — exact bits", Run: caseKMeansFreezeF32},
		{Name: "KMeans Freeze 1-hop × all 34 dtypes — spectrum", Run: caseKMeansFreezeAllDTypes},
		{Name: "GDN Freeze 1-hop Float32 — exact bits", Run: caseGDNFreezeF32},
		{Name: "NestedStack Freeze 1-hop Float32 — exact bits", Run: caseNestedStackFreezeF32},
		{Name: "NestedStack Freeze 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackFreezeAllDTypes},
		{Name: "Embedding Freeze 1-hop Float32 — exact bits", Run: caseEmbeddingFreezeF32},
		{Name: "Embedding Freeze 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingFreezeAllDTypes},
		{Name: "MHA Freeze 1-hop Float32 — exact bits", Run: caseMHAFreezeF32},
		{Name: "MHA Freeze 1-hop × all 34 dtypes — spectrum", Run: caseMHAFreezeAllDTypes},
		{Name: "CNN1 Freeze 1-hop Float32 — exact bits", Run: caseCNN1FreezeF32},
		{Name: "CNN1 Freeze 1-hop × all 34 dtypes — spectrum", Run: caseCNN1FreezeAllDTypes},
		{Name: "ConvT1 Freeze 1-hop Float32 — exact bits", Run: caseConvT1FreezeF32},
		{Name: "ConvT1 Freeze 1-hop × all 34 dtypes — spectrum", Run: caseConvT1FreezeAllDTypes},
		{Name: "RNN Freeze 1-hop Float32 — exact bits", Run: caseRNNFreezeF32},
		{Name: "RNN Freeze 1-hop × all 34 dtypes — spectrum", Run: caseRNNFreezeAllDTypes},
		{Name: "LSTM Freeze 1-hop Float32 — exact bits", Run: caseLSTMFreezeF32},
		{Name: "LSTM Freeze 1-hop × all 34 dtypes — spectrum", Run: caseLSTMFreezeAllDTypes},
		{Name: "Dense Freeze volume 1×1×1×1 Float32 — exact bits", Run: caseDenseFreezeVol111F32},
		{Name: "Dense Freeze volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseFreezeVol211F32},
		{Name: "Dense Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseDenseFreezeVol222F32},
		{Name: "RMSNorm Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormFreezeVol222F32},
		{Name: "LayerNorm Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormFreezeVol222F32},
		{Name: "Softmax Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxFreezeVol222F32},
		{Name: "SwiGLU Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUFreezeVol222F32},
		{Name: "Sequential Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseSeqFreezeVol222F32},
		{Name: "Residual Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseReFreezeVol222F32},
		{Name: "Mamba Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseMbFreezeVol222F32},
		{Name: "Metacognition Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseMcFreezeVol222F32},
		{Name: "KMeans Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseKmFreezeVol222F32},
		{Name: "GDN Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseGdFreezeVol222F32},
		{Name: "NestedStack Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseSkFreezeVol222F32},
		{Name: "Embedding Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingFreezeVol222F32},
		{Name: "MHA Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseMHAFreezeVol222F32},
		{Name: "CNN1 Freeze volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNFreezeVol222F32(1) }},
		{Name: "CNN2 Freeze volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNFreezeVol222F32(2) }},
		{Name: "CNN3 Freeze volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNFreezeVol222F32(3) }},
		{Name: "ConvT1 Freeze volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTFreezeVol222F32(1) }},
		{Name: "ConvT2 Freeze volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTFreezeVol222F32(2) }},
		{Name: "ConvT3 Freeze volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTFreezeVol222F32(3) }},
		{Name: "RNN Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseRNNFreezeVol222F32},
		{Name: "LSTM Freeze volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMFreezeVol222F32},
	}
}

func RunFreezeTrain() error {
	var fails []string
	cs := CasesFreezeTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [freeze %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_freeze_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_freeze_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp freeze_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

// --- Parallel BranchMode Freeze (FZB1 / FZBO; FRP1 taken by TweenSplitFastProxy vol) ---

func caseParallelFreezeMixedF32() error {
	modes := []parallel.TrainMode{parallel.ModeFreeze, parallel.ModeNormalBP}
	return runCameralModesCaseParent("FZB1", "FZBO", modes, parallel.CamKit{},
		parallel.ModeNormalBP, true, core.DTypeFloat32, true, 40)
}

func caseParallelFreezeParentF32() error {
	// Parent ModeFreeze early-out: Inherit BranchModes → all frozen, W unchanged.
	modes := []parallel.TrainMode{parallel.ModeInherit, parallel.ModeInherit}
	return runCameralModesCaseParent("FZB1", "FZBO", modes, parallel.CamKit{},
		parallel.ModeFreeze, true, core.DTypeFloat32, true, 40)
}

func caseParallelFreezeMixedAllDTypes() error {
	modes := []parallel.TrainMode{parallel.ModeFreeze, parallel.ModeNormalBP}
	return runCameralModesSpectrumParent("Freeze∥NormalBP", "FZB1", "FZBO", modes,
		parallel.CamKit{}, parallel.ModeNormalBP, true)
}

type freezeOut struct {
	loss   float64
	WAfter [][]float32
}

func runCppFRB1(payload []byte, benchIters int) (freezeOut, time.Duration, error) {
	var z freezeOut
	var bench time.Duration
	tool, err := findDenseFwd()
	if err != nil {
		return z, 0, err
	}
	args := []string{}
	if benchIters > 0 {
		args = append(args, fmt.Sprintf("--bench=%d", benchIters))
	}
	cmd := exec.Command(tool, args...)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return z, 0, fmt.Errorf("cpp twb1: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
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
	if len(raw) < 8 || string(raw[:4]) != "FRBO" {
		return z, bench, fmt.Errorf("cpp tween bad magic")
	}
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	var nChildren int32
	if err := binary.Read(r, binary.LittleEndian, &nChildren); err != nil {
		return z, bench, err
	}
	z.WAfter = make([][]float32, nChildren)
	for i := 0; i < int(nChildren); i++ {
		var n int32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return z, bench, err
		}
		v := make([]float32, n)
		for j := range v {
			if err := binary.Read(r, binary.LittleEndian, &v[j]); err != nil {
				return z, bench, err
			}
		}
		z.WAfter[i] = v
	}
	return z, bench, nil
}

func compareFreeze(goO, cppO freezeOut, requireExact bool) error {
	if len(goO.WAfter) != len(cppO.WAfter) {
		return fmt.Errorf("n_children go=%d cpp=%d", len(goO.WAfter), len(cppO.WAfter))
	}
	for i := range goO.WAfter {
		m := suites.CompareF32Spectrum(goO.WAfter[i], cppO.WAfter[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
		if !requireExact && (m.Band == suites.DetBroken || m.Band == suites.DetBallpark) {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
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

// runFreezeSpectrum prints Dense-style dtype×speed table then requires exact_bit on each.
func runFreezeSpectrum(label string, build func(core.DType) (freezeOut, []byte, error)) error {
	const benchIters = defaultGoBenchIters
	fmt.Printf("\n      [%s Tween ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		goO, payload, err := build(dt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		_, cppDur, err := runCppFRB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp-bench:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		cppO, _, err := runCppFRB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		var m suites.DetMetrics
		if len(goO.WAfter) > 0 && len(goO.WAfter[0]) > 0 {
			m = suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		} else {
			// weightless (e.g. Softmax): band from loss bits
			m = suites.DetMetrics{Band: suites.DetExactBit, ExactPct: 100}
			if goO.loss != cppO.loss {
				m.Band = suites.DetBroken
				m.ExactPct = 0
			}
		}
		goUS, berr := benchGoUS(func() error {
			_, _, err := build(dt)
			return err
		}, benchIters)
		if berr != nil {
			fails = append(fails, fmt.Sprintf("%s go-bench:%v", dt, berr))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %8.1f%%\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cppUS), spd)
		if err := compareFreeze(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- Dense Tween ---

func encodeDenseFRB1(dtype core.DType, lr float64, children []denseStepChild, batch int, xShape, tShape []int32, ticksX, ticksT [][]float32) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("FRB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(children)))
	_ = binary.Write(&in, binary.LittleEndian, int32(len(ticksX)))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	for _, ch := range children {
		_ = binary.Write(&in, binary.LittleEndian, int32(1)) // Dense
		_ = binary.Write(&in, binary.LittleEndian, int32(ch.in))
		_ = binary.Write(&in, binary.LittleEndian, int32(ch.out))
		_ = binary.Write(&in, binary.LittleEndian, int32(ch.act))
		_ = binary.Write(&in, binary.LittleEndian, int32(dtype))
		for _, v := range ch.W {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		_ = binary.Write(&in, binary.LittleEndian, uint32(len(ch.bias)))
		for _, v := range ch.bias {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(len(xShape)))
	for _, d := range xShape {
		_ = binary.Write(&in, binary.LittleEndian, d)
	}
	_ = binary.Write(&in, binary.LittleEndian, int32(len(tShape)))
	for _, d := range tShape {
		_ = binary.Write(&in, binary.LittleEndian, d)
	}
	for i := range ticksX {
		for _, v := range ticksX[i] {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		for _, v := range ticksT[i] {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
	}
	return in.Bytes()
}

func runGoDenseFreeze(dtype core.DType, lr float64, children []denseStepChild, batch int, ticksX, ticksT [][]float32, mode parallel.TrainMode) (freezeOut, error) {
	var z freezeOut
	ops := make([]any, len(children))
	for i, ch := range children {
		l, err := dense.NewConfigured(ch.in, ch.out, ch.act, dtype, quant.FormatNone, ch.W)
		if err != nil {
			return z, err
		}
		l.Weights.Bias = append([]float64(nil), ch.bias...)
		disableDense(l)
		ops[i] = l
	}
	st, err := parallel.NewStack(ops...)
	if err != nil {
		return z, err
	}
	var loss float64
	for t := range ticksX {
		xin := core.NewTensor[float32](batch, children[0].in)
		copy(xin.Data, ticksX[t])
		tgt := core.NewTensor[float32](batch, children[len(children)-1].out)
		copy(tgt.Data, ticksT[t])
		loss, err = parallel.TrainStackMSE(st, xin, tgt, mode, lr)
		if err != nil {
			return z, err
		}
	}
	z.loss = loss
	z.WAfter = make([][]float32, len(children))
	for i, op := range st.Children {
		Wa, err := weights.MatrixF32(op.(*dense.Layer).Weights)
		if err != nil {
			return z, err
		}
		z.WAfter[i] = Wa
	}
	return z, nil
}

func caseDenseFreezeMultiHopF32() error {
	const batch = 2
	lr := 1e-2
	children := []denseStepChild{
		{in: 4, out: 3, act: core.ActivationReLU, W: []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1, 0.05, 0.15, -0.25, 0.35}, bias: []float64{0.1, -0.05, 0.02}},
		{in: 3, out: 3, act: core.ActivationLinear, W: []float32{0.2, -0.1, 0.05, 0.1, 0.3, -0.2, -0.15, 0.25, 0.4}, bias: []float64{0.01, -0.02, 0.03}},
		{in: 3, out: 2, act: core.ActivationReLU, W: []float32{0.3, -0.2, 0.1, -0.1, 0.4, 0.2}, bias: []float64{0.05, -0.05}},
	}
	x := []float32{1, 0.5, -0.5, 2, -1, 0.25, 0.75, -0.25}
	tgt := []float32{0.2, 0.4, 0.1, 0.3}
	ticksX := [][]float32{append([]float32(nil), x...)}
	ticksT := [][]float32{append([]float32(nil), tgt...)}
	goO, err := runGoDenseFreeze(core.DTypeFloat32, lr, children, batch, ticksX, ticksT, parallel.ModeFreeze)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseFRB1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, true)
}

func caseDenseFreeze1HopAllDTypes() error {
	const in, out, batch, benchIters = 6, 4, 2, defaultGoBenchIters
	W, x, bias := makeSharedInit(out, in, batch)
	tgt := make([]float32, batch*out)
	for i := range tgt {
		tgt[i] = float32(0.1*float64(i+1) + 0.05)
	}
	children := []denseStepChild{{in: in, out: out, act: core.ActivationLinear, W: W, bias: bias}}
	ticksX := [][]float32{x}
	ticksT := [][]float32{tgt}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		ch := []denseStepChild{{in: in, out: out, act: core.ActivationLinear, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...)}}
		tx := [][]float32{append([]float32(nil), x...)}
		tt := [][]float32{append([]float32(nil), tgt...)}
		goO, err := runGoDenseFreeze(dt, 1e-2, ch, batch, tx, tt, parallel.ModeFreeze)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseFRB1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		cppO, cppDur, err := runCppFRB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			continue
		}
		// fresh cpp once for compare (bench mutated)
		cppO, _, err = runCppFRB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp2:%v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS, berr := benchGoUS(func() error {
			_, err := runGoDenseFreeze(dt, 1e-2, ch, batch, tx, tt, parallel.ModeFreeze)
			return err
		}, benchIters)
		if berr != nil {
			fails = append(fails, fmt.Sprintf("%s go-bench:%v", dt, berr))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %8.1f%%\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cppUS), spd)
		if err := compareFreeze(goO, cppO, false); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
		_ = children
		_ = ticksX
		_ = ticksT
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- helper: 1-hop shape-preserving packs ---

func encodeFRB1Header(nChildren, nTicks, batch int, lr float64) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte("FRB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(nChildren))
	_ = binary.Write(&in, binary.LittleEndian, int32(nTicks))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return &in
}

func caseRMSNormFreezeF32() error {
	return runRMSNormFreeze(core.DTypeFloat32, true)
}

func caseRMSNormFreezeAllDTypes() error {
	return runFreezeSpectrum("RMSNorm", func(dt core.DType) (freezeOut, []byte, error) { return buildRMSNormFreeze(dt, parallel.ModeFreeze) })
}

func buildRMSNormFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	gamma, x := makeRMSGammaInit(dim, batch)
	tgt := make([]float32, batch*dim)
	for i := range tgt {
		tgt[i] = float32(0.05 * float64(i+1))
	}
	cfg := rmsnorm.Config{Dim: dim, Eps: 1e-6}
	l, err := rmsnorm.NewConfigured(cfg, dtype, quant.FormatNone, gamma)
	if err != nil {
		return freezeOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ga, err := weights.MatrixF32(st.Children[0].(*rmsnorm.Layer).Gamma)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{ga}}

	in := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(2)) // RMSNorm
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, 1e-6)
	for _, v := range gamma {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	writeShapesTicks(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runRMSNormFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildRMSNormFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseLayerNormFreezeF32() error {
	return runLayerNormFreeze(core.DTypeFloat32, true)
}

func caseLayerNormFreezeAllDTypes() error {
	return runFreezeSpectrum("LayerNorm", func(dt core.DType) (freezeOut, []byte, error) { return buildLayerNormFreeze(dt, parallel.ModeFreeze) })
}

func buildLayerNormFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	gamma, x := makeRMSGammaInit(dim, batch)
	beta := make([]float32, dim)
	for i := range beta {
		beta[i] = 0.01 * float32(i+1)
	}
	tgt := make([]float32, batch*dim)
	for i := range tgt {
		tgt[i] = float32(0.04 * float64(i+1))
	}
	cfg := layernorm.Config{Dim: dim, Eps: 1e-5}
	l, err := layernorm.NewConfigured(cfg, dtype, quant.FormatNone, gamma, beta)
	if err != nil {
		return freezeOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ll := st.Children[0].(*layernorm.Layer)
	gA, err := weights.MatrixF32(ll.Gamma)
	if err != nil {
		return freezeOut{}, nil, err
	}
	bA, err := weights.MatrixF32(ll.Beta)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{append(append([]float32(nil), gA...), bA...)}}

	in := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(3)) // LayerNorm
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, 1e-5)
	for _, v := range gamma {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	for _, v := range beta {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	writeShapesTicks(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runLayerNormFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildLayerNormFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseSoftmaxFreezeF32() error {
	return runSoftmaxFreeze(core.DTypeFloat32, true)
}

func caseSoftmaxFreezeAllDTypes() error {
	return runFreezeSpectrum("Softmax", func(dt core.DType) (freezeOut, []byte, error) { return buildSoftmaxFreeze(dt, parallel.ModeFreeze) })
}

func buildSoftmaxFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	x := []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1}
	tgt := []float32{0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4}
	cfg := softmax.Config{Dim: dim, Kind: softmax.KindStandard, Temperature: 1}
	l, err := softmax.New(cfg)
	if err != nil {
		return freezeOut{}, nil, err
	}
	_ = l.SetDType(dtype)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{{}}}

	in := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(5)) // Softmax
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, int32(0)) // Standard
	_ = binary.Write(in, binary.LittleEndian, 1.0)
	writeShapesTicks(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runSoftmaxFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildSoftmaxFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseSwiGLUFreezeF32() error {
	return runSwiGLUFreeze(core.DTypeFloat32, true)
}

func caseSwiGLUFreezeAllDTypes() error {
	return runFreezeSpectrum("SwiGLU", func(dt core.DType) (freezeOut, []byte, error) { return buildSwiGLUFreeze(dt, parallel.ModeFreeze) })
}

func buildSwiGLUFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	const batch, inDim, inter = 2, 4, 8
	lr := 1e-2
	ng := inter * inDim
	nu := inter * inDim
	nd := inDim * inter
	packed := make([]float32, ng+nu+nd)
	for i := range packed {
		packed[i] = float32(0.01*float64(i%17) - 0.05)
	}
	x := make([]float32, batch*inDim)
	tgt := make([]float32, batch*inDim)
	for i := range x {
		x[i] = float32(0.1 * float64(i+1))
		tgt[i] = float32(0.05 * float64(i+1))
	}
	cfg := swiglu.Config{InputDim: inDim, IntermediateDim: inter}
	l, err := swiglu.NewConfigured(cfg, dtype, quant.FormatNone, packed[:ng], packed[ng:ng+nu], packed[ng+nu:])
	if err != nil {
		return freezeOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, inDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, inDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	sl := st.Children[0].(*swiglu.Layer)
	gW, err := weights.MatrixF32(sl.Gate.Weights)
	if err != nil {
		return freezeOut{}, nil, err
	}
	uW, err := weights.MatrixF32(sl.Up.Weights)
	if err != nil {
		return freezeOut{}, nil, err
	}
	dW, err := weights.MatrixF32(sl.Down.Weights)
	if err != nil {
		return freezeOut{}, nil, err
	}
	wa := append(append(append([]float32(nil), gW...), uW...), dW...)
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(4)) // SwiGLU
	_ = binary.Write(buf, binary.LittleEndian, int32(inDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(inter))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, inDim}, []int32{batch, inDim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSwiGLUFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildSwiGLUFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseSequentialFreezeF32() error {
	return runSeqResFreeze(true, core.DTypeFloat32, true)
}

func caseSequentialFreezeAllDTypes() error {
	return runFreezeSpectrum("Sequential", func(dt core.DType) (freezeOut, []byte, error) {
		return buildSeqResFreeze(true, dt, parallel.ModeFreeze)
	})
}

func caseResidualFreezeF32() error {
	return runSeqResFreeze(false, core.DTypeFloat32, true)
}

func caseResidualFreezeAllDTypes() error {
	return runFreezeSpectrum("Residual", func(dt core.DType) (freezeOut, []byte, error) {
		return buildSeqResFreeze(false, dt, parallel.ModeFreeze)
	})
}

func buildSeqResFreeze(isSeq bool, dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	const batch, dim, depth, seq = 2, 4, 2, 1
	lr := 1e-2
	packed := make([]float32, depth*dim*dim)
	for i := range packed {
		packed[i] = float32(0.02*float64(i%11) - 0.1)
	}
	x := make([]float32, batch*dim)
	tgt := make([]float32, batch*dim)
	for i := range x {
		x[i] = float32(0.1 * float64(i+1))
		tgt[i] = float32(0.07 * float64(i+1))
	}
	var op any
	if isSeq {
		cfg := sequential.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := sequential.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return freezeOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	} else {
		cfg := residual.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := residual.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return freezeOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	}
	st, err := parallel.NewStack(op)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	var wa []float32
	if isSeq {
		sl := st.Children[0].(*sequential.Layer)
		for _, ch := range sl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return freezeOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	} else {
		rl := st.Children[0].(*residual.Layer)
		for _, ch := range rl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return freezeOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}

	kind := int32(16)
	if !isSeq {
		kind = 17
	}
	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, kind)
	_ = binary.Write(buf, binary.LittleEndian, int32(dim))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(depth))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSeqResFreeze(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResFreeze(isSeq, dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseMambaFreezeF32() error {
	return runMambaFreeze(core.DTypeFloat32, true)
}

func caseMambaFreezeAllDTypes() error {
	return runFreezeSpectrum("Mamba", func(dt core.DType) (freezeOut, []byte, error) { return buildMambaFreeze(dt, parallel.ModeFreeze) })
}

func buildMambaFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	const batch = 2
	lr := 1e-2
	// Same init family as Mamba NormalBP leaf spectrum (avoids pathological tiny-dtype packs).
	fc, err := makeMbLeafCase(cfg, batch, dtype, 0x53545042^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return freezeOut{}, nil, err
	}
	packed := fc.W
	x := fc.x
	tgt := fc.target
	lr = fc.lr
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	l, err := mamba.NewConfigured(cfg, dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return freezeOut{}, nil, err
	}
	disableMambaTrain(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*mamba.Layer)
	wa, err := packMambaWeightsF32(ml)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(18)) // Mamba
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DModel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DState))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Expand))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMambaFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildMambaFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseEmbeddingFreezeF32() error { return runEmbeddingFreeze(core.DTypeFloat32, true) }
func caseEmbeddingFreezeAllDTypes() error {
	return runFreezeSpectrum("Embedding", func(dt core.DType) (freezeOut, []byte, error) { return buildEmbeddingFreeze(dt, parallel.ModeFreeze) })
}

func buildEmbeddingFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	const batch, vocab, emb, seq = 2, 8, 4, 3
	lr := 1e-2
	W, x := makeEmbeddingTrainInit(vocab, emb, batch, seq, 0x5354454d)
	tgt := make([]float32, batch*seq*emb)
	for i := range tgt {
		tgt[i] = float32(0.03 * float64(i%5+1))
	}
	cfg := embedding.Config{VocabSize: vocab, EmbeddingDim: emb, SeqLen: seq}
	l, err := embedding.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return freezeOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, emb)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*embedding.Layer).Weights)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(6))
	_ = binary.Write(buf, binary.LittleEndian, int32(vocab))
	_ = binary.Write(buf, binary.LittleEndian, int32(emb))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, seq}, []int32{batch, seq, emb}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runEmbeddingFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildEmbeddingFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseMHAFreezeF32() error { return runMHAFreeze(core.DTypeFloat32, true) }
func caseMHAFreezeAllDTypes() error {
	return runFreezeSpectrum("MHA", func(dt core.DType) (freezeOut, []byte, error) { return buildMHAFreeze(dt, parallel.ModeFreeze) })
}

func buildMHAFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 16
	const batch, seq = 2, 4
	fc, err := makeMHANBPCase(cfg, batch, seq, dtype, 0x534d4841, 1e-2)
	if err != nil {
		return freezeOut{}, nil, err
	}
	c := fc.cfg // validated
	l, err := mha.NewConfigured(c, dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return freezeOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, c.DModel)
	copy(xin.Data, fc.x)
	target := core.NewTensor[float32](batch, seq, c.DModel)
	copy(target.Data, fc.target)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, fc.lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := exportMHAWeights(st.Children[0].(*mha.Layer))
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, fc.lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(7))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(c.DModel))
	_ = binary.Write(buf, binary.LittleEndian, int32(c.NumHeads))
	_ = binary.Write(buf, binary.LittleEndian, int32(c.NumKVHeads))
	_ = binary.Write(buf, binary.LittleEndian, int32(c.HeadDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(c.MaxSeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(c.Mask))
	_ = binary.Write(buf, binary.LittleEndian, int32(c.Pos))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, part := range [][]float32{fc.q, fc.k, fc.v, fc.o} {
		for _, v := range part {
			_ = binary.Write(buf, binary.LittleEndian, v)
		}
	}
	writeShapesTicks(buf, []int32{batch, seq, int32(c.DModel)}, []int32{batch, seq, int32(c.DModel)},
		[][]float32{fc.x}, [][]float32{fc.target})
	return goO, buf.Bytes(), nil
}

func runMHAFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildMHAFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseCNN1FreezeF32() error { return runCNN1Freeze(core.DTypeFloat32, true) }
func caseCNN1FreezeAllDTypes() error {
	return runFreezeSpectrum("CNN1", func(dt core.DType) (freezeOut, []byte, error) { return buildCNN1Freeze(dt, parallel.ModeFreeze) })
}

func buildCNN1Freeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	cfg := preservingCNN1()
	const batch = 2
	lr := 1e-2
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel
	W := randInit(1, nw, 0x53434e31)
	xLen := batch * cfg.InChannels * cfg.SeqLen
	x := randInit(1, xLen, 0x53434e32)
	outLen := cnn1.OutputLength(cfg.SeqLen, cfg.Kernel, cfg.Stride, cfg.Padding)
	tgt := make([]float32, batch*cfg.Filters*outLen)
	for i := range tgt {
		tgt[i] = float32(0.04 * float64(i%6+1))
	}
	l, err := cnn1.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return freezeOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn1.Layer).Proj.Weights)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(8))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InChannels))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Filters))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Kernel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Stride))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Padding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runCNN1Freeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildCNN1Freeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseConvT1FreezeF32() error { return runConvT1Freeze(core.DTypeFloat32, true) }
func caseConvT1FreezeAllDTypes() error {
	return runFreezeSpectrum("ConvT1", func(dt core.DType) (freezeOut, []byte, error) { return buildConvT1Freeze(dt, parallel.ModeFreeze) })
}

func buildConvT1Freeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	cfg := preservingConvT1()
	const batch = 2
	lr := 1e-2
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel
	W := randInit(1, nw, 0x53543131)
	xLen := batch * cfg.InChannels * cfg.SeqLen
	x := randInit(1, xLen, 0x53543132)
	outLen := convt1.OutputLength(cfg.SeqLen, cfg.Kernel, cfg.Stride, cfg.Padding, cfg.OutputPadding)
	tgt := make([]float32, batch*cfg.Filters*outLen)
	for i := range tgt {
		tgt[i] = float32(0.04 * float64(i%6+1))
	}
	l, err := convt1.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return freezeOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt1.Layer).Proj.Weights)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(11))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InChannels))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Filters))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Kernel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Stride))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Padding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.OutputPadding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runConvT1Freeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildConvT1Freeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseRNNFreezeF32() error { return runRNNFreeze(core.DTypeFloat32, true) }
func caseRNNFreezeAllDTypes() error {
	return runFreezeSpectrum("RNN", func(dt core.DType) (freezeOut, []byte, error) { return buildRNNFreeze(dt, parallel.ModeFreeze) })
}

func buildRNNFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	cfg := smokeRNNCfg()
	const batch = 2
	lr := 1e-2
	nw := cfg.WeightCount()
	W := randInit(1, nw, 0x53524e4e)
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x53524e58)
	tgt := make([]float32, batch*cfg.SeqLen*cfg.HiddenSize)
	for i := range tgt {
		tgt[i] = float32(0.05 * float64(i%7+1))
	}
	l, err := rnn.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return freezeOut{}, nil, err
	}
	disableRNN(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packRNNWeightsF32(st.Children[0].(*rnn.Layer))
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(14))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runRNNFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildRNNFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseLSTMFreezeF32() error { return runLSTMFreeze(core.DTypeFloat32, true) }
func caseLSTMFreezeAllDTypes() error {
	return runFreezeSpectrum("LSTM", func(dt core.DType) (freezeOut, []byte, error) { return buildLSTMFreeze(dt, parallel.ModeFreeze) })
}

func buildLSTMFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {

	cfg := smokeLSTMCfg()
	const batch = 2
	lr := 1e-2
	nw := cfg.WeightCount()
	W := randInit(1, nw, 0x534c5354)
	x := randInit(batch*cfg.SeqLen, cfg.InputSize, 0x534c5358)
	tgt := make([]float32, batch*cfg.SeqLen*cfg.HiddenSize)
	for i := range tgt {
		tgt[i] = float32(0.05 * float64(i%7+1))
	}
	l, err := lstm.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return freezeOut{}, nil, err
	}
	disableLSTMTrainGates(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packLSTMWeightsF32(st.Children[0].(*lstm.Layer))
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(15))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runLSTMFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildLSTMFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

// --- Dense Freeze volume (TWV1) ---

func encodeDenseFRV1(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("FRV1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0, 0} {
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
		_ = binary.Write(&in, binary.LittleEndian, int32(c.in))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.out))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.act))
		_ = binary.Write(&in, binary.LittleEndian, int32(c.dtype))
		for _, v := range c.W {
			_ = binary.Write(&in, binary.LittleEndian, v)
		}
		_ = binary.Write(&in, binary.LittleEndian, uint32(len(c.bias)))
		for _, v := range c.bias {
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

func runCppFRV1(payload []byte, benchIters int) (meshbpOut, time.Duration, error) {
	// reuse meshbpOut shape: loss + Ws
	var z meshbpOut
	var bench time.Duration
	tool, err := findDenseFwd()
	if err != nil {
		return z, 0, err
	}
	args := []string{}
	if benchIters > 0 {
		args = append(args, fmt.Sprintf("--bench=%d", benchIters))
	}
	cmd := exec.Command(tool, args...)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return z, 0, fmt.Errorf("cpp twv1: %v: %s", err, stderr.String())
	}
	raw := stdout.Bytes()
	if len(raw) >= 4+8+4 && string(raw[:4]) == "WVB1" {
		r := bytes.NewReader(raw[4:])
		var ns int64
		var iters int32
		_ = binary.Read(r, binary.LittleEndian, &ns)
		_ = binary.Read(r, binary.LittleEndian, &iters)
		rest, _ := io.ReadAll(r)
		raw = rest
		bench = time.Duration(ns)
	}
	if len(raw) < 8 || string(raw[:4]) != "FRVO" {
		return z, bench, fmt.Errorf("cpp tween vol bad magic")
	}
	r := bytes.NewReader(raw[4:])
	if err := binary.Read(r, binary.LittleEndian, &z.loss); err != nil {
		return z, bench, err
	}
	var nCells int32
	if err := binary.Read(r, binary.LittleEndian, &nCells); err != nil {
		return z, bench, err
	}
	for i := int32(0); i < nCells; i++ {
		var o, inn int32
		_ = binary.Read(r, binary.LittleEndian, &o)
		_ = binary.Read(r, binary.LittleEndian, &inn)
		w := make([]float32, o*inn)
		for j := range w {
			_ = binary.Read(r, binary.LittleEndian, &w[j])
		}
		z.Ws = append(z.Ws, w)
	}
	return z, bench, nil
}

func runGoDenseFreezeVol(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) (meshbpOut, error) {
	var z meshbpOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	g.Exec.MultiCore = false
	hops := g.HopOrder()
	for i, c := range hops {
		vc := cells[i]
		if !vc.enabled {
			if cell := g.At(c.Z, c.Y, c.X, c.L); cell != nil {
				cell.Layer.IsDisabled = true
			}
			continue
		}
		dl, err := dense.NewConfigured(vc.in, vc.out, vc.act, vc.dtype, quant.FormatNone, vc.W)
		if err != nil {
			return z, err
		}
		dl.Weights.Bias = append([]float64(nil), vc.bias...)
		disableDense(dl)
		if err := g.BindOp(c.Z, c.Y, c.X, c.L, core.Layer{
			Type: core.LayerDense, DType: vc.dtype, Activation: vc.act,
			InputHeight: vc.in, OutputHeight: vc.out,
		}, dl); err != nil {
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
	loss, err := applyModeFreezeVolume(g, fwd, tgt, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		Wa, err := weights.MatrixF32(cell.Op.(*dense.Layer).Weights)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), Wa...))
	}
	return z, nil
}


func runDenseFreezeVolCase(depth, rows, cols, lpc, batch, benchIters int) error {
	stack := depth * rows * cols * lpc
	dim := 4
	W, x, bias := makeSharedInit(dim, dim, batch)
	for i := range W {
		W[i] *= 0.05
	}
	for i := range x {
		x[i] *= 0.05
	}
	target := make([]float32, dim*batch)
	for i := range target {
		target[i] = float32(0.05 * float64(i%dim+1))
	}
	cells := make([]volCell, stack)
	for i := range cells {
		cells[i] = volCell{
			enabled: true, in: dim, out: dim, act: core.ActivationLinear,
			dtype: core.DTypeFloat32, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...),
		}
	}
	goO, err := runGoDenseFreezeVol(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeDenseFRV1(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	_, cppDur, err := runCppFRV1(payload, benchIters)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := runCppFRV1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp2: %w", err)
	}
	if err := compareMeshBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, _ = runGoDenseFreezeVol(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		return nil
	}, defaultGoBenchIters)
	if err != nil {
		return err
	}
	bi := benchIters
	if bi < 1 {
		bi = 1
	}
	cppUS := float64(cppDur.Nanoseconds()) / float64(bi) / 1000
	spd := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      Tween vol %dx%dx%dx%d loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		depth, rows, cols, lpc, goO.loss, FormatUS(goUS), FormatUS(cppUS), spd)
	return nil
}

func caseDenseFreezeVol111F32() error { return runDenseFreezeVolCase(1, 1, 1, 1, 1, defaultGoBenchIters) }
func caseDenseFreezeVol211F32() error { return runDenseFreezeVolCase(2, 1, 1, 1, 1, defaultGoBenchIters) }
func caseDenseFreezeVol222F32() error { return runDenseFreezeVolCase(2, 2, 2, 1, 1, defaultGoBenchIters) }

// --- CASCADE Meta/GDN/Stack leaf (auto) ---

func caseMetacognitionFreezeF32() error {
	return runMetacognitionFreeze(core.DTypeFloat32, true)
}

func caseMetacognitionFreezeAllDTypes() error {
	return runFreezeSpectrum("Metacognition", func(dt core.DType) (freezeOut, []byte, error) { return buildMetacognitionFreeze(dt, parallel.ModeFreeze) })
}

func buildMetacognitionFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d435450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return freezeOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMetacognitionFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseGDNFreezeF32() error {
	return runGDNFreeze(true)
}

func buildGDNFreeze(mode parallel.TrainMode) (freezeOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x47444e50)
	if err != nil {
		return freezeOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runGDNFreeze(exact bool) error {
	goO, payload, err := buildGDNFreeze(parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

func caseNestedStackFreezeF32() error {
	return runNestedStackFreeze(core.DTypeFloat32, true)
}

func caseNestedStackFreezeAllDTypes() error {
	return runFreezeSpectrum("NestedStack", func(dt core.DType) (freezeOut, []byte, error) { return buildNestedStackFreeze(dt, parallel.ModeFreeze) })
}

func buildNestedStackFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53544b50^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return freezeOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return freezeOut{}, nil, err
	}
	writeShapesTicks(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runNestedStackFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}

// --- CASCADE KMeans leaf (auto) ---

func caseKMeansFreezeF32() error {
	return runKMeansFreeze(core.DTypeFloat32, true)
}

func caseKMeansFreezeAllDTypes() error {
	return runFreezeSpectrum("KMeans", func(dt core.DType) (freezeOut, []byte, error) { return buildKMeansFreeze(dt, parallel.ModeFreeze) })
}

func buildKMeansFreeze(dtype core.DType, mode parallel.TrainMode) (freezeOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeKmModeLeaf(dtype, 0x4b4d5450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return freezeOut{}, nil, err
	}
	st, err := newKmStack(cfg, dtype, packed)
	if err != nil {
		return freezeOut{}, nil, err
	}
	outDim := cfg.OutDim()
	xin := core.NewTensor[float32](batch, seq, cfg.FeatureDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, outDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, mode, lr)
	if err != nil {
		return freezeOut{}, nil, fmt.Errorf("go: %w", err)
	}
	kl := st.Children[0].(*kmeans.Layer)
	wa, err := packKMeansWeightsF32(kl)
	if err != nil {
		return freezeOut{}, nil, err
	}
	goO := freezeOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeFRB1Header(1, 1, batch, lr)
	writeKmChild(buf, cfg, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.FeatureDim)},
		[]int32{int32(batch), int32(seq), int32(outDim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runKMeansFreeze(dtype core.DType, exact bool) error {
	goO, payload, err := buildKMeansFreeze(dtype, parallel.ModeFreeze)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFRB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareFreeze(goO, cppO, exact)
}


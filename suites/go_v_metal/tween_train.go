package go_v_metal

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
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
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
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesTweenTrain — ModeTween (projected local @ lr*0.5) (FamilyBP on pop) Go↔Metal for NormalBP-signed kinds.
func CasesTweenTrain() []Case {
	return []Case{
		{Name: "Dense Tween multi-hop Float32 — exact tensors; Metal loss band (W/loss)", Run: caseDenseTweenMultiHopF32},
		{Name: "Dense Tween 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseTween1HopAllDTypes},
		{Name: "RMSNorm Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseRMSNormTweenF32},
		{Name: "RMSNorm Tween 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormTweenAllDTypes},
		{Name: "LayerNorm Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseLayerNormTweenF32},
		{Name: "LayerNorm Tween 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormTweenAllDTypes},
		{Name: "Softmax Tween 1-hop Float32 — exact tensors; Metal loss band (loss)", Run: caseSoftmaxTweenF32},
		{Name: "Softmax Tween 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxTweenAllDTypes},
		{Name: "SwiGLU Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseSwiGLUTweenF32},
		{Name: "SwiGLU Tween 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUTweenAllDTypes},
		{Name: "Sequential Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseSequentialTweenF32},
		{Name: "Sequential Tween 1-hop × all 34 dtypes — spectrum", Run: caseSequentialTweenAllDTypes},
		{Name: "Residual Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseResidualTweenF32},
		{Name: "Residual Tween 1-hop × all 34 dtypes — spectrum", Run: caseResidualTweenAllDTypes},
		{Name: "Mamba Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseMambaTweenF32},
		{Name: "Mamba Tween 1-hop × all 34 dtypes — spectrum", Run: caseMambaTweenAllDTypes},
		{Name: "Metacognition Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseMetacognitionTweenF32},
		{Name: "Metacognition Tween 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionTweenAllDTypes},

		{Name: "CNN2 Tween rank 4 × all 34 dtypes", Run: func() error { return runTweenSpectrum("CNN2", buildCNN2Tween) }},
		{Name: "CNN3 Tween rank 5 × all 34 dtypes", Run: func() error { return runTweenSpectrum("CNN3", buildCNN3Tween) }},
		{Name: "ConvT2 Tween rank 4 × all 34 dtypes", Run: func() error { return runTweenSpectrum("ConvT2", buildConvT2Tween) }},
		{Name: "ConvT3 Tween rank 5 × all 34 dtypes", Run: func() error { return runTweenSpectrum("ConvT3", buildConvT3Tween) }},
		{Name: "GDN Tween 1-hop Float32 — ≥strict weights; Metal loss band", Run: caseGDNTweenF32},
		{Name: "NestedStack Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseNestedStackTweenF32},
		{Name: "NestedStack Tween 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackTweenAllDTypes},
		{Name: "Embedding Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseEmbeddingTweenF32},
		{Name: "Embedding Tween 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingTweenAllDTypes},
		{Name: "MHA Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseMHATweenF32},
		{Name: "MHA Tween 1-hop × all 34 dtypes — spectrum", Run: caseMHATweenAllDTypes},
		{Name: "CNN1 Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseCNN1TweenF32},
		{Name: "CNN1 Tween 1-hop × all 34 dtypes — spectrum", Run: caseCNN1TweenAllDTypes},
		{Name: "ConvT1 Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseConvT1TweenF32},
		{Name: "ConvT1 Tween 1-hop × all 34 dtypes — spectrum", Run: caseConvT1TweenAllDTypes},
		{Name: "RNN Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseRNNTweenF32},
		{Name: "RNN Tween 1-hop × all 34 dtypes — spectrum", Run: caseRNNTweenAllDTypes},
		{Name: "LSTM Tween 1-hop Float32 — exact tensors; Metal loss band", Run: caseLSTMTweenF32},
		{Name: "LSTM Tween 1-hop × all 34 dtypes — spectrum", Run: caseLSTMTweenAllDTypes},
		{Name: "RMSNorm Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runRMSNormTweenVol211DType) }},
		{Name: "LayerNorm Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runLayerNormTweenVol211DType) }},
		{Name: "SwiGLU Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runSwiGLUTweenVol211DType) }},
		{Name: "Softmax Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runSoftmaxTweenVol211DType) }},
		{Name: "Embedding Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runEmbeddingTweenVol211DType) }},
		{Name: "MHA Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runMHATweenVol211DType) }},
		{Name: "RNN Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runRNNTweenVol211DType) }},
		{Name: "LSTM Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runLSTMTweenVol211DType) }},
		{Name: "Seq Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runSeqTweenVol211DType) }},
		{Name: "Re Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runReTweenVol211DType) }},
		{Name: "Mb Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runMbTweenVol211DType) }},
		{Name: "Mc Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runMcTweenVol211DType) }},
		{Name: "Sk Tween volume 211 × all 34 dtypes", Run: func() error { return runTweenVolumeSpectrum(runSkTweenVol211DType) }},
		{Name: "CNN1 Tween volume 211 × all 34 dtypes", Run: func() error {
			return runTweenVolumeSpectrum(func(dtype core.DType) error { return runCNNTweenVol211DType(1, dtype) })
		}},
		{Name: "CNN2 Tween volume 211 × all 34 dtypes", Run: func() error {
			return runTweenVolumeSpectrum(func(dtype core.DType) error { return runCNNTweenVol211DType(2, dtype) })
		}},
		{Name: "CNN3 Tween volume 211 × all 34 dtypes", Run: func() error {
			return runTweenVolumeSpectrum(func(dtype core.DType) error { return runCNNTweenVol211DType(3, dtype) })
		}},
		{Name: "ConvT1 Tween volume 211 × all 34 dtypes", Run: func() error {
			return runTweenVolumeSpectrum(func(dtype core.DType) error { return runConvTTweenVol211DType(1, dtype) })
		}},
		{Name: "ConvT2 Tween volume 211 × all 34 dtypes", Run: func() error {
			return runTweenVolumeSpectrum(func(dtype core.DType) error { return runConvTTweenVol211DType(2, dtype) })
		}},
		{Name: "ConvT3 Tween volume 211 × all 34 dtypes", Run: func() error {
			return runTweenVolumeSpectrum(func(dtype core.DType) error { return runConvTTweenVol211DType(3, dtype) })
		}},
		{Name: "Dense Tween volume 211 × all 34 dtypes", Run: caseDenseTweenVolAllDTypes},
		{Name: "Dense Tween volume 1×1×1×1 Float32 — exact tensors; Metal loss band", Run: caseDenseTweenVol111F32},
		{Name: "Dense Tween volume 2×1×1×1 chain Float32 — exact tensors; Metal loss band", Run: caseDenseTweenVol211F32},
		{Name: "Dense Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseDenseTweenVol222F32},
		{Name: "RMSNorm Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseRMSNormTweenVol222F32},
		{Name: "LayerNorm Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseLayerNormTweenVol222F32},
		{Name: "Softmax Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseSoftmaxTweenVol222F32},
		{Name: "SwiGLU Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseSwiGLUTweenVol222F32},
		{Name: "Sequential Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseSeqTweenVol222F32},
		{Name: "Residual Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseReTweenVol222F32},
		{Name: "Mamba Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseMbTweenVol222F32},
		{Name: "Metacognition Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseMcTweenVol222F32},
		{Name: "GDN Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseGdTweenVol222F32},
		{Name: "NestedStack Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseSkTweenVol222F32},
		{Name: "Embedding Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseEmbeddingTweenVol222F32},
		{Name: "MHA Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseMHATweenVol222F32},
		{Name: "CNN1 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: func() error { return caseCNNTweenVol222F32(1) }},
		{Name: "CNN2 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: func() error { return caseCNNTweenVol222F32(2) }},
		{Name: "CNN3 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: func() error { return caseCNNTweenVol222F32(3) }},
		{Name: "ConvT1 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: func() error { return caseConvTTweenVol222F32(1) }},
		{Name: "ConvT2 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: func() error { return caseConvTTweenVol222F32(2) }},
		{Name: "ConvT3 Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: func() error { return caseConvTTweenVol222F32(3) }},
		{Name: "RNN Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseRNNTweenVol222F32},
		{Name: "LSTM Tween volume 2×2×2×1 Float32 — exact tensors; Metal loss band", Run: caseLSTMTweenVol222F32},
	}
}

func RunTweenTrain() error {
	var fails []string
	cs := CasesTweenTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [tween %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_metal_tween_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_metal_tween_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_metal tween_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type tweenOut struct {
	loss   float64
	WAfter [][]float32
}

func runMetalTWB1(payload []byte, benchIters int) (tweenOut, time.Duration, error) {
	var z tweenOut
	var bench time.Duration
	tool, err := findMetalFwd()
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
		return z, 0, fmt.Errorf("mtl twb1: %v: %s", err, stderr.String())
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
	if len(raw) < 8 || string(raw[:4]) != "TWBO" {
		return z, bench, fmt.Errorf("mtl tween bad magic")
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

func compareTween(goO, mtlO tweenOut, requireExact bool) error {
	if len(goO.WAfter) != len(mtlO.WAfter) {
		return fmt.Errorf("n_children go=%d mtl=%d", len(goO.WAfter), len(mtlO.WAfter))
	}
	for i := range goO.WAfter {
		m := suites.CompareF32Spectrum(goO.WAfter[i], mtlO.WAfter[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
		if !requireExact && (m.Band == suites.DetBroken || m.Band == suites.DetBallpark) {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	if !metalLossWithinBand(goO.loss, mtlO.loss) {
		return fmt.Errorf("loss outside Metal float2 band: go=%g metal=%g", goO.loss, mtlO.loss)
	}

	return nil
}

// runTweenSpectrum prints Dense-style dtype×speed table then requires exact_bit on each.
func runTweenSpectrum(label string, build func(core.DType) (tweenOut, []byte, error)) error {
	var benchIters = defaultGoBenchIters
	fmt.Printf("\n      [%s Tween ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "mtlµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		goO, payload, err := build(dt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		benchO, mtlDur, err := runMetalTWB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s mtl-bench:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		mtlO, _, err := runMetalTWB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s mtl:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		if err := compareTween(mtlO, benchO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s benchmark reset: %v", dt, err))
		}
		var m suites.DetMetrics
		if len(goO.WAfter) > 0 && len(goO.WAfter[0]) > 0 {
			m = suites.CompareF32Spectrum(goO.WAfter[0], mtlO.WAfter[0])
		} else {
			// Weightless Softmax uses the finite Metal loss band.
			m = suites.DetMetrics{Band: suites.DetExactBit, ExactPct: 100}
			if !metalLossWithinBand(goO.loss, mtlO.loss) {
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
		mtlUS := perIterUS(mtlDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, mtlUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %8.1f%%\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(mtlUS), spd)
		if err := compareTween(goO, mtlO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- Dense Tween ---

func encodeDenseTWB1(dtype core.DType, lr float64, children []denseStepChild, batch int, xShape, tShape []int32, ticksX, ticksT [][]float32) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("TWB1"))
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

func runGoDenseTween(dtype core.DType, lr float64, children []denseStepChild, batch int, ticksX, ticksT [][]float32) (tweenOut, error) {
	var z tweenOut
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
		loss, err = parallel.TrainStackMSE(st, xin, tgt, parallel.ModeTween, lr)
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

func caseDenseTweenMultiHopF32() error {
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
	goO, err := runGoDenseTween(core.DTypeFloat32, lr, children, batch, ticksX, ticksT)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseTWB1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, true)
}

func caseDenseTween1HopAllDTypes() error {
	const in, out, batch = 6, 4, 2
	var benchIters = defaultGoBenchIters
	W, x, bias := makeSharedInit(out, in, batch)
	tgt := make([]float32, batch*out)
	for i := range tgt {
		tgt[i] = float32(0.1*float64(i+1) + 0.05)
	}
	children := []denseStepChild{{in: in, out: out, act: core.ActivationLinear, W: W, bias: bias}}
	ticksX := [][]float32{x}
	ticksT := [][]float32{tgt}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "mtlµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		ch := []denseStepChild{{in: in, out: out, act: core.ActivationLinear, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...)}}
		tx := [][]float32{append([]float32(nil), x...)}
		tt := [][]float32{append([]float32(nil), tgt...)}
		goO, err := runGoDenseTween(dt, 1e-2, ch, batch, tx, tt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseTWB1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		benchO, mtlDur, err := runMetalTWB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s mtl:%v", dt, err))
			continue
		}
		// Benchmark runs must leave the same result as an independent run.
		mtlO, _, err := runMetalTWB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s mtl2:%v", dt, err))
			continue
		}
		if err := compareTween(mtlO, benchO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s benchmark reset: %v", dt, err))
		}
		m := suites.CompareF32Spectrum(goO.WAfter[0], mtlO.WAfter[0])
		goUS, berr := benchGoUS(func() error {
			_, err := runGoDenseTween(dt, 1e-2, ch, batch, tx, tt)
			return err
		}, benchIters)
		if berr != nil {
			fails = append(fails, fmt.Sprintf("%s go-bench:%v", dt, berr))
			continue
		}
		mtlUS := perIterUS(mtlDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, mtlUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %8.1f%%\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(mtlUS), spd)
		if err := compareTween(goO, mtlO, false); err != nil {
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

func encodeTWB1Header(nChildren, nTicks, batch int, lr float64) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte("TWB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(nChildren))
	_ = binary.Write(&in, binary.LittleEndian, int32(nTicks))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return &in
}

func caseRMSNormTweenF32() error {
	return runRMSNormTween(core.DTypeFloat32, true)
}

func caseRMSNormTweenAllDTypes() error {
	return runTweenSpectrum("RMSNorm", buildRMSNormTween)
}

func buildRMSNormTween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ga, err := weights.MatrixF32(st.Children[0].(*rmsnorm.Layer).Gamma)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{ga}}

	in := encodeTWB1Header(1, 1, batch, lr)
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

func runRMSNormTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildRMSNormTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseLayerNormTweenF32() error {
	return runLayerNormTween(core.DTypeFloat32, true)
}

func caseLayerNormTweenAllDTypes() error {
	return runTweenSpectrum("LayerNorm", buildLayerNormTween)
}

func buildLayerNormTween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ll := st.Children[0].(*layernorm.Layer)
	gA, err := weights.MatrixF32(ll.Gamma)
	if err != nil {
		return tweenOut{}, nil, err
	}
	bA, err := weights.MatrixF32(ll.Beta)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{append(append([]float32(nil), gA...), bA...)}}

	in := encodeTWB1Header(1, 1, batch, lr)
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

func runLayerNormTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildLayerNormTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseSoftmaxTweenF32() error {
	return runSoftmaxTween(core.DTypeFloat32, true)
}

func caseSoftmaxTweenAllDTypes() error {
	return runTweenSpectrum("Softmax", buildSoftmaxTween)
}

func buildSoftmaxTween(dtype core.DType) (tweenOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	x := []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1}
	tgt := []float32{0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4}
	cfg := softmax.Config{Dim: dim, Kind: softmax.KindStandard, Temperature: 1}
	l, err := softmax.New(cfg)
	if err != nil {
		return tweenOut{}, nil, err
	}
	_ = l.SetDType(dtype)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{{}}}

	in := encodeTWB1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(5)) // Softmax
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, int32(0)) // Standard
	_ = binary.Write(in, binary.LittleEndian, 1.0)
	writeShapesTicks(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runSoftmaxTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildSoftmaxTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseSwiGLUTweenF32() error {
	return runSwiGLUTween(core.DTypeFloat32, true)
}

func caseSwiGLUTweenAllDTypes() error {
	return runTweenSpectrum("SwiGLU", buildSwiGLUTween)
}

func buildSwiGLUTween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, inDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, inDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	sl := st.Children[0].(*swiglu.Layer)
	gW, err := weights.MatrixF32(sl.Gate.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	uW, err := weights.MatrixF32(sl.Up.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	dW, err := weights.MatrixF32(sl.Down.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	wa := append(append(append([]float32(nil), gW...), uW...), dW...)
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runSwiGLUTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildSwiGLUTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseSequentialTweenF32() error {
	return runSeqResTween(true, core.DTypeFloat32, true)
}

func caseSequentialTweenAllDTypes() error {
	return runTweenSpectrum("Sequential", func(dt core.DType) (tweenOut, []byte, error) {
		return buildSeqResTween(true, dt)
	})
}

func caseResidualTweenF32() error {
	return runSeqResTween(false, core.DTypeFloat32, true)
}

func caseResidualTweenAllDTypes() error {
	return runTweenSpectrum("Residual", func(dt core.DType) (tweenOut, []byte, error) {
		return buildSeqResTween(false, dt)
	})
}

func buildSeqResTween(isSeq bool, dtype core.DType) (tweenOut, []byte, error) {

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
			return tweenOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	} else {
		cfg := residual.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := residual.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return tweenOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	}
	st, err := parallel.NewStack(op)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	var wa []float32
	if isSeq {
		sl := st.Children[0].(*sequential.Layer)
		for _, ch := range sl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return tweenOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	} else {
		rl := st.Children[0].(*residual.Layer)
		for _, ch := range rl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return tweenOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}

	kind := int32(16)
	if !isSeq {
		kind = 17
	}
	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runSeqResTween(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResTween(isSeq, dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseMambaTweenF32() error {
	return runMambaTween(core.DTypeFloat32, true)
}

func caseMambaTweenAllDTypes() error {
	return runTweenSpectrum("Mamba", buildMambaTween)
}

func buildMambaTween(dtype core.DType) (tweenOut, []byte, error) {

	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	const batch = 2
	lr := 1e-2
	// Same init family as Mamba NormalBP leaf spectrum (avoids pathological tiny-dtype packs).
	fc, err := makeMbLeafCase(cfg, batch, dtype, 0x53545042^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenOut{}, nil, err
	}
	packed := fc.W
	x := fc.x
	tgt := fc.target
	lr = fc.lr
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, packed)
	if err != nil {
		return tweenOut{}, nil, err
	}
	l, err := mamba.NewConfigured(cfg, dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return tweenOut{}, nil, err
	}
	disableMambaTrain(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*mamba.Layer)
	wa, err := packMambaWeightsF32(ml)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runMambaTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildMambaTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseEmbeddingTweenF32() error { return runEmbeddingTween(core.DTypeFloat32, true) }
func caseEmbeddingTweenAllDTypes() error {
	return runTweenSpectrum("Embedding", buildEmbeddingTween)
}

func buildEmbeddingTween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, emb)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*embedding.Layer).Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runEmbeddingTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildEmbeddingTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseMHATweenF32() error { return runMHATween(core.DTypeFloat32, true) }
func caseMHATweenAllDTypes() error {
	return runTweenSpectrum("MHA", buildMHATween)
}

func buildMHATween(dtype core.DType) (tweenOut, []byte, error) {

	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 16
	const batch, seq = 2, 4
	fc, err := makeMHANBPCase(cfg, batch, seq, dtype, 0x534d4841, 1e-2)
	if err != nil {
		return tweenOut{}, nil, err
	}
	c := fc.cfg // validated
	l, err := mha.NewConfigured(c, dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, c.DModel)
	copy(xin.Data, fc.x)
	target := core.NewTensor[float32](batch, seq, c.DModel)
	copy(target.Data, fc.target)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, fc.lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := exportMHAWeights(st.Children[0].(*mha.Layer))
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, fc.lr)
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

func runMHATween(dtype core.DType, exact bool) error {
	goO, payload, err := buildMHATween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseCNN1TweenF32() error { return runCNN1Tween(core.DTypeFloat32, true) }
func caseCNN1TweenAllDTypes() error {
	return runTweenSpectrum("CNN1", buildCNN1Tween)
}

func buildCNN1Tween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn1.Layer).Proj.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runCNN1Tween(dtype core.DType, exact bool) error {
	goO, payload, err := buildCNN1Tween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseConvT1TweenF32() error { return runConvT1Tween(core.DTypeFloat32, true) }
func caseConvT1TweenAllDTypes() error {
	return runTweenSpectrum("ConvT1", buildConvT1Tween)
}

func buildConvT1Tween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt1.Layer).Proj.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runConvT1Tween(dtype core.DType, exact bool) error {
	goO, payload, err := buildConvT1Tween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseRNNTweenF32() error { return runRNNTween(core.DTypeFloat32, true) }
func caseRNNTweenAllDTypes() error {
	return runTweenSpectrum("RNN", buildRNNTween)
}

func buildRNNTween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	disableRNN(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packRNNWeightsF32(st.Children[0].(*rnn.Layer))
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runRNNTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildRNNTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseLSTMTweenF32() error { return runLSTMTween(core.DTypeFloat32, true) }
func caseLSTMTweenAllDTypes() error {
	return runTweenSpectrum("LSTM", buildLSTMTween)
}

func buildLSTMTween(dtype core.DType) (tweenOut, []byte, error) {

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
		return tweenOut{}, nil, err
	}
	disableLSTMTrainGates(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packLSTMWeightsF32(st.Children[0].(*lstm.Layer))
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
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

func runLSTMTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildLSTMTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

// --- Dense Tween volume (TWV1) ---

func encodeDenseTWV1(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("TWV1"))
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

func runMetalTWV1(payload []byte, benchIters int) (meshbpOut, time.Duration, error) {
	// reuse meshbpOut shape: loss + Ws
	var z meshbpOut
	var bench time.Duration
	tool, err := findMetalFwd()
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
		return z, 0, fmt.Errorf("mtl twv1: %v: %s", err, stderr.String())
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
	if len(raw) < 8 || string(raw[:4]) != "TWVO" {
		return z, bench, fmt.Errorf("mtl tween vol bad magic")
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

func runGoDenseTweenVol(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) (meshbpOut, error) {
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
	// ModeTween on grid via Stack of placed ops is awkward; match Metal reverse-hop:
	// TrainStackMSE isn't volume. Replicate trainTweenLocal reverse via parallel place.
	// Use training-equivalent: wrap as placing already done — call soft local updates.
	loss, err := training.MSE(fwd.Output, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(fwd.Output, tgt)
	if err != nil {
		return z, err
	}
	soft := lr * 0.5
	curGy := gy
	for i := len(fwd.Steps) - 1; i >= 0; i-- {
		st := fwd.Steps[i]
		cell := g.At(st.Coord.Z, st.Coord.Y, st.Coord.X, st.Coord.L)
		if cell == nil || cell.Layer.IsDisabled {
			continue
		}
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return z, fmt.Errorf("expected dense")
		}
		proj := projectGapF32(curGy, st.Post.Shape...)
		_, dW, err := dense.Backward(dl, proj, st.Input, st.Pre)
		if err != nil {
			return z, err
		}
		if dW != nil && dW.Len() > 0 {
			if err := dense.ApplyGradSGD(dl, dW, soft); err != nil {
				return z, err
			}
		}
		next := core.NewTensor[float32](st.Input.Shape...)
		if curGy.Len() == next.Len() {
			copy(next.Data, curGy.Data)
		}
		curGy = next
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

func projectGapF32(src *core.Tensor[float32], dstShape ...int) *core.Tensor[float32] {
	dst := core.NewTensor[float32](dstShape...)
	if src == nil || dst.Len() == 0 {
		return dst
	}
	sn, dn := src.Len(), dst.Len()
	if sn == dn {
		copy(dst.Data, src.Data)
		return dst
	}
	if sn <= 0 {
		return dst
	}
	for i := 0; i < dn; i++ {
		lo := i * sn / dn
		hi := (i + 1) * sn / dn
		if hi <= lo {
			hi = lo + 1
		}
		if hi > sn {
			hi = sn
		}
		var sum float64
		for j := lo; j < hi; j++ {
			sum += float64(src.Data[j])
		}
		dst.Data[i] = float32(sum / float64(hi-lo))
	}
	return dst
}

func runDenseTweenVolCase(depth, rows, cols, lpc, batch, benchIters int) error {
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
	goO, err := runGoDenseTweenVol(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeDenseTWV1(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	_, mtlDur, err := runMetalTWV1(payload, benchIters)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	mtlO, _, err := runMetalTWV1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl2: %w", err)
	}
	if err := compareMeshBP(goO, mtlO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, _ = runGoDenseTweenVol(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
		return nil
	}, defaultGoBenchIters)
	if err != nil {
		return err
	}
	bi := benchIters
	if bi < 1 {
		bi = 1
	}
	mtlUS := float64(mtlDur.Nanoseconds()) / float64(bi) / 1000
	spd := suites.SpeedPctDiff(goUS, mtlUS)
	fmt.Printf("\n      Tween vol %dx%dx%dx%d loss=%.6g exact_bit  go=%s  mtl=%s  spd%%Δ=%+.1f%%",
		depth, rows, cols, lpc, goO.loss, FormatUS(goUS), FormatUS(mtlUS), spd)
	return nil
}

func caseDenseTweenVol111F32() error { return runDenseTweenVolCase(1, 1, 1, 1, 1, defaultGoBenchIters) }
func caseDenseTweenVol211F32() error { return runDenseTweenVolCase(2, 1, 1, 1, 1, defaultGoBenchIters) }
func caseDenseTweenVol222F32() error { return runDenseTweenVolCase(2, 2, 2, 1, 1, defaultGoBenchIters) }

// --- CASCADE Meta/GDN/Stack leaf (auto) ---

func caseMetacognitionTweenF32() error {
	return runMetacognitionTween(core.DTypeFloat32, true)
}

func caseMetacognitionTweenAllDTypes() error {
	return runTweenSpectrum("Metacognition", buildMetacognitionTween)
}

func buildMetacognitionTween(dtype core.DType) (tweenOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d435450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMetacognitionTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

func caseGDNTweenF32() error {
	return runGDNTween(false)
}

func buildGDNTween() (tweenOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x47444e50)
	if err != nil {
		return tweenOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runGDNTween(exact bool) error {
	goO, payload, err := buildGDNTween()
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	for i := range goO.WAfter {
		m := suites.CompareF32Spectrum(goO.WAfter[i], mtlO.WAfter[i])
		if m.Band < suites.DetStrict {
			return fmt.Errorf("GDN W[%d] below strict: %s", i, suites.FormatSpectrum(m))
		}
	}
	return compareTween(goO, mtlO, exact)
}

func caseNestedStackTweenF32() error {
	return runNestedStackTween(core.DTypeFloat32, true)
}

func caseNestedStackTweenAllDTypes() error {
	return runTweenSpectrum("NestedStack", buildNestedStackTween)
}

func buildNestedStackTween(dtype core.DType) (tweenOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53544b50^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return tweenOut{}, nil, err
	}
	writeShapesTicks(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runNestedStackTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackTween(dtype)
	if err != nil {
		return err
	}
	mtlO, _, err := runMetalTWB1(payload, 0)
	if err != nil {
		return fmt.Errorf("mtl: %w", err)
	}
	return compareTween(goO, mtlO, exact)
}

// --- CASCADE KMeans leaf (auto) ---

func buildCNN2Tween(dtype core.DType) (tweenOut, []byte, error) {

	cfg := preservingCNN2()
	const batch = 2
	lr := 1e-2
	nw := cfg.Filters * cfg.PatchDim()
	W := randInit(1, nw, 0x53434e31)
	xLen := batch * cfg.InChannels * cfg.Height * cfg.Width
	x := randInit(1, xLen, 0x53434e32)
	outHeight := cnn2.OutputSpatial(cfg.Height, cfg.Kernel, cfg.Stride, cfg.Padding)
	outWidth := cnn2.OutputSpatial(cfg.Width, cfg.Kernel, cfg.Stride, cfg.Padding)

	tgt := make([]float32, batch*cfg.Filters*outHeight*outWidth)
	for i := range tgt {
		tgt[i] = float32(0.04 * float64(i%6+1))
	}
	l, err := cnn2.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn2.Layer).Proj.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(9))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InChannels))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Filters))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Height))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Width))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Kernel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Stride))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Padding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.Height), int32(cfg.Width)},
		[]int32{batch, int32(cfg.Filters), int32(outHeight), int32(outWidth)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func buildCNN3Tween(dtype core.DType) (tweenOut, []byte, error) {

	cfg := preservingCNN3()
	const batch = 2
	lr := 1e-2
	nw := cfg.Filters * cfg.PatchDim()
	W := randInit(1, nw, 0x53434e31)
	xLen := batch * cfg.InChannels * cfg.Depth * cfg.Height * cfg.Width
	x := randInit(1, xLen, 0x53434e32)
	outDepth := cnn3.OutputSpatial(cfg.Depth, cfg.Kernel, cfg.Stride, cfg.Padding)
	outHeight := cnn3.OutputSpatial(cfg.Height, cfg.Kernel, cfg.Stride, cfg.Padding)
	outWidth := cnn3.OutputSpatial(cfg.Width, cfg.Kernel, cfg.Stride, cfg.Padding)

	tgt := make([]float32, batch*cfg.Filters*outDepth*outHeight*outWidth)
	for i := range tgt {
		tgt[i] = float32(0.04 * float64(i%6+1))
	}
	l, err := cnn3.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outDepth, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn3.Layer).Proj.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(10))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InChannels))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Filters))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Depth))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Height))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Width))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Kernel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Stride))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Padding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.Depth), int32(cfg.Height), int32(cfg.Width)},
		[]int32{batch, int32(cfg.Filters), int32(outDepth), int32(outHeight), int32(outWidth)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func buildConvT2Tween(dtype core.DType) (tweenOut, []byte, error) {

	cfg := preservingConvT2()
	const batch = 2
	lr := 1e-2
	nw := cfg.Filters * cfg.PatchDim()
	W := randInit(1, nw, 0x53543131)
	xLen := batch * cfg.InChannels * cfg.Height * cfg.Width
	x := randInit(1, xLen, 0x53543132)
	outHeight := convt2.OutputSpatial(cfg.Height, cfg.Kernel, cfg.Stride, cfg.Padding, cfg.OutputPadding)
	outWidth := convt2.OutputSpatial(cfg.Width, cfg.Kernel, cfg.Stride, cfg.Padding, cfg.OutputPadding)

	tgt := make([]float32, batch*cfg.Filters*outHeight*outWidth)
	for i := range tgt {
		tgt[i] = float32(0.04 * float64(i%6+1))
	}
	l, err := convt2.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt2.Layer).Proj.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(12))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InChannels))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Filters))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Height))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Width))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Kernel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Stride))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Padding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.OutputPadding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.Height), int32(cfg.Width)},
		[]int32{batch, int32(cfg.Filters), int32(outHeight), int32(outWidth)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func buildConvT3Tween(dtype core.DType) (tweenOut, []byte, error) {

	cfg := preservingConvT3()
	const batch = 2
	lr := 1e-2
	nw := cfg.Filters * cfg.PatchDim()
	W := randInit(1, nw, 0x53543131)
	xLen := batch * cfg.InChannels * cfg.Depth * cfg.Height * cfg.Width
	x := randInit(1, xLen, 0x53543132)
	outDepth := convt3.OutputSpatial(cfg.Depth, cfg.Kernel, cfg.Stride, cfg.Padding, cfg.OutputPadding)
	outHeight := convt3.OutputSpatial(cfg.Height, cfg.Kernel, cfg.Stride, cfg.Padding, cfg.OutputPadding)
	outWidth := convt3.OutputSpatial(cfg.Width, cfg.Kernel, cfg.Stride, cfg.Padding, cfg.OutputPadding)

	tgt := make([]float32, batch*cfg.Filters*outDepth*outHeight*outWidth)
	for i := range tgt {
		tgt[i] = float32(0.04 * float64(i%6+1))
	}
	l, err := convt3.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return tweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outDepth, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTween, lr)
	if err != nil {
		return tweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt3.Layer).Proj.Weights)
	if err != nil {
		return tweenOut{}, nil, err
	}
	goO := tweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(13))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InChannels))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Filters))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Depth))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Height))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Width))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Kernel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Stride))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Padding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.OutputPadding))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicks(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.Depth), int32(cfg.Height), int32(cfg.Width)},
		[]int32{batch, int32(cfg.Filters), int32(outDepth), int32(outHeight), int32(outWidth)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

// Exercise real double-buffer mesh ticks, temporal gradient accumulation and
// native storage updates. These are distinct from ModeMeshBP volume aliases.

func caseDenseTweenVolAllDTypes() error {
	return runTweenVolumeSpectrum(func(dtype core.DType) error {
		cells := []volCell{
			{enabled: true, in: 4, out: 4, act: core.ActivationLinear, dtype: dtype, W: randInit(4, 4, 0x54574e31)},
			{enabled: true, in: 4, out: 4, act: core.ActivationLinear, dtype: dtype, W: randInit(4, 4, 0x54574e32)},
		}
		x := randInit(1, 4, 0x54574e33)
		target := []float32{0.1, 0.2, -0.3, 0.4}
		goO, err := runGoDenseTweenVol(2, 1, 1, 1, 1, cells, x, target, 0.01)
		if err != nil {
			return err
		}
		payload, err := encodeDenseTWV1(2, 1, 1, 1, 1, cells, x, target, 0.01)
		if err != nil {
			return err
		}
		metalO, _, err := runMetalTWV1(payload, 0)
		if err != nil {
			return err
		}
		if err := compareMeshBP(goO, metalO, true); err != nil {
			return err
		}
		benchO, _, err := runMetalTWV1(payload, 2)
		if err != nil {
			return err
		}
		return compareMeshBP(metalO, benchO, true)
	})
}

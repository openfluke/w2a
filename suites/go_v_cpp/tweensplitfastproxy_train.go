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

// CasesTweenSplitFastProxyTrain — ModeTweenSplitFastProxy (head GradWOnly; hidden 1/(N-1) P(Wᵀgy)) Go↔C++ for NormalBP-signed kinds.
func CasesTweenSplitFastProxyTrain() []Case {
	return []Case{
		{Name: "Dense TweenSplitFastProxy multi-hop Float32 — exact bits (W/loss)", Run: caseDenseTweenSplitFastProxyMultiHopF32},
		{Name: "Dense TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseTweenSplitFastProxy1HopAllDTypes},
		{Name: "RMSNorm TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseRMSNormTweenSplitFastProxyF32},
		{Name: "RMSNorm TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormTweenSplitFastProxyAllDTypes},
		{Name: "LayerNorm TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseLayerNormTweenSplitFastProxyF32},
		{Name: "LayerNorm TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormTweenSplitFastProxyAllDTypes},
		{Name: "Softmax TweenSplitFastProxy 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxTweenSplitFastProxyF32},
		{Name: "Softmax TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxTweenSplitFastProxyAllDTypes},
		{Name: "SwiGLU TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseSwiGLUTweenSplitFastProxyF32},
		{Name: "SwiGLU TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUTweenSplitFastProxyAllDTypes},
		{Name: "Sequential TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseSequentialTweenSplitFastProxyF32},
		{Name: "Sequential TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseSequentialTweenSplitFastProxyAllDTypes},
		{Name: "Residual TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseResidualTweenSplitFastProxyF32},
		{Name: "Residual TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseResidualTweenSplitFastProxyAllDTypes},
		{Name: "Mamba TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseMambaTweenSplitFastProxyF32},
		{Name: "Mamba TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseMambaTweenSplitFastProxyAllDTypes},
		{Name: "Metacognition TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseMetacognitionTweenSplitFastProxyF32},
		{Name: "Metacognition TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionTweenSplitFastProxyAllDTypes},
		{Name: "KMeans TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseKMeansTweenSplitFastProxyF32},
		{Name: "KMeans TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseKMeansTweenSplitFastProxyAllDTypes},
		{Name: "GDN TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseGDNTweenSplitFastProxyF32},
		{Name: "NestedStack TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseNestedStackTweenSplitFastProxyF32},
		{Name: "NestedStack TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackTweenSplitFastProxyAllDTypes},
		{Name: "Embedding TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseEmbeddingTweenSplitFastProxyF32},
		{Name: "Embedding TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingTweenSplitFastProxyAllDTypes},
		{Name: "MHA TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseMHATweenSplitFastProxyF32},
		{Name: "MHA TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseMHATweenSplitFastProxyAllDTypes},
		{Name: "CNN1 TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseCNN1TweenSplitFastProxyF32},
		{Name: "CNN1 TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseCNN1TweenSplitFastProxyAllDTypes},
		{Name: "ConvT1 TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseConvT1TweenSplitFastProxyF32},
		{Name: "ConvT1 TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseConvT1TweenSplitFastProxyAllDTypes},
		{Name: "RNN TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseRNNTweenSplitFastProxyF32},
		{Name: "RNN TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseRNNTweenSplitFastProxyAllDTypes},
		{Name: "LSTM TweenSplitFastProxy 1-hop Float32 — exact bits", Run: caseLSTMTweenSplitFastProxyF32},
		{Name: "LSTM TweenSplitFastProxy 1-hop × all 34 dtypes — spectrum", Run: caseLSTMTweenSplitFastProxyAllDTypes},
		{Name: "Dense TweenSplitFastProxy volume 1×1×1×1 Float32 — exact bits", Run: caseDenseTweenSplitFastProxyVol111F32},
		{Name: "Dense TweenSplitFastProxy volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseTweenSplitFastProxyVol211F32},
		{Name: "Dense TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseDenseTweenSplitFastProxyVol222F32},
		{Name: "RMSNorm TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormTweenSplitFastProxyVol222F32},
		{Name: "LayerNorm TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormTweenSplitFastProxyVol222F32},
		{Name: "Softmax TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxTweenSplitFastProxyVol222F32},
		{Name: "SwiGLU TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUTweenSplitFastProxyVol222F32},
		{Name: "Sequential TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseSeqTweenSplitFastProxyVol222F32},
		{Name: "Residual TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseReTweenSplitFastProxyVol222F32},
		{Name: "Mamba TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseMbTweenSplitFastProxyVol222F32},
		{Name: "Metacognition TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseMcTweenSplitFastProxyVol222F32},
		{Name: "KMeans TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseKmTweenSplitFastProxyVol222F32},
		{Name: "GDN TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseGdTweenSplitFastProxyVol222F32},
		{Name: "NestedStack TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseSkTweenSplitFastProxyVol222F32},
		{Name: "Embedding TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingTweenSplitFastProxyVol222F32},
		{Name: "MHA TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseMHATweenSplitFastProxyVol222F32},
		{Name: "CNN1 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenSplitFastProxyVol222F32(1) }},
		{Name: "CNN2 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenSplitFastProxyVol222F32(2) }},
		{Name: "CNN3 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenSplitFastProxyVol222F32(3) }},
		{Name: "ConvT1 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenSplitFastProxyVol222F32(1) }},
		{Name: "ConvT2 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenSplitFastProxyVol222F32(2) }},
		{Name: "ConvT3 TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenSplitFastProxyVol222F32(3) }},
		{Name: "RNN TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseRNNTweenSplitFastProxyVol222F32},
		{Name: "LSTM TweenSplitFastProxy volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMTweenSplitFastProxyVol222F32},
	}
}

func RunTweenSplitFastProxyTrain() error {
	var fails []string
	cs := CasesTweenSplitFastProxyTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [tweensplitfastproxy %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_tweensplitfastproxy_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_tweensplitfastproxy_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp tweensplitfastproxy_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type tweenSplitFastProxyOut struct {
	loss   float64
	WAfter [][]float32
}

func runCppFPB1(payload []byte, benchIters int) (tweenSplitFastProxyOut, time.Duration, error) {
	var z tweenSplitFastProxyOut
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
	if len(raw) < 8 || string(raw[:4]) != "FPBO" {
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

func compareTweenSplitFastProxy(goO, cppO tweenSplitFastProxyOut, requireExact bool) error {
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

// runTweenSplitFastProxySpectrum prints Dense-style dtype×speed table then requires exact_bit on each.
func runTweenSplitFastProxySpectrum(label string, build func(core.DType) (tweenSplitFastProxyOut, []byte, error)) error {
	const benchIters = 20
	fmt.Printf("\n      [%s Tween ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		t0 := time.Now()
		goO, payload, err := build(dt)
		goDur := time.Since(t0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		_, cppDur, err := runCppFPB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp-bench:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10.2f %10s %8s  %v\n", dt, suites.DetBroken, "—", float64(goDur.Nanoseconds())/1000, "—", "—", err)
			continue
		}
		cppO, _, err := runCppFPB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10.2f %10s %8s  %v\n", dt, suites.DetBroken, "—", float64(goDur.Nanoseconds())/1000, "—", "—", err)
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
		goUS := float64(goDur.Nanoseconds()) / 1000
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %8.1f%%\n",
			dt, m.Band, m.ExactPct, goUS, cppUS, spd)
		if err := compareTweenSplitFastProxy(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- Dense Tween ---

func encodeDenseFPB1(dtype core.DType, lr float64, children []denseStepChild, batch int, xShape, tShape []int32, ticksX, ticksT [][]float32) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("FPB1"))
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

func runGoDenseTweenSplitFastProxy(dtype core.DType, lr float64, children []denseStepChild, batch int, ticksX, ticksT [][]float32) (tweenSplitFastProxyOut, error) {
	var z tweenSplitFastProxyOut
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
		loss, err = parallel.TrainStackMSE(st, xin, tgt, parallel.ModeTweenSplitFastProxy, lr)
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

func caseDenseTweenSplitFastProxyMultiHopF32() error {
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
	goO, err := runGoDenseTweenSplitFastProxy(core.DTypeFloat32, lr, children, batch, ticksX, ticksT)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseFPB1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, true)
}

func caseDenseTweenSplitFastProxy1HopAllDTypes() error {
	const in, out, batch, benchIters = 6, 4, 2, 20
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
		t0 := time.Now()
		goO, err := runGoDenseTweenSplitFastProxy(dt, 1e-2, ch, batch, tx, tt)
		goDur := time.Since(t0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseFPB1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		cppO, cppDur, err := runCppFPB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			continue
		}
		// fresh cpp once for compare (bench mutated)
		cppO, _, err = runCppFPB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp2:%v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS := float64(goDur.Nanoseconds()) / 1000
		cppUS := float64(cppDur.Nanoseconds()) / float64(benchIters) / 1000
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10.2f %10.2f %8.1f%%\n",
			dt, m.Band, m.ExactPct, goUS, cppUS, spd)
		if err := compareTweenSplitFastProxy(goO, cppO, false); err != nil {
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

func encodeTWB1HeaderFastProxy(nChildren, nTicks, batch int, lr float64) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte("FPB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(nChildren))
	_ = binary.Write(&in, binary.LittleEndian, int32(nTicks))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return &in
}

func caseRMSNormTweenSplitFastProxyF32() error {
	return runRMSNormTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseRMSNormTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("RMSNorm", buildRMSNormTweenSplitFastProxy)
}

func buildRMSNormTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ga, err := weights.MatrixF32(st.Children[0].(*rmsnorm.Layer).Gamma)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{ga}}

	in := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runRMSNormTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildRMSNormTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseLayerNormTweenSplitFastProxyF32() error {
	return runLayerNormTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseLayerNormTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("LayerNorm", buildLayerNormTweenSplitFastProxy)
}

func buildLayerNormTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ll := st.Children[0].(*layernorm.Layer)
	gA, err := weights.MatrixF32(ll.Gamma)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	bA, err := weights.MatrixF32(ll.Beta)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{append(append([]float32(nil), gA...), bA...)}}

	in := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runLayerNormTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildLayerNormTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseSoftmaxTweenSplitFastProxyF32() error {
	return runSoftmaxTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseSoftmaxTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("Softmax", buildSoftmaxTweenSplitFastProxy)
}

func buildSoftmaxTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	x := []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1}
	tgt := []float32{0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4}
	cfg := softmax.Config{Dim: dim, Kind: softmax.KindStandard, Temperature: 1}
	l, err := softmax.New(cfg)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	_ = l.SetDType(dtype)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{{}}}

	in := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(5)) // Softmax
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, int32(0)) // Standard
	_ = binary.Write(in, binary.LittleEndian, 1.0)
	writeShapesTicks(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runSoftmaxTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildSoftmaxTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseSwiGLUTweenSplitFastProxyF32() error {
	return runSwiGLUTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseSwiGLUTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("SwiGLU", buildSwiGLUTweenSplitFastProxy)
}

func buildSwiGLUTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, inDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, inDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	sl := st.Children[0].(*swiglu.Layer)
	gW, err := weights.MatrixF32(sl.Gate.Weights)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	uW, err := weights.MatrixF32(sl.Up.Weights)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	dW, err := weights.MatrixF32(sl.Down.Weights)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	wa := append(append(append([]float32(nil), gW...), uW...), dW...)
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runSwiGLUTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildSwiGLUTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseSequentialTweenSplitFastProxyF32() error {
	return runSeqResTweenSplitFastProxy(true, core.DTypeFloat32, true)
}

func caseSequentialTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("Sequential", func(dt core.DType) (tweenSplitFastProxyOut, []byte, error) {
		return buildSeqResTweenSplitFastProxy(true, dt)
	})
}

func caseResidualTweenSplitFastProxyF32() error {
	return runSeqResTweenSplitFastProxy(false, core.DTypeFloat32, true)
}

func caseResidualTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("Residual", func(dt core.DType) (tweenSplitFastProxyOut, []byte, error) {
		return buildSeqResTweenSplitFastProxy(false, dt)
	})
}

func buildSeqResTweenSplitFastProxy(isSeq bool, dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
			return tweenSplitFastProxyOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	} else {
		cfg := residual.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := residual.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return tweenSplitFastProxyOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	}
	st, err := parallel.NewStack(op)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	var wa []float32
	if isSeq {
		sl := st.Children[0].(*sequential.Layer)
		for _, ch := range sl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return tweenSplitFastProxyOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	} else {
		rl := st.Children[0].(*residual.Layer)
		for _, ch := range rl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return tweenSplitFastProxyOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}

	kind := int32(16)
	if !isSeq {
		kind = 17
	}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runSeqResTweenSplitFastProxy(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResTweenSplitFastProxy(isSeq, dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseMambaTweenSplitFastProxyF32() error {
	return runMambaTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseMambaTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("Mamba", buildMambaTweenSplitFastProxy)
}

func buildMambaTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	const batch = 2
	lr := 1e-2
	// Same init family as Mamba NormalBP leaf spectrum (avoids pathological tiny-dtype packs).
	fc, err := makeMbLeafCase(cfg, batch, dtype, 0x53545042^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	packed := fc.W
	x := fc.x
	tgt := fc.target
	lr = fc.lr
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, packed)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	l, err := mamba.NewConfigured(cfg, dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	disableMambaTrain(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*mamba.Layer)
	wa, err := packMambaWeightsF32(ml)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runMambaTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildMambaTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseEmbeddingTweenSplitFastProxyF32() error { return runEmbeddingTweenSplitFastProxy(core.DTypeFloat32, true) }
func caseEmbeddingTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("Embedding", buildEmbeddingTweenSplitFastProxy)
}

func buildEmbeddingTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, emb)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*embedding.Layer).Weights)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runEmbeddingTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildEmbeddingTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseMHATweenSplitFastProxyF32() error { return runMHATweenSplitFastProxy(core.DTypeFloat32, true) }
func caseMHATweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("MHA", buildMHATweenSplitFastProxy)
}

func buildMHATweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 16
	const batch, seq = 2, 4
	fc, err := makeMHANBPCase(cfg, batch, seq, dtype, 0x534d4841, 1e-2)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	c := fc.cfg // validated
	l, err := mha.NewConfigured(c, dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, c.DModel)
	copy(xin.Data, fc.x)
	target := core.NewTensor[float32](batch, seq, c.DModel)
	copy(target.Data, fc.target)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, fc.lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := exportMHAWeights(st.Children[0].(*mha.Layer))
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, fc.lr)
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

func runMHATweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildMHATweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseCNN1TweenSplitFastProxyF32() error { return runCNN1TweenSplitFastProxy(core.DTypeFloat32, true) }
func caseCNN1TweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("CNN1", buildCNN1TweenSplitFastProxy)
}

func buildCNN1TweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn1.Layer).Proj.Weights)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runCNN1TweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildCNN1TweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseConvT1TweenSplitFastProxyF32() error { return runConvT1TweenSplitFastProxy(core.DTypeFloat32, true) }
func caseConvT1TweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("ConvT1", buildConvT1TweenSplitFastProxy)
}

func buildConvT1TweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt1.Layer).Proj.Weights)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runConvT1TweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildConvT1TweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseRNNTweenSplitFastProxyF32() error { return runRNNTweenSplitFastProxy(core.DTypeFloat32, true) }
func caseRNNTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("RNN", buildRNNTweenSplitFastProxy)
}

func buildRNNTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	disableRNN(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packRNNWeightsF32(st.Children[0].(*rnn.Layer))
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runRNNTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildRNNTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseLSTMTweenSplitFastProxyF32() error { return runLSTMTweenSplitFastProxy(core.DTypeFloat32, true) }
func caseLSTMTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("LSTM", buildLSTMTweenSplitFastProxy)
}

func buildLSTMTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {

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
		return tweenSplitFastProxyOut{}, nil, err
	}
	disableLSTMTrainGates(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packLSTMWeightsF32(st.Children[0].(*lstm.Layer))
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
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

func runLSTMTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildLSTMTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

// --- Dense TweenSplitFastProxy volume (TWV1) ---

func encodeDenseFPV1(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("FPV1"))
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

func runCppFPV1(payload []byte, benchIters int) (meshbpOut, time.Duration, error) {
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
	if len(raw) < 8 || string(raw[:4]) != "FPVO" {
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

func runGoDenseTweenSplitFastProxyVol(depth, rows, cols, lpc, batch int, cells []volCell, x, target []float32, lr float64) (meshbpOut, error) {
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
	loss, err := applyModeTweenSplitFastProxyVolume(g, fwd, tgt, lr)
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


func runDenseTweenSplitFastProxyVolCase(depth, rows, cols, lpc, batch, benchIters int) error {
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
	goO, err := runGoDenseTweenSplitFastProxyVol(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeDenseFPV1(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	_, cppDur, err := runCppFPV1(payload, benchIters)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := runCppFPV1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp2: %w", err)
	}
	if err := compareMeshBP(goO, cppO, true); err != nil {
		return err
	}
	t0 := time.Now()
	_, _ = runGoDenseTweenSplitFastProxyVol(depth, rows, cols, lpc, batch, cells, x, target, 1e-2)
	goUS := float64(time.Since(t0).Nanoseconds()) / 1000
	bi := benchIters
	if bi < 1 {
		bi = 1
	}
	cppUS := float64(cppDur.Nanoseconds()) / float64(bi) / 1000
	spd := suites.SpeedPctDiff(goUS, cppUS)
	fmt.Printf("\n      Tween vol %dx%dx%dx%d loss=%.6g exact_bit  go=%.2fµs  cpp=%.2fµs  spd%%Δ=%+.1f%%",
		depth, rows, cols, lpc, goO.loss, goUS, cppUS, spd)
	return nil
}

func caseDenseTweenSplitFastProxyVol111F32() error { return runDenseTweenSplitFastProxyVolCase(1, 1, 1, 1, 1, 40) }
func caseDenseTweenSplitFastProxyVol211F32() error { return runDenseTweenSplitFastProxyVolCase(2, 1, 1, 1, 1, 40) }
func caseDenseTweenSplitFastProxyVol222F32() error { return runDenseTweenSplitFastProxyVolCase(2, 2, 2, 1, 1, 20) }

// --- CASCADE Meta/GDN/Stack leaf (auto) ---

func caseMetacognitionTweenSplitFastProxyF32() error {
	return runMetacognitionTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseMetacognitionTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("Metacognition", buildMetacognitionTweenSplitFastProxy)
}

func buildMetacognitionTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d435450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMetacognitionTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseGDNTweenSplitFastProxyF32() error {
	return runGDNTweenSplitFastProxy(true)
}

func buildGDNTweenSplitFastProxy() (tweenSplitFastProxyOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x47444e50)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runGDNTweenSplitFastProxy(exact bool) error {
	goO, payload, err := buildGDNTweenSplitFastProxy()
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

func caseNestedStackTweenSplitFastProxyF32() error {
	return runNestedStackTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseNestedStackTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("NestedStack", buildNestedStackTweenSplitFastProxy)
}

func buildNestedStackTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53544b50^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	writeShapesTicks(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runNestedStackTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}

// --- CASCADE KMeans leaf (auto) ---

func caseKMeansTweenSplitFastProxyF32() error {
	return runKMeansTweenSplitFastProxy(core.DTypeFloat32, true)
}

func caseKMeansTweenSplitFastProxyAllDTypes() error {
	return runTweenSplitFastProxySpectrum("KMeans", buildKMeansTweenSplitFastProxy)
}

func buildKMeansTweenSplitFastProxy(dtype core.DType) (tweenSplitFastProxyOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeKmModeLeaf(dtype, 0x4b4d5450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	st, err := newKmStack(cfg, dtype, packed)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	outDim := cfg.OutDim()
	xin := core.NewTensor[float32](batch, seq, cfg.FeatureDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, outDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeTweenSplitFastProxy, lr)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, fmt.Errorf("go: %w", err)
	}
	kl := st.Children[0].(*kmeans.Layer)
	wa, err := packKMeansWeightsF32(kl)
	if err != nil {
		return tweenSplitFastProxyOut{}, nil, err
	}
	goO := tweenSplitFastProxyOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeTWB1HeaderFastProxy(1, 1, batch, lr)
	writeKmChild(buf, cfg, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.FeatureDim)},
		[]int32{int32(batch), int32(seq), int32(outDim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runKMeansTweenSplitFastProxy(dtype core.DType, exact bool) error {
	goO, payload, err := buildKMeansTweenSplitFastProxy(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppFPB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareTweenSplitFastProxy(goO, cppO, exact)
}


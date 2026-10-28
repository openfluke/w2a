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
	"github.com/openfluke/welvet/weights"
)

// CasesStepTweenTrain — Line StepTween (FamilyTween on pop @ lr*0.5) Go↔CUDA for NormalBP-signed kinds.
func casesStepTweenBase() []Case {
	cs := []Case{
		{Name: "Dense StepTween multi-hop Float32 — exact tensors; CUDA loss band (W/loss)", Run: caseDenseStepTweenMultiHopF32},
		{Name: "Dense StepTween 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseStepTween1HopAllDTypes},
		{Name: "RMSNorm StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseRMSNormStepTweenF32},
		{Name: "RMSNorm StepTween 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormStepTweenAllDTypes},
		{Name: "LayerNorm StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseLayerNormStepTweenF32},
		{Name: "LayerNorm StepTween 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormStepTweenAllDTypes},
		{Name: "Softmax StepTween 1-hop Float32 — exact tensors; CUDA loss band (loss)", Run: caseSoftmaxStepTweenF32},
		{Name: "Softmax StepTween 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxStepTweenAllDTypes},
		{Name: "SwiGLU StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseSwiGLUStepTweenF32},
		{Name: "SwiGLU StepTween 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUStepTweenAllDTypes},
		{Name: "Sequential StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseSequentialStepTweenF32},
		{Name: "Sequential StepTween 1-hop × all 34 dtypes — spectrum", Run: caseSequentialStepTweenAllDTypes},
		{Name: "Residual StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseResidualStepTweenF32},
		{Name: "Residual StepTween 1-hop × all 34 dtypes — spectrum", Run: caseResidualStepTweenAllDTypes},
		{Name: "Mamba StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseMambaStepTweenF32},
		{Name: "Mamba StepTween 1-hop × all 34 dtypes — spectrum", Run: caseMambaStepTweenAllDTypes},
		{Name: "Embedding StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseEmbeddingStepTweenF32},
		{Name: "Embedding StepTween 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingStepTweenAllDTypes},
		{Name: "MHA StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseMHAStepTweenF32},
		{Name: "MHA StepTween 1-hop × all 34 dtypes — spectrum", Run: caseMHAStepTweenAllDTypes},
		{Name: "CNN1 StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseCNN1StepTweenF32},
		{Name: "CNN1 StepTween 1-hop × all 34 dtypes — spectrum", Run: caseCNN1StepTweenAllDTypes},
		{Name: "ConvT1 StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseConvT1StepTweenF32},
		{Name: "ConvT1 StepTween 1-hop × all 34 dtypes — spectrum", Run: caseConvT1StepTweenAllDTypes},
		{Name: "RNN StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseRNNStepTweenF32},
		{Name: "RNN StepTween 1-hop × all 34 dtypes — spectrum", Run: caseRNNStepTweenAllDTypes},
		{Name: "LSTM StepTween 1-hop Float32 — exact tensors; CUDA loss band", Run: caseLSTMStepTweenF32},
		{Name: "LSTM StepTween 1-hop × all 34 dtypes — spectrum", Run: caseLSTMStepTweenAllDTypes},
		{Name: "Metacognition StepTween Float32", Run: caseMetacognitionStepTweenF32},
		{Name: "Metacognition StepTween × all 34 dtypes", Run: caseMetacognitionStepTweenAllDTypes},
		{Name: "GDN StepTween Float32 — ≥strict weights; CUDA loss band", Run: caseGDNStepTweenF32},
		{Name: "NestedStack StepTween Float32", Run: caseNestedStackStepTweenF32},
		{Name: "NestedStack StepTween × all 34 dtypes", Run: caseNestedStackStepTweenAllDTypes},
		{Name: "Metacognition StepTween volume 222 Float32", Run: caseMcTweenVol222F32},
		{Name: "GDN StepTween volume 222 Float32", Run: caseGdTweenVol222F32},
		{Name: "NestedStack StepTween volume 222 Float32", Run: caseSkTweenVol222F32},
		{Name: "CNN2 StepTween rank 4 × all 34 dtypes", Run: func() error { return runStepTweenSpectrum("CNN2", buildCNN2StepTween) }},
		{Name: "CNN3 StepTween rank 5 × all 34 dtypes", Run: func() error { return runStepTweenSpectrum("CNN3", buildCNN3StepTween) }},
		{Name: "ConvT2 StepTween rank 4 × all 34 dtypes", Run: func() error { return runStepTweenSpectrum("ConvT2", buildConvT2StepTween) }},
		{Name: "ConvT3 StepTween rank 5 × all 34 dtypes", Run: func() error { return runStepTweenSpectrum("ConvT3", buildConvT3StepTween) }},
		// 3D = MeshTween volume credit (= ModeTween volume wires)
		{Name: "Dense MeshTween volume 1×1×1×1 Float32 — exact tensors; CUDA loss band", Run: caseDenseTweenVol111F32},
		{Name: "Dense MeshTween volume 2×1×1×1 chain Float32 — exact tensors; CUDA loss band", Run: caseDenseTweenVol211F32},
		{Name: "Dense MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseDenseTweenVol222F32},
		{Name: "RMSNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseRMSNormTweenVol222F32},
		{Name: "LayerNorm MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseLayerNormTweenVol222F32},
		{Name: "Softmax MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseSoftmaxTweenVol222F32},
		{Name: "SwiGLU MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseSwiGLUTweenVol222F32},
		{Name: "Sequential MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseSeqTweenVol222F32},
		{Name: "Residual MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseReTweenVol222F32},
		{Name: "Mamba MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseMbTweenVol222F32},
		{Name: "Embedding MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseEmbeddingTweenVol222F32},
		{Name: "MHA MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseMHATweenVol222F32},
		{Name: "CNN1 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseCNNTweenVol222F32(1) }},
		{Name: "CNN2 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseCNNTweenVol222F32(2) }},
		{Name: "CNN3 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseCNNTweenVol222F32(3) }},
		{Name: "ConvT1 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseConvTTweenVol222F32(1) }},
		{Name: "ConvT2 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseConvTTweenVol222F32(2) }},
		{Name: "ConvT3 MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: func() error { return caseConvTTweenVol222F32(3) }},
		{Name: "RNN MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseRNNTweenVol222F32},
		{Name: "LSTM MeshTween volume 2×2×2×1 Float32 — exact tensors; CUDA loss band", Run: caseLSTMTweenVol222F32},
	}
	return cs
}

func RunStepTweenTrain() error {
	var fails []string
	cs := CasesStepTweenTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [steptween %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cuda_steptween_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cuda_steptween_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cuda steptween_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type steptweenOut struct {
	loss   float64
	WAfter [][]float32
}

func disableDenseStepTween(l *dense.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
}

func runCUDASTW1(payload []byte, benchIters int) (steptweenOut, time.Duration, error) {
	var z steptweenOut
	var bench time.Duration
	tool, err := findCUDAFwd()
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
		return z, 0, fmt.Errorf("cuda stb1: %v: %s", err, stderr.String())
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
	if len(raw) < 8 || string(raw[:4]) != "STWO" {
		return z, bench, fmt.Errorf("cuda steptween bad magic")
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

func compareStepTween(goO, cudaO steptweenOut, requireExact bool) error {
	if len(goO.WAfter) != len(cudaO.WAfter) {
		return fmt.Errorf("n_children go=%d cuda=%d", len(goO.WAfter), len(cudaO.WAfter))
	}
	for i := range goO.WAfter {
		m := suites.CompareF32Spectrum(goO.WAfter[i], cudaO.WAfter[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
		if !requireExact && (m.Band == suites.DetBroken || m.Band == suites.DetBallpark) {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	if !cudaLossWithinBand(goO.loss, cudaO.loss) {
		return fmt.Errorf("loss outside CUDA float2 band: go=%g cuda=%g", goO.loss, cudaO.loss)
	}

	return nil
}

// runStepTweenSpectrum prints Dense-style dtype×speed table then requires exact_bit on each.
func runStepTweenSpectrum(label string, build func(core.DType) (steptweenOut, []byte, error)) error {
	var benchIters = defaultGoBenchIters
	fmt.Printf("\n      [%s StepTween ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cudaµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		goO, payload, err := build(dt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		benchO, cudaDur, err := runCUDASTW1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda-bench:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		cudaO, _, err := runCUDASTW1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		if err := compareStepTween(cudaO, benchO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s benchmark reset: %v", dt, err))
		}
		var m suites.DetMetrics
		if len(goO.WAfter) > 0 && len(goO.WAfter[0]) > 0 {
			m = suites.CompareF32Spectrum(goO.WAfter[0], cudaO.WAfter[0])
		} else {
			// weightless (e.g. Softmax): band from loss bits
			m = suites.DetMetrics{Band: suites.DetExactBit, ExactPct: 100}
			if !cudaLossWithinBand(goO.loss, cudaO.loss) {
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
		cudaUS := perIterUS(cudaDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, cudaUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %8.1f%%\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spd)
		if err := compareStepTween(goO, cudaO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- Dense StepTween ---

type denseStepTweenChild struct {
	in, out int
	act     core.ActivationType
	W       []float32
	bias    []float64
}

func encodeDenseSTW1(dtype core.DType, lr float64, children []denseStepTweenChild, batch int, xShape, tShape []int32, ticksX, ticksT [][]float32) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("STW1"))
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

func runGoDenseStepTween(dtype core.DType, lr float64, children []denseStepTweenChild, batch int, ticksX, ticksT [][]float32) (steptweenOut, error) {
	var z steptweenOut
	ops := make([]any, len(children))
	for i, ch := range children {
		l, err := dense.NewConfigured(ch.in, ch.out, ch.act, dtype, quant.FormatNone, ch.W)
		if err != nil {
			return z, err
		}
		l.Weights.Bias = append([]float64(nil), ch.bias...)
		disableDenseStepTween(l)
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
		loss, err = parallel.TrainStackMSE(st, xin, tgt, parallel.ModeStepTween, lr)
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

func caseDenseStepTweenMultiHopF32() error {
	const batch = 2
	lr := 1e-2
	children := []denseStepTweenChild{
		{in: 4, out: 3, act: core.ActivationReLU, W: []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1, 0.05, 0.15, -0.25, 0.35}, bias: []float64{0.1, -0.05, 0.02}},
		{in: 3, out: 3, act: core.ActivationLinear, W: []float32{0.2, -0.1, 0.05, 0.1, 0.3, -0.2, -0.15, 0.25, 0.4}, bias: []float64{0.01, -0.02, 0.03}},
		{in: 3, out: 2, act: core.ActivationReLU, W: []float32{0.3, -0.2, 0.1, -0.1, 0.4, 0.2}, bias: []float64{0.05, -0.05}},
	}
	x := []float32{1, 0.5, -0.5, 2, -1, 0.25, 0.75, -0.25}
	tgt := []float32{0.2, 0.4, 0.1, 0.3}
	nTicks := len(children) // warm-up until first pop
	ticksX := make([][]float32, nTicks)
	ticksT := make([][]float32, nTicks)
	for i := 0; i < nTicks; i++ {
		ticksX[i] = append([]float32(nil), x...)
		ticksT[i] = append([]float32(nil), tgt...)
	}
	goO, err := runGoDenseStepTween(core.DTypeFloat32, lr, children, batch, ticksX, ticksT)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseSTW1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, true)
}

func caseDenseStepTween1HopAllDTypes() error {
	const in, out, batch = 6, 4, 2
	var benchIters = defaultGoBenchIters
	W, x, bias := makeSharedInit(out, in, batch)
	tgt := make([]float32, batch*out)
	for i := range tgt {
		tgt[i] = float32(0.1*float64(i+1) + 0.05)
	}
	children := []denseStepTweenChild{{in: in, out: out, act: core.ActivationLinear, W: W, bias: bias}}
	ticksX := [][]float32{x}
	ticksT := [][]float32{tgt}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cudaµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		ch := []denseStepTweenChild{{in: in, out: out, act: core.ActivationLinear, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...)}}
		tx := [][]float32{append([]float32(nil), x...)}
		tt := [][]float32{append([]float32(nil), tgt...)}
		goO, err := runGoDenseStepTween(dt, 1e-2, ch, batch, tx, tt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseSTW1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		benchO, cudaDur, err := runCUDASTW1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda:%v", dt, err))
			continue
		}
		// Compare the measured fixture with an independent fresh run.
		cudaO, _, err := runCUDASTW1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cuda2:%v", dt, err))
			continue
		}
		if err := compareStepTween(cudaO, benchO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s benchmark reset: %v", dt, err))
		}
		m := suites.CompareF32Spectrum(goO.WAfter[0], cudaO.WAfter[0])
		goUS, berr := benchGoUS(func() error {
			_, err := runGoDenseStepTween(dt, 1e-2, ch, batch, tx, tt)
			return err
		}, benchIters)
		if berr != nil {
			fails = append(fails, fmt.Sprintf("%s go-bench:%v", dt, berr))
			continue
		}
		cudaUS := perIterUS(cudaDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, cudaUS)
		fmt.Printf("      %-12s %-12s %7.2f%% %10s %10s %8.1f%%\n",
			dt, m.Band, m.ExactPct, FormatUS(goUS), FormatUS(cudaUS), spd)
		if err := compareStepTween(goO, cudaO, false); err != nil {
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

func encodeSTW1Header(nChildren, nTicks, batch int, lr float64) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte("STW1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(nChildren))
	_ = binary.Write(&in, binary.LittleEndian, int32(nTicks))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return &in
}

func writeShapesTicksST(in *bytes.Buffer, xShape, tShape []int32, ticksX, ticksT [][]float32) {
	_ = binary.Write(in, binary.LittleEndian, int32(len(xShape)))
	for _, d := range xShape {
		_ = binary.Write(in, binary.LittleEndian, d)
	}
	_ = binary.Write(in, binary.LittleEndian, int32(len(tShape)))
	for _, d := range tShape {
		_ = binary.Write(in, binary.LittleEndian, d)
	}
	for i := range ticksX {
		for _, v := range ticksX[i] {
			_ = binary.Write(in, binary.LittleEndian, v)
		}
		for _, v := range ticksT[i] {
			_ = binary.Write(in, binary.LittleEndian, v)
		}
	}
}

func caseRMSNormStepTweenF32() error {
	return runRMSNormStepTween(core.DTypeFloat32, true)
}

func caseRMSNormStepTweenAllDTypes() error {
	return runStepTweenSpectrum("RMSNorm", buildRMSNormStepTween)
}

func buildRMSNormStepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ga, err := weights.MatrixF32(st.Children[0].(*rmsnorm.Layer).Gamma)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{ga}}

	in := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(2)) // RMSNorm
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, 1e-6)
	for _, v := range gamma {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	writeShapesTicksST(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runRMSNormStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildRMSNormStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseLayerNormStepTweenF32() error {
	return runLayerNormStepTween(core.DTypeFloat32, true)
}

func caseLayerNormStepTweenAllDTypes() error {
	return runStepTweenSpectrum("LayerNorm", buildLayerNormStepTween)
}

func buildLayerNormStepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ll := st.Children[0].(*layernorm.Layer)
	gA, err := weights.MatrixF32(ll.Gamma)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	bA, err := weights.MatrixF32(ll.Beta)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{append(append([]float32(nil), gA...), bA...)}}

	in := encodeSTW1Header(1, 1, batch, lr)
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
	writeShapesTicksST(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runLayerNormStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildLayerNormStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseSoftmaxStepTweenF32() error {
	return runSoftmaxStepTween(core.DTypeFloat32, true)
}

func caseSoftmaxStepTweenAllDTypes() error {
	return runStepTweenSpectrum("Softmax", buildSoftmaxStepTween)
}

func buildSoftmaxStepTween(dtype core.DType) (steptweenOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	x := []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1}
	tgt := []float32{0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4}
	cfg := softmax.Config{Dim: dim, Kind: softmax.KindStandard, Temperature: 1}
	l, err := softmax.New(cfg)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	_ = l.SetDType(dtype)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{{}}}

	in := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(5)) // Softmax
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, int32(0)) // Standard
	_ = binary.Write(in, binary.LittleEndian, 1.0)
	writeShapesTicksST(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runSoftmaxStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildSoftmaxStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseSwiGLUStepTweenF32() error {
	return runSwiGLUStepTween(core.DTypeFloat32, true)
}

func caseSwiGLUStepTweenAllDTypes() error {
	return runStepTweenSpectrum("SwiGLU", buildSwiGLUStepTween)
}

func buildSwiGLUStepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, inDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, inDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	sl := st.Children[0].(*swiglu.Layer)
	gW, err := weights.MatrixF32(sl.Gate.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	uW, err := weights.MatrixF32(sl.Up.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	dW, err := weights.MatrixF32(sl.Down.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	wa := append(append(append([]float32(nil), gW...), uW...), dW...)
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(4)) // SwiGLU
	_ = binary.Write(buf, binary.LittleEndian, int32(inDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(inter))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksST(buf, []int32{batch, inDim}, []int32{batch, inDim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSwiGLUStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildSwiGLUStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseSequentialStepTweenF32() error {
	return runSeqResStepTween(true, core.DTypeFloat32, true)
}

func caseSequentialStepTweenAllDTypes() error {
	return runStepTweenSpectrum("Sequential", func(dt core.DType) (steptweenOut, []byte, error) {
		return buildSeqResStepTween(true, dt)
	})
}

func caseResidualStepTweenF32() error {
	return runSeqResStepTween(false, core.DTypeFloat32, true)
}

func caseResidualStepTweenAllDTypes() error {
	return runStepTweenSpectrum("Residual", func(dt core.DType) (steptweenOut, []byte, error) {
		return buildSeqResStepTween(false, dt)
	})
}

func buildSeqResStepTween(isSeq bool, dtype core.DType) (steptweenOut, []byte, error) {

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
			return steptweenOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	} else {
		cfg := residual.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := residual.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return steptweenOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	}
	st, err := parallel.NewStack(op)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	var wa []float32
	if isSeq {
		sl := st.Children[0].(*sequential.Layer)
		for _, ch := range sl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return steptweenOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	} else {
		rl := st.Children[0].(*residual.Layer)
		for _, ch := range rl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return steptweenOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}

	kind := int32(16)
	if !isSeq {
		kind = 17
	}
	buf := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, kind)
	_ = binary.Write(buf, binary.LittleEndian, int32(dim))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(depth))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksST(buf, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSeqResStepTween(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResStepTween(isSeq, dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseMambaStepTweenF32() error {
	return runMambaStepTween(core.DTypeFloat32, true)
}

func caseMambaStepTweenAllDTypes() error {
	return runStepTweenSpectrum("Mamba", buildMambaStepTween)
}

func buildMambaStepTween(dtype core.DType) (steptweenOut, []byte, error) {

	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	const batch = 2
	lr := 1e-2
	// Same init family as Mamba NormalBP leaf spectrum (avoids pathological tiny-dtype packs).
	fc, err := makeMbLeafCase(cfg, batch, dtype, 0x53545042^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	packed := fc.W
	x := fc.x
	tgt := fc.target
	lr = fc.lr
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, packed)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	l, err := mamba.NewConfigured(cfg, dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	disableMambaTrain(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*mamba.Layer)
	wa, err := packMambaWeightsF32(ml)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(18)) // Mamba
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DModel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DState))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Expand))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksST(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMambaStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildMambaStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseEmbeddingStepTweenF32() error { return runEmbeddingStepTween(core.DTypeFloat32, true) }
func caseEmbeddingStepTweenAllDTypes() error {
	return runStepTweenSpectrum("Embedding", buildEmbeddingStepTween)
}

func buildEmbeddingStepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, emb)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*embedding.Layer).Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(6))
	_ = binary.Write(buf, binary.LittleEndian, int32(vocab))
	_ = binary.Write(buf, binary.LittleEndian, int32(emb))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksST(buf, []int32{batch, seq}, []int32{batch, seq, emb}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runEmbeddingStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildEmbeddingStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseMHAStepTweenF32() error { return runMHAStepTween(core.DTypeFloat32, true) }
func caseMHAStepTweenAllDTypes() error {
	return runStepTweenSpectrum("MHA", buildMHAStepTween)
}

func buildMHAStepTween(dtype core.DType) (steptweenOut, []byte, error) {

	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 16
	const batch, seq = 2, 4
	fc, err := makeMHANBPCase(cfg, batch, seq, dtype, 0x534d4841, 1e-2)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	c := fc.cfg // validated
	l, err := mha.NewConfigured(c, dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, c.DModel)
	copy(xin.Data, fc.x)
	target := core.NewTensor[float32](batch, seq, c.DModel)
	copy(target.Data, fc.target)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, fc.lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := exportMHAWeights(st.Children[0].(*mha.Layer))
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, fc.lr)
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
	writeShapesTicksST(buf, []int32{batch, seq, int32(c.DModel)}, []int32{batch, seq, int32(c.DModel)},
		[][]float32{fc.x}, [][]float32{fc.target})
	return goO, buf.Bytes(), nil
}

func runMHAStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildMHAStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseCNN1StepTweenF32() error { return runCNN1StepTween(core.DTypeFloat32, true) }
func caseCNN1StepTweenAllDTypes() error {
	return runStepTweenSpectrum("CNN1", buildCNN1StepTween)
}

func buildCNN1StepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn1.Layer).Proj.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
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
	writeShapesTicksST(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runCNN1StepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildCNN1StepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseConvT1StepTweenF32() error { return runConvT1StepTween(core.DTypeFloat32, true) }
func caseConvT1StepTweenAllDTypes() error {
	return runStepTweenSpectrum("ConvT1", buildConvT1StepTween)
}

func buildConvT1StepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt1.Layer).Proj.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
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
	writeShapesTicksST(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runConvT1StepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildConvT1StepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseRNNStepTweenF32() error { return runRNNStepTween(core.DTypeFloat32, true) }
func caseRNNStepTweenAllDTypes() error {
	return runStepTweenSpectrum("RNN", buildRNNStepTween)
}

func buildRNNStepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	disableRNN(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packRNNWeightsF32(st.Children[0].(*rnn.Layer))
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(14))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksST(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runRNNStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildRNNStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseLSTMStepTweenF32() error { return runLSTMStepTween(core.DTypeFloat32, true) }
func caseLSTMStepTweenAllDTypes() error {
	return runStepTweenSpectrum("LSTM", buildLSTMStepTween)
}

func buildLSTMStepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	disableLSTMTrainGates(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packLSTMWeightsF32(st.Children[0].(*lstm.Layer))
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(15))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksST(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runLSTMStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildLSTMStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func buildCNN2StepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn2.Layer).Proj.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
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

func buildCNN3StepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outDepth, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn3.Layer).Proj.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
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

func buildConvT2StepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt2.Layer).Proj.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
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

func buildConvT3StepTween(dtype core.DType) (steptweenOut, []byte, error) {

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
		return steptweenOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.Depth, cfg.Height, cfg.Width)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outDepth, outHeight, outWidth)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt3.Layer).Proj.Weights)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
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

func CasesStepTweenTrain() []Case {
	return append(withTypedVolumes(casesStepTweenBase(), CasesTweenTrain(), "StepTween "), CasesKMeansParallelModes(parallel.ModeStepTween)...)
}

func caseMetacognitionStepTweenF32() error {
	return runMetacognitionStepTween(core.DTypeFloat32, true)
}

func caseMetacognitionStepTweenAllDTypes() error {
	return runStepTweenSpectrum("Metacognition", buildMetacognitionStepTween)
}

func buildMetacognitionStepTween(dtype core.DType) (steptweenOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d435450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicksSTS(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMetacognitionStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseGDNStepTweenF32() error {
	return runGDNStepTween(false)
}

func buildGDNStepTween() (steptweenOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x47444e50)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicksSTS(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runGDNStepTween(exact bool) error {
	goO, payload, err := buildGDNStepTween()
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	for i := range goO.WAfter {
		m := suites.CompareF32Spectrum(goO.WAfter[i], cudaO.WAfter[i])
		if m.Band < suites.DetStrict {
			return fmt.Errorf("GDN W[%d] below strict: %s", i, suites.FormatSpectrum(m))
		}
	}
	return compareStepTween(goO, cudaO, exact)
}

func caseNestedStackStepTweenF32() error {
	return runNestedStackStepTween(core.DTypeFloat32, true)
}

func caseNestedStackStepTweenAllDTypes() error {
	return runStepTweenSpectrum("NestedStack", buildNestedStackStepTween)
}

func buildNestedStackStepTween(dtype core.DType) (steptweenOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53544b50^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTween, lr)
	if err != nil {
		return steptweenOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return steptweenOut{}, nil, err
	}
	goO := steptweenOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTW1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return steptweenOut{}, nil, err
	}
	writeShapesTicksSTS(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runNestedStackStepTween(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackStepTween(dtype)
	if err != nil {
		return err
	}
	cudaO, _, err := runCUDASTW1(payload, 0)
	if err != nil {
		return fmt.Errorf("cuda: %w", err)
	}
	return compareStepTween(goO, cudaO, exact)
}

// --- CASCADE KMeans leaf (auto) ---

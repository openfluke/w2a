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
	"github.com/openfluke/welvet/weights"
)

// CasesStepTweenChainTrain — Line StepTweenChain (FamilyBP on pop @ full lr) Go↔C++ for NormalBP-signed kinds.
func CasesStepTweenChainTrain() []Case {
	cs := []Case{
		{Name: "Dense StepTweenChain multi-hop Float32 — exact bits (W/loss)", Run: caseDenseStepTweenChainMultiHopF32},
		{Name: "Dense StepTweenChain 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseStepTweenChain1HopAllDTypes},
		{Name: "RMSNorm StepTweenChain 1-hop Float32 — exact bits", Run: caseRMSNormStepTweenChainF32},
		{Name: "RMSNorm StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormStepTweenChainAllDTypes},
		{Name: "LayerNorm StepTweenChain 1-hop Float32 — exact bits", Run: caseLayerNormStepTweenChainF32},
		{Name: "LayerNorm StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormStepTweenChainAllDTypes},
		{Name: "Softmax StepTweenChain 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxStepTweenChainF32},
		{Name: "Softmax StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxStepTweenChainAllDTypes},
		{Name: "SwiGLU StepTweenChain 1-hop Float32 — exact bits", Run: caseSwiGLUStepTweenChainF32},
		{Name: "SwiGLU StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUStepTweenChainAllDTypes},
		{Name: "Sequential StepTweenChain 1-hop Float32 — exact bits", Run: caseSequentialStepTweenChainF32},
		{Name: "Sequential StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseSequentialStepTweenChainAllDTypes},
		{Name: "Residual StepTweenChain 1-hop Float32 — exact bits", Run: caseResidualStepTweenChainF32},
		{Name: "Residual StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseResidualStepTweenChainAllDTypes},
		{Name: "Mamba StepTweenChain 1-hop Float32 — exact bits", Run: caseMambaStepTweenChainF32},
		{Name: "Mamba StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseMambaStepTweenChainAllDTypes},
		{Name: "Metacognition StepTweenChain 1-hop Float32 — exact bits", Run: caseMetacognitionStepTweenChainF32},
		{Name: "Metacognition StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionStepTweenChainAllDTypes},
		{Name: "KMeans StepTweenChain 1-hop Float32 — exact bits", Run: caseKMeansStepTweenChainF32},
		{Name: "KMeans StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseKMeansStepTweenChainAllDTypes},
		{Name: "GDN StepTweenChain 1-hop Float32 — exact bits", Run: caseGDNStepTweenChainF32},
		{Name: "NestedStack StepTweenChain 1-hop Float32 — exact bits", Run: caseNestedStackStepTweenChainF32},
		{Name: "NestedStack StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackStepTweenChainAllDTypes},
		{Name: "Embedding StepTweenChain 1-hop Float32 — exact bits", Run: caseEmbeddingStepTweenChainF32},
		{Name: "Embedding StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingStepTweenChainAllDTypes},
		{Name: "MHA StepTweenChain 1-hop Float32 — exact bits", Run: caseMHAStepTweenChainF32},
		{Name: "MHA StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseMHAStepTweenChainAllDTypes},
		{Name: "CNN1 StepTweenChain 1-hop Float32 — exact bits", Run: caseCNN1StepTweenChainF32},
		{Name: "CNN1 StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseCNN1StepTweenChainAllDTypes},
		{Name: "ConvT1 StepTweenChain 1-hop Float32 — exact bits", Run: caseConvT1StepTweenChainF32},
		{Name: "ConvT1 StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseConvT1StepTweenChainAllDTypes},
		{Name: "RNN StepTweenChain 1-hop Float32 — exact bits", Run: caseRNNStepTweenChainF32},
		{Name: "RNN StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseRNNStepTweenChainAllDTypes},
		{Name: "LSTM StepTweenChain 1-hop Float32 — exact bits", Run: caseLSTMStepTweenChainF32},
		{Name: "LSTM StepTweenChain 1-hop × all 34 dtypes — spectrum", Run: caseLSTMStepTweenChainAllDTypes},
		// 3D = MeshTweenChain volume credit (= ModeTweenChain volume wires)
		{Name: "Dense MeshTweenChain volume 1×1×1×1 Float32 — exact bits", Run: caseDenseTweenChainVol111F32},
		{Name: "Dense MeshTweenChain volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseTweenChainVol211F32},
		{Name: "Dense MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseDenseTweenChainVol222F32},
		{Name: "RMSNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormTweenChainVol222F32},
		{Name: "LayerNorm MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormTweenChainVol222F32},
		{Name: "Softmax MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxTweenChainVol222F32},
		{Name: "SwiGLU MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUTweenChainVol222F32},
		{Name: "Sequential MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSeqTweenChainVol222F32},
		{Name: "Residual MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseReTweenChainVol222F32},
		{Name: "Mamba MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseMbTweenChainVol222F32},
		{Name: "Metacognition MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseMcTweenChainVol222F32},
		{Name: "KMeans MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseKmTweenChainVol222F32},
		{Name: "GDN MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseGdTweenChainVol222F32},
		{Name: "NestedStack MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseSkTweenChainVol222F32},
		{Name: "Embedding MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingTweenChainVol222F32},
		{Name: "MHA MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseMHATweenChainVol222F32},
		{Name: "CNN1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(1) }},
		{Name: "CNN2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(2) }},
		{Name: "CNN3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenChainVol222F32(3) }},
		{Name: "ConvT1 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(1) }},
		{Name: "ConvT2 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(2) }},
		{Name: "ConvT3 MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenChainVol222F32(3) }},
		{Name: "RNN MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseRNNTweenChainVol222F32},
		{Name: "LSTM MeshTweenChain volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMTweenChainVol222F32},
	}
	return cs
}

func RunStepTweenChainTrain() error {
	var fails []string
	cs := CasesStepTweenChainTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [steptweenchain %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_steptweenchain_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_steptweenchain_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp steptween_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type steptweenchainOut struct {
	loss   float64
	WAfter [][]float32
}

func disableDenseStepTweenChain(l *dense.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
}

func runCppSTC1(payload []byte, benchIters int) (steptweenchainOut, time.Duration, error) {
	var z steptweenchainOut
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
		return z, 0, fmt.Errorf("cpp stb1: %v: %s", err, stderr.String())
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
	if len(raw) < 8 || string(raw[:4]) != "STCO" {
		return z, bench, fmt.Errorf("cpp steptween bad magic")
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

func compareStepTweenChain(goO, cppO steptweenchainOut, requireExact bool) error {
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
	if !lossWithin1Ulp(goO.loss, cppO.loss) {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact || rel > 1e-5 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

// runStepTweenChainSpectrum prints Dense-style dtype×speed table then requires exact_bit on each.
func runStepTweenChainSpectrum(label string, build func(core.DType) (steptweenchainOut, []byte, error)) error {
	const benchIters = defaultGoBenchIters
	fmt.Printf("\n      [%s StepTweenChain ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		goO, payload, err := build(dt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		_, cppDur, err := runCppSTC1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp-bench:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		cppO, _, err := runCppSTC1(payload, 0)
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
			if !lossWithin1Ulp(goO.loss, cppO.loss) {
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
		if err := compareStepTweenChain(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- Dense StepTweenChain ---

type denseStepTweenChainChild struct {
	in, out int
	act     core.ActivationType
	W       []float32
	bias    []float64
}

func encodeDenseSTC1(dtype core.DType, lr float64, children []denseStepTweenChainChild, batch int, xShape, tShape []int32, ticksX, ticksT [][]float32) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("STC1"))
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

func runGoDenseStepTweenChain(dtype core.DType, lr float64, children []denseStepTweenChainChild, batch int, ticksX, ticksT [][]float32) (steptweenchainOut, error) {
	var z steptweenchainOut
	ops := make([]any, len(children))
	for i, ch := range children {
		l, err := dense.NewConfigured(ch.in, ch.out, ch.act, dtype, quant.FormatNone, ch.W)
		if err != nil {
			return z, err
		}
		l.Weights.Bias = append([]float64(nil), ch.bias...)
		disableDenseStepTweenChain(l)
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
		loss, err = parallel.TrainStackMSE(st, xin, tgt, parallel.ModeStepTweenChain, lr)
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

func caseDenseStepTweenChainMultiHopF32() error {
	const batch = 2
	lr := 1e-2
	children := []denseStepTweenChainChild{
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
	goO, err := runGoDenseStepTweenChain(core.DTypeFloat32, lr, children, batch, ticksX, ticksT)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseSTC1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, true)
}

func caseDenseStepTweenChain1HopAllDTypes() error {
	const in, out, batch, benchIters = 6, 4, 2, defaultGoBenchIters
	W, x, bias := makeSharedInit(out, in, batch)
	tgt := make([]float32, batch*out)
	for i := range tgt {
		tgt[i] = float32(0.1*float64(i+1) + 0.05)
	}
	children := []denseStepTweenChainChild{{in: in, out: out, act: core.ActivationLinear, W: W, bias: bias}}
	ticksX := [][]float32{x}
	ticksT := [][]float32{tgt}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		ch := []denseStepTweenChainChild{{in: in, out: out, act: core.ActivationLinear, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...)}}
		tx := [][]float32{append([]float32(nil), x...)}
		tt := [][]float32{append([]float32(nil), tgt...)}
		goO, err := runGoDenseStepTweenChain(dt, 1e-2, ch, batch, tx, tt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseSTC1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		cppO, cppDur, err := runCppSTC1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			continue
		}
		// fresh cpp once for compare (bench mutated)
		cppO, _, err = runCppSTC1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp2:%v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS, berr := benchGoUS(func() error {
			_, err := runGoDenseStepTweenChain(dt, 1e-2, ch, batch, tx, tt)
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
		if err := compareStepTweenChain(goO, cppO, false); err != nil {
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

func encodeSTC1Header(nChildren, nTicks, batch int, lr float64) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte("STC1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(nChildren))
	_ = binary.Write(&in, binary.LittleEndian, int32(nTicks))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return &in
}

func writeShapesTicksSTC(in *bytes.Buffer, xShape, tShape []int32, ticksX, ticksT [][]float32) {
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

func caseRMSNormStepTweenChainF32() error {
	return runRMSNormStepTweenChain(core.DTypeFloat32, true)
}

func caseRMSNormStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("RMSNorm", buildRMSNormStepTweenChain)
}

func buildRMSNormStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ga, err := weights.MatrixF32(st.Children[0].(*rmsnorm.Layer).Gamma)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{ga}}

	in := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(2)) // RMSNorm
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, 1e-6)
	for _, v := range gamma {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	writeShapesTicksSTC(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runRMSNormStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildRMSNormStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseLayerNormStepTweenChainF32() error {
	return runLayerNormStepTweenChain(core.DTypeFloat32, true)
}

func caseLayerNormStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("LayerNorm", buildLayerNormStepTweenChain)
}

func buildLayerNormStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ll := st.Children[0].(*layernorm.Layer)
	gA, err := weights.MatrixF32(ll.Gamma)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	bA, err := weights.MatrixF32(ll.Beta)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{append(append([]float32(nil), gA...), bA...)}}

	in := encodeSTC1Header(1, 1, batch, lr)
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
	writeShapesTicksSTC(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runLayerNormStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildLayerNormStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseSoftmaxStepTweenChainF32() error {
	return runSoftmaxStepTweenChain(core.DTypeFloat32, true)
}

func caseSoftmaxStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("Softmax", buildSoftmaxStepTweenChain)
}

func buildSoftmaxStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	x := []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1}
	tgt := []float32{0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4}
	cfg := softmax.Config{Dim: dim, Kind: softmax.KindStandard, Temperature: 1}
	l, err := softmax.New(cfg)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	_ = l.SetDType(dtype)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{{}}}

	in := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(5)) // Softmax
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, int32(0)) // Standard
	_ = binary.Write(in, binary.LittleEndian, 1.0)
	writeShapesTicksSTC(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runSoftmaxStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildSoftmaxStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseSwiGLUStepTweenChainF32() error {
	return runSwiGLUStepTweenChain(core.DTypeFloat32, true)
}

func caseSwiGLUStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("SwiGLU", buildSwiGLUStepTweenChain)
}

func buildSwiGLUStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, inDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, inDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	sl := st.Children[0].(*swiglu.Layer)
	gW, err := weights.MatrixF32(sl.Gate.Weights)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	uW, err := weights.MatrixF32(sl.Up.Weights)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	dW, err := weights.MatrixF32(sl.Down.Weights)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	wa := append(append(append([]float32(nil), gW...), uW...), dW...)
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(4)) // SwiGLU
	_ = binary.Write(buf, binary.LittleEndian, int32(inDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(inter))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSTC(buf, []int32{batch, inDim}, []int32{batch, inDim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSwiGLUStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildSwiGLUStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseSequentialStepTweenChainF32() error {
	return runSeqResStepTweenChain(true, core.DTypeFloat32, true)
}

func caseSequentialStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("Sequential", func(dt core.DType) (steptweenchainOut, []byte, error) {
		return buildSeqResStepTweenChain(true, dt)
	})
}

func caseResidualStepTweenChainF32() error {
	return runSeqResStepTweenChain(false, core.DTypeFloat32, true)
}

func caseResidualStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("Residual", func(dt core.DType) (steptweenchainOut, []byte, error) {
		return buildSeqResStepTweenChain(false, dt)
	})
}

func buildSeqResStepTweenChain(isSeq bool, dtype core.DType) (steptweenchainOut, []byte, error) {

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
			return steptweenchainOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	} else {
		cfg := residual.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := residual.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return steptweenchainOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	}
	st, err := parallel.NewStack(op)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	var wa []float32
	if isSeq {
		sl := st.Children[0].(*sequential.Layer)
		for _, ch := range sl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return steptweenchainOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	} else {
		rl := st.Children[0].(*residual.Layer)
		for _, ch := range rl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return steptweenchainOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}

	kind := int32(16)
	if !isSeq {
		kind = 17
	}
	buf := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, kind)
	_ = binary.Write(buf, binary.LittleEndian, int32(dim))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(depth))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSTC(buf, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSeqResStepTweenChain(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResStepTweenChain(isSeq, dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseMambaStepTweenChainF32() error {
	return runMambaStepTweenChain(core.DTypeFloat32, true)
}

func caseMambaStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("Mamba", buildMambaStepTweenChain)
}

func buildMambaStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	const batch = 2
	lr := 1e-2
	// Same init family as Mamba NormalBP leaf spectrum (avoids pathological tiny-dtype packs).
	fc, err := makeMbLeafCase(cfg, batch, dtype, 0x53545042^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	packed := fc.W
	x := fc.x
	tgt := fc.target
	lr = fc.lr
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, packed)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	l, err := mamba.NewConfigured(cfg, dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	disableMambaTrain(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*mamba.Layer)
	wa, err := packMambaWeightsF32(ml)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(18)) // Mamba
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DModel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DState))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Expand))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSTC(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMambaStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildMambaStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseEmbeddingStepTweenChainF32() error { return runEmbeddingStepTweenChain(core.DTypeFloat32, true) }
func caseEmbeddingStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("Embedding", buildEmbeddingStepTweenChain)
}

func buildEmbeddingStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, emb)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*embedding.Layer).Weights)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(6))
	_ = binary.Write(buf, binary.LittleEndian, int32(vocab))
	_ = binary.Write(buf, binary.LittleEndian, int32(emb))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSTC(buf, []int32{batch, seq}, []int32{batch, seq, emb}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runEmbeddingStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildEmbeddingStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseMHAStepTweenChainF32() error { return runMHAStepTweenChain(core.DTypeFloat32, true) }
func caseMHAStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("MHA", buildMHAStepTweenChain)
}

func buildMHAStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 16
	const batch, seq = 2, 4
	fc, err := makeMHANBPCase(cfg, batch, seq, dtype, 0x534d4841, 1e-2)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	c := fc.cfg // validated
	l, err := mha.NewConfigured(c, dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, c.DModel)
	copy(xin.Data, fc.x)
	target := core.NewTensor[float32](batch, seq, c.DModel)
	copy(target.Data, fc.target)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, fc.lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := exportMHAWeights(st.Children[0].(*mha.Layer))
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, fc.lr)
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
	writeShapesTicksSTC(buf, []int32{batch, seq, int32(c.DModel)}, []int32{batch, seq, int32(c.DModel)},
		[][]float32{fc.x}, [][]float32{fc.target})
	return goO, buf.Bytes(), nil
}

func runMHAStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildMHAStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseCNN1StepTweenChainF32() error { return runCNN1StepTweenChain(core.DTypeFloat32, true) }
func caseCNN1StepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("CNN1", buildCNN1StepTweenChain)
}

func buildCNN1StepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn1.Layer).Proj.Weights)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
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
	writeShapesTicksSTC(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runCNN1StepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildCNN1StepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseConvT1StepTweenChainF32() error { return runConvT1StepTweenChain(core.DTypeFloat32, true) }
func caseConvT1StepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("ConvT1", buildConvT1StepTweenChain)
}

func buildConvT1StepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt1.Layer).Proj.Weights)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
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
	writeShapesTicksSTC(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runConvT1StepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildConvT1StepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseRNNStepTweenChainF32() error { return runRNNStepTweenChain(core.DTypeFloat32, true) }
func caseRNNStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("RNN", buildRNNStepTweenChain)
}

func buildRNNStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	disableRNN(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packRNNWeightsF32(st.Children[0].(*rnn.Layer))
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(14))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSTC(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runRNNStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildRNNStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseLSTMStepTweenChainF32() error { return runLSTMStepTweenChain(core.DTypeFloat32, true) }
func caseLSTMStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("LSTM", buildLSTMStepTweenChain)
}

func buildLSTMStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {

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
		return steptweenchainOut{}, nil, err
	}
	disableLSTMTrainGates(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packLSTMWeightsF32(st.Children[0].(*lstm.Layer))
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(15))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSTC(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runLSTMStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildLSTMStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

// --- CASCADE Meta/GDN/Stack leaf (auto) ---

func caseMetacognitionStepTweenChainF32() error {
	return runMetacognitionStepTweenChain(core.DTypeFloat32, true)
}

func caseMetacognitionStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("Metacognition", buildMetacognitionStepTweenChain)
}

func buildMetacognitionStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d435450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicksSTC(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMetacognitionStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseGDNStepTweenChainF32() error {
	return runGDNStepTweenChain(true)
}

func buildGDNStepTweenChain() (steptweenchainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x47444e50)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicksSTC(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runGDNStepTweenChain(exact bool) error {
	goO, payload, err := buildGDNStepTweenChain()
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

func caseNestedStackStepTweenChainF32() error {
	return runNestedStackStepTweenChain(core.DTypeFloat32, true)
}

func caseNestedStackStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("NestedStack", buildNestedStackStepTweenChain)
}

func buildNestedStackStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53544b50^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return steptweenchainOut{}, nil, err
	}
	writeShapesTicksSTC(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runNestedStackStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}

// --- CASCADE KMeans leaf (auto) ---

func caseKMeansStepTweenChainF32() error {
	return runKMeansStepTweenChain(core.DTypeFloat32, true)
}

func caseKMeansStepTweenChainAllDTypes() error {
	return runStepTweenChainSpectrum("KMeans", buildKMeansStepTweenChain)
}

func buildKMeansStepTweenChain(dtype core.DType) (steptweenchainOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeKmModeLeaf(dtype, 0x4b4d5450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	st, err := newKmStack(cfg, dtype, packed)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	outDim := cfg.OutDim()
	xin := core.NewTensor[float32](batch, seq, cfg.FeatureDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, outDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenChain, lr)
	if err != nil {
		return steptweenchainOut{}, nil, fmt.Errorf("go: %w", err)
	}
	kl := st.Children[0].(*kmeans.Layer)
	wa, err := packKMeansWeightsF32(kl)
	if err != nil {
		return steptweenchainOut{}, nil, err
	}
	goO := steptweenchainOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTC1Header(1, 1, batch, lr)
	writeKmChild(buf, cfg, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.FeatureDim)},
		[]int32{int32(batch), int32(seq), int32(outDim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runKMeansStepTweenChain(dtype core.DType, exact bool) error {
	goO, payload, err := buildKMeansStepTweenChain(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTC1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenChain(goO, cppO, exact)
}


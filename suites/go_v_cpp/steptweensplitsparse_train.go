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

// CasesStepTweenSplitSparseTrain — Line StepTweenSplitSparse (FamilyTweenSplitSparse head Jᵀgy + hidden 1/(N−1) P(gx)) Go↔C++ for NormalBP-signed kinds.
func CasesStepTweenSplitSparseTrain() []Case {
	cs := []Case{
		{Name: "Dense StepTweenSplitSparse multi-hop Float32 — exact bits (W/loss)", Run: caseDenseStepTweenSplitSparseMultiHopF32},
		{Name: "Dense StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseStepTweenSplitSparse1HopAllDTypes},
		{Name: "RMSNorm StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseRMSNormStepTweenSplitSparseF32},
		{Name: "RMSNorm StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormStepTweenSplitSparseAllDTypes},
		{Name: "LayerNorm StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseLayerNormStepTweenSplitSparseF32},
		{Name: "LayerNorm StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormStepTweenSplitSparseAllDTypes},
		{Name: "Softmax StepTweenSplitSparse 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxStepTweenSplitSparseF32},
		{Name: "Softmax StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxStepTweenSplitSparseAllDTypes},
		{Name: "SwiGLU StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseSwiGLUStepTweenSplitSparseF32},
		{Name: "SwiGLU StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUStepTweenSplitSparseAllDTypes},
		{Name: "Sequential StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseSequentialStepTweenSplitSparseF32},
		{Name: "Sequential StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseSequentialStepTweenSplitSparseAllDTypes},
		{Name: "Residual StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseResidualStepTweenSplitSparseF32},
		{Name: "Residual StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseResidualStepTweenSplitSparseAllDTypes},
		{Name: "Mamba StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseMambaStepTweenSplitSparseF32},
		{Name: "Mamba StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseMambaStepTweenSplitSparseAllDTypes},
		{Name: "Metacognition StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseMetacognitionStepTweenSplitSparseF32},
		{Name: "Metacognition StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionStepTweenSplitSparseAllDTypes},
		{Name: "GDN StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseGDNStepTweenSplitSparseF32},
		{Name: "NestedStack StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseNestedStackStepTweenSplitSparseF32},
		{Name: "NestedStack StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackStepTweenSplitSparseAllDTypes},
		{Name: "Embedding StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseEmbeddingStepTweenSplitSparseF32},
		{Name: "Embedding StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingStepTweenSplitSparseAllDTypes},
		{Name: "MHA StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseMHAStepTweenSplitSparseF32},
		{Name: "MHA StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseMHAStepTweenSplitSparseAllDTypes},
		{Name: "CNN1 StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseCNN1StepTweenSplitSparseF32},
		{Name: "CNN1 StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseCNN1StepTweenSplitSparseAllDTypes},
		{Name: "ConvT1 StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseConvT1StepTweenSplitSparseF32},
		{Name: "ConvT1 StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseConvT1StepTweenSplitSparseAllDTypes},
		{Name: "RNN StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseRNNStepTweenSplitSparseF32},
		{Name: "RNN StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseRNNStepTweenSplitSparseAllDTypes},
		{Name: "LSTM StepTweenSplitSparse 1-hop Float32 — exact bits", Run: caseLSTMStepTweenSplitSparseF32},
		{Name: "LSTM StepTweenSplitSparse 1-hop × all 34 dtypes — spectrum", Run: caseLSTMStepTweenSplitSparseAllDTypes},
		// 3D = TweenSplitSparse volume credit (= ModeTweenSplit / TweenSplit volume wires)
		{Name: "Dense TweenSplitSparse volume 1×1×1×1 Float32 — exact bits", Run: caseDenseTweenSplitSparseVol111F32},
		{Name: "Dense TweenSplitSparse volume 2×1×1×1 chain Float32 — exact bits", Run: caseDenseTweenSplitSparseVol211F32},
		{Name: "Dense TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseDenseTweenSplitSparseVol222F32},
		{Name: "RMSNorm TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormTweenSplitSparseVol222F32},
		{Name: "LayerNorm TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormTweenSplitSparseVol222F32},
		{Name: "Softmax TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxTweenSplitSparseVol222F32},
		{Name: "SwiGLU TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUTweenSplitSparseVol222F32},
		{Name: "Sequential TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseSeqTweenSplitSparseVol222F32},
		{Name: "Residual TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseReTweenSplitSparseVol222F32},
		{Name: "Mamba TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseMbTweenSplitSparseVol222F32},
		{Name: "Metacognition TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseMcTweenSplitSparseVol222F32},
		{Name: "GDN TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseGdTweenSplitSparseVol222F32},
		{Name: "NestedStack TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseSkTweenSplitSparseVol222F32},
		{Name: "Embedding TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingTweenSplitSparseVol222F32},
		{Name: "MHA TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseMHATweenSplitSparseVol222F32},
		{Name: "CNN1 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenSplitSparseVol222F32(1) }},
		{Name: "CNN2 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenSplitSparseVol222F32(2) }},
		{Name: "CNN3 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNTweenSplitSparseVol222F32(3) }},
		{Name: "ConvT1 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenSplitSparseVol222F32(1) }},
		{Name: "ConvT2 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenSplitSparseVol222F32(2) }},
		{Name: "ConvT3 TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTTweenSplitSparseVol222F32(3) }},
		{Name: "RNN TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseRNNTweenSplitSparseVol222F32},
		{Name: "LSTM TweenSplitSparse volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMTweenSplitSparseVol222F32},
	}
	return cs
}

func RunStepTweenSplitSparseTrain() error {
	var fails []string
	cs := CasesStepTweenSplitSparseTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [steptweensplitsparse %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_steptweensplitsparse_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_steptweensplitsparse_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp steptweensplitsparse_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type steptweensplitsparseOut struct {
	loss   float64
	WAfter [][]float32
}

func disableDenseStepTweenSplitSparse(l *dense.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
}

func runCppSSP1(payload []byte, benchIters int) (steptweensplitsparseOut, time.Duration, error) {
	var z steptweensplitsparseOut
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
		return z, 0, fmt.Errorf("cpp sts1: %v: %s", err, stderr.String())
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
	if len(raw) < 8 || string(raw[:4]) != "SSPO" {
		return z, bench, fmt.Errorf("cpp steptweensplitsparse bad magic")
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

func compareStepTweenSplitSparse(goO, cppO steptweensplitsparseOut, requireExact bool) error {
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

// runStepTweenSplitSparseSpectrum prints Dense-style dtype×speed table then requires exact_bit on each.
func runStepTweenSplitSparseSpectrum(label string, build func(core.DType) (steptweensplitsparseOut, []byte, error)) error {
	const benchIters = 20
	fmt.Printf("\n      [%s StepTweenSplitSparse ×34]\n", label)
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
		_, cppDur, err := runCppSSP1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp-bench:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10.2f %10s %8s  %v\n", dt, suites.DetBroken, "—", float64(goDur.Nanoseconds())/1000, "—", "—", err)
			continue
		}
		cppO, _, err := runCppSSP1(payload, 0)
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
		if err := compareStepTweenSplitSparse(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- Dense StepTweenSplitSparse ---

type denseStepTweenSplitSparseChild struct {
	in, out int
	act     core.ActivationType
	W       []float32
	bias    []float64
}

func encodeDenseSSP1(dtype core.DType, lr float64, children []denseStepTweenSplitSparseChild, batch int, xShape, tShape []int32, ticksX, ticksT [][]float32) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("SSP1"))
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

func runGoDenseStepTweenSplitSparse(dtype core.DType, lr float64, children []denseStepTweenSplitSparseChild, batch int, ticksX, ticksT [][]float32) (steptweensplitsparseOut, error) {
	var z steptweensplitsparseOut
	ops := make([]any, len(children))
	for i, ch := range children {
		l, err := dense.NewConfigured(ch.in, ch.out, ch.act, dtype, quant.FormatNone, ch.W)
		if err != nil {
			return z, err
		}
		l.Weights.Bias = append([]float64(nil), ch.bias...)
		disableDenseStepTweenSplitSparse(l)
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
		loss, err = parallel.TrainStackMSE(st, xin, tgt, parallel.ModeStepTweenSplitSparse, lr)
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

func caseDenseStepTweenSplitSparseMultiHopF32() error {
	const batch = 2
	lr := 1e-2
	children := []denseStepTweenSplitSparseChild{
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
	goO, err := runGoDenseStepTweenSplitSparse(core.DTypeFloat32, lr, children, batch, ticksX, ticksT)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseSSP1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, true)
}

func caseDenseStepTweenSplitSparse1HopAllDTypes() error {
	const in, out, batch, benchIters = 6, 4, 2, 20
	W, x, bias := makeSharedInit(out, in, batch)
	tgt := make([]float32, batch*out)
	for i := range tgt {
		tgt[i] = float32(0.1*float64(i+1) + 0.05)
	}
	children := []denseStepTweenSplitSparseChild{{in: in, out: out, act: core.ActivationLinear, W: W, bias: bias}}
	ticksX := [][]float32{x}
	ticksT := [][]float32{tgt}
	fmt.Printf("\n      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		ch := []denseStepTweenSplitSparseChild{{in: in, out: out, act: core.ActivationLinear, W: append([]float32(nil), W...), bias: append([]float64(nil), bias...)}}
		tx := [][]float32{append([]float32(nil), x...)}
		tt := [][]float32{append([]float32(nil), tgt...)}
		t0 := time.Now()
		goO, err := runGoDenseStepTweenSplitSparse(dt, 1e-2, ch, batch, tx, tt)
		goDur := time.Since(t0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseSSP1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		cppO, cppDur, err := runCppSSP1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			continue
		}
		// fresh cpp once for compare (bench mutated)
		cppO, _, err = runCppSSP1(payload, 0)
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
		if err := compareStepTweenSplitSparse(goO, cppO, false); err != nil {
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

func encodeSSP1Header(nChildren, nTicks, batch int, lr float64) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte("SSP1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(nChildren))
	_ = binary.Write(&in, binary.LittleEndian, int32(nTicks))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return &in
}

func writeShapesTicksSSP(in *bytes.Buffer, xShape, tShape []int32, ticksX, ticksT [][]float32) {
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

func caseRMSNormStepTweenSplitSparseF32() error {
	return runRMSNormStepTweenSplitSparse(core.DTypeFloat32, true)
}

func caseRMSNormStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("RMSNorm", buildRMSNormStepTweenSplitSparse)
}

func buildRMSNormStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ga, err := weights.MatrixF32(st.Children[0].(*rmsnorm.Layer).Gamma)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{ga}}

	in := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(2)) // RMSNorm
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, 1e-6)
	for _, v := range gamma {
		_ = binary.Write(in, binary.LittleEndian, v)
	}
	writeShapesTicksSSP(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runRMSNormStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildRMSNormStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseLayerNormStepTweenSplitSparseF32() error {
	return runLayerNormStepTweenSplitSparse(core.DTypeFloat32, true)
}

func caseLayerNormStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("LayerNorm", buildLayerNormStepTweenSplitSparse)
}

func buildLayerNormStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ll := st.Children[0].(*layernorm.Layer)
	gA, err := weights.MatrixF32(ll.Gamma)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	bA, err := weights.MatrixF32(ll.Beta)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{append(append([]float32(nil), gA...), bA...)}}

	in := encodeSSP1Header(1, 1, batch, lr)
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
	writeShapesTicksSSP(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runLayerNormStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildLayerNormStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseSoftmaxStepTweenSplitSparseF32() error {
	return runSoftmaxStepTweenSplitSparse(core.DTypeFloat32, true)
}

func caseSoftmaxStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("Softmax", buildSoftmaxStepTweenSplitSparse)
}

func buildSoftmaxStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	x := []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1}
	tgt := []float32{0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4}
	cfg := softmax.Config{Dim: dim, Kind: softmax.KindStandard, Temperature: 1}
	l, err := softmax.New(cfg)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	_ = l.SetDType(dtype)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{{}}}

	in := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(5)) // Softmax
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, int32(0)) // Standard
	_ = binary.Write(in, binary.LittleEndian, 1.0)
	writeShapesTicksSSP(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runSoftmaxStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildSoftmaxStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseSwiGLUStepTweenSplitSparseF32() error {
	return runSwiGLUStepTweenSplitSparse(core.DTypeFloat32, true)
}

func caseSwiGLUStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("SwiGLU", buildSwiGLUStepTweenSplitSparse)
}

func buildSwiGLUStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, inDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, inDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	sl := st.Children[0].(*swiglu.Layer)
	gW, err := weights.MatrixF32(sl.Gate.Weights)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	uW, err := weights.MatrixF32(sl.Up.Weights)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	dW, err := weights.MatrixF32(sl.Down.Weights)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	wa := append(append(append([]float32(nil), gW...), uW...), dW...)
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(4)) // SwiGLU
	_ = binary.Write(buf, binary.LittleEndian, int32(inDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(inter))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSSP(buf, []int32{batch, inDim}, []int32{batch, inDim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSwiGLUStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildSwiGLUStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseSequentialStepTweenSplitSparseF32() error {
	return runSeqResStepTweenSplitSparse(true, core.DTypeFloat32, true)
}

func caseSequentialStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("Sequential", func(dt core.DType) (steptweensplitsparseOut, []byte, error) {
		return buildSeqResStepTweenSplitSparse(true, dt)
	})
}

func caseResidualStepTweenSplitSparseF32() error {
	return runSeqResStepTweenSplitSparse(false, core.DTypeFloat32, true)
}

func caseResidualStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("Residual", func(dt core.DType) (steptweensplitsparseOut, []byte, error) {
		return buildSeqResStepTweenSplitSparse(false, dt)
	})
}

func buildSeqResStepTweenSplitSparse(isSeq bool, dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
			return steptweensplitsparseOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	} else {
		cfg := residual.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := residual.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return steptweensplitsparseOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	}
	st, err := parallel.NewStack(op)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	var wa []float32
	if isSeq {
		sl := st.Children[0].(*sequential.Layer)
		for _, ch := range sl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return steptweensplitsparseOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	} else {
		rl := st.Children[0].(*residual.Layer)
		for _, ch := range rl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return steptweensplitsparseOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}

	kind := int32(16)
	if !isSeq {
		kind = 17
	}
	buf := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, kind)
	_ = binary.Write(buf, binary.LittleEndian, int32(dim))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(depth))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSSP(buf, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runSeqResStepTweenSplitSparse(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResStepTweenSplitSparse(isSeq, dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseMambaStepTweenSplitSparseF32() error {
	return runMambaStepTweenSplitSparse(core.DTypeFloat32, true)
}

func caseMambaStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("Mamba", buildMambaStepTweenSplitSparse)
}

func buildMambaStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	const batch = 2
	lr := 1e-2
	// Same init family as Mamba NormalBP leaf spectrum (avoids pathological tiny-dtype packs).
	fc, err := makeMbLeafCase(cfg, batch, dtype, 0x53545042^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	packed := fc.W
	x := fc.x
	tgt := fc.target
	lr = fc.lr
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, packed)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	l, err := mamba.NewConfigured(cfg, dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	disableMambaTrain(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*mamba.Layer)
	wa, err := packMambaWeightsF32(ml)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(18)) // Mamba
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DModel))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.DState))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Expand))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range packed {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSSP(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.DModel)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMambaStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildMambaStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseEmbeddingStepTweenSplitSparseF32() error { return runEmbeddingStepTweenSplitSparse(core.DTypeFloat32, true) }
func caseEmbeddingStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("Embedding", buildEmbeddingStepTweenSplitSparse)
}

func buildEmbeddingStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, emb)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*embedding.Layer).Weights)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(6))
	_ = binary.Write(buf, binary.LittleEndian, int32(vocab))
	_ = binary.Write(buf, binary.LittleEndian, int32(emb))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSSP(buf, []int32{batch, seq}, []int32{batch, seq, emb}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runEmbeddingStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildEmbeddingStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseMHAStepTweenSplitSparseF32() error { return runMHAStepTweenSplitSparse(core.DTypeFloat32, true) }
func caseMHAStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("MHA", buildMHAStepTweenSplitSparse)
}

func buildMHAStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 16
	const batch, seq = 2, 4
	fc, err := makeMHANBPCase(cfg, batch, seq, dtype, 0x534d4841, 1e-2)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	c := fc.cfg // validated
	l, err := mha.NewConfigured(c, dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, c.DModel)
	copy(xin.Data, fc.x)
	target := core.NewTensor[float32](batch, seq, c.DModel)
	copy(target.Data, fc.target)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, fc.lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := exportMHAWeights(st.Children[0].(*mha.Layer))
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, fc.lr)
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
	writeShapesTicksSSP(buf, []int32{batch, seq, int32(c.DModel)}, []int32{batch, seq, int32(c.DModel)},
		[][]float32{fc.x}, [][]float32{fc.target})
	return goO, buf.Bytes(), nil
}

func runMHAStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildMHAStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseCNN1StepTweenSplitSparseF32() error { return runCNN1StepTweenSplitSparse(core.DTypeFloat32, true) }
func caseCNN1StepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("CNN1", buildCNN1StepTweenSplitSparse)
}

func buildCNN1StepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn1.Layer).Proj.Weights)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
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
	writeShapesTicksSSP(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runCNN1StepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildCNN1StepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseConvT1StepTweenSplitSparseF32() error { return runConvT1StepTweenSplitSparse(core.DTypeFloat32, true) }
func caseConvT1StepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("ConvT1", buildConvT1StepTweenSplitSparse)
}

func buildConvT1StepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt1.Layer).Proj.Weights)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
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
	writeShapesTicksSSP(buf, []int32{batch, int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{batch, int32(cfg.Filters), int32(outLen)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runConvT1StepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildConvT1StepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseRNNStepTweenSplitSparseF32() error { return runRNNStepTweenSplitSparse(core.DTypeFloat32, true) }
func caseRNNStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("RNN", buildRNNStepTweenSplitSparse)
}

func buildRNNStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	disableRNN(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packRNNWeightsF32(st.Children[0].(*rnn.Layer))
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(14))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSSP(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runRNNStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildRNNStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseLSTMStepTweenSplitSparseF32() error { return runLSTMStepTweenSplitSparse(core.DTypeFloat32, true) }
func caseLSTMStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("LSTM", buildLSTMStepTweenSplitSparse)
}

func buildLSTMStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {

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
		return steptweensplitsparseOut{}, nil, err
	}
	disableLSTMTrainGates(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packLSTMWeightsF32(st.Children[0].(*lstm.Layer))
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
	_ = binary.Write(buf, binary.LittleEndian, int32(15))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.InputSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.SeqLen))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
	writeShapesTicksSSP(buf, []int32{batch, int32(cfg.SeqLen), int32(cfg.InputSize)},
		[]int32{batch, int32(cfg.SeqLen), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runLSTMStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildLSTMStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

// --- CASCADE Meta/GDN/Stack leaf (auto) ---

func caseMetacognitionStepTweenSplitSparseF32() error {
	return runMetacognitionStepTweenSplitSparse(core.DTypeFloat32, true)
}

func caseMetacognitionStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("Metacognition", buildMetacognitionStepTweenSplitSparse)
}

func buildMetacognitionStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d435450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicksSSP(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMetacognitionStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseGDNStepTweenSplitSparseF32() error {
	return runGDNStepTweenSplitSparse(true)
}

func buildGDNStepTweenSplitSparse() (steptweensplitsparseOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x47444e50)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicksSSP(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runGDNStepTweenSplitSparse(exact bool) error {
	goO, payload, err := buildGDNStepTweenSplitSparse()
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

func caseNestedStackStepTweenSplitSparseF32() error {
	return runNestedStackStepTweenSplitSparse(core.DTypeFloat32, true)
}

func caseNestedStackStepTweenSplitSparseAllDTypes() error {
	return runStepTweenSplitSparseSpectrum("NestedStack", buildNestedStackStepTweenSplitSparse)
}

func buildNestedStackStepTweenSplitSparse(dtype core.DType) (steptweensplitsparseOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53544b50^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepTweenSplitSparse, lr)
	if err != nil {
		return steptweensplitsparseOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	goO := steptweensplitsparseOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSSP1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return steptweensplitsparseOut{}, nil, err
	}
	writeShapesTicksSSP(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runNestedStackStepTweenSplitSparse(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackStepTweenSplitSparse(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSSP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepTweenSplitSparse(goO, cppO, exact)
}

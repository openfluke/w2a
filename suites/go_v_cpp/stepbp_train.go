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
	"github.com/openfluke/welvet/runtime/step"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

// CasesStepBPTrain — Line StepBP (FamilyBP on pop) Go↔C++ for NormalBP-signed kinds.
func CasesStepBPTrain() []Case {
	return []Case{
		{Name: "Dense StepBP multi-hop Float32 — exact bits (W/loss)", Run: caseDenseStepBPMultiHopF32},
		{Name: "Dense StepBP 1-hop × all 34 dtypes — spectrum + speed", Run: caseDenseStepBP1HopAllDTypes},
		{Name: "RMSNorm StepBP 1-hop Float32 — exact bits", Run: caseRMSNormStepBPF32},
		{Name: "RMSNorm StepBP 1-hop × all 34 dtypes — spectrum", Run: caseRMSNormStepBPAllDTypes},
		{Name: "LayerNorm StepBP 1-hop Float32 — exact bits", Run: caseLayerNormStepBPF32},
		{Name: "LayerNorm StepBP 1-hop × all 34 dtypes — spectrum", Run: caseLayerNormStepBPAllDTypes},
		{Name: "Softmax StepBP 1-hop Float32 — exact bits (loss)", Run: caseSoftmaxStepBPF32},
		{Name: "Softmax StepBP 1-hop × all 34 dtypes — spectrum", Run: caseSoftmaxStepBPAllDTypes},
		{Name: "SwiGLU StepBP 1-hop Float32 — exact bits", Run: caseSwiGLUStepBPF32},
		{Name: "SwiGLU StepBP 1-hop × all 34 dtypes — spectrum", Run: caseSwiGLUStepBPAllDTypes},
		{Name: "Sequential StepBP 1-hop Float32 — exact bits", Run: caseSequentialStepBPF32},
		{Name: "Sequential StepBP 1-hop × all 34 dtypes — spectrum", Run: caseSequentialStepBPAllDTypes},
		{Name: "Residual StepBP 1-hop Float32 — exact bits", Run: caseResidualStepBPF32},
		{Name: "Residual StepBP 1-hop × all 34 dtypes — spectrum", Run: caseResidualStepBPAllDTypes},
		{Name: "Mamba StepBP 1-hop Float32 — exact bits", Run: caseMambaStepBPF32},
		{Name: "Mamba StepBP 1-hop × all 34 dtypes — spectrum", Run: caseMambaStepBPAllDTypes},
		{Name: "Metacognition StepBP 1-hop Float32 — exact bits", Run: caseMetacognitionStepBPF32},
		{Name: "Metacognition StepBP 1-hop × all 34 dtypes — spectrum", Run: caseMetacognitionStepBPAllDTypes},
		{Name: "KMeans StepBP 1-hop Float32 — exact bits", Run: caseKMeansStepBPF32},
		{Name: "KMeans StepBP 1-hop × all 34 dtypes — spectrum", Run: caseKMeansStepBPAllDTypes},
		{Name: "GDN StepBP 1-hop Float32 — exact bits", Run: caseGDNStepBPF32},
		{Name: "NestedStack StepBP 1-hop Float32 — exact bits", Run: caseNestedStackStepBPF32},
		{Name: "NestedStack StepBP 1-hop × all 34 dtypes — spectrum", Run: caseNestedStackStepBPAllDTypes},
		{Name: "Embedding StepBP 1-hop Float32 — exact bits", Run: caseEmbeddingStepBPF32},
		{Name: "Embedding StepBP 1-hop × all 34 dtypes — spectrum", Run: caseEmbeddingStepBPAllDTypes},
		{Name: "MHA StepBP 1-hop Float32 — exact bits", Run: caseMHAStepBPF32},
		{Name: "MHA StepBP 1-hop × all 34 dtypes — spectrum", Run: caseMHAStepBPAllDTypes},
		{Name: "CNN1 StepBP 1-hop Float32 — exact bits", Run: caseCNN1StepBPF32},
		{Name: "CNN1 StepBP 1-hop × all 34 dtypes — spectrum", Run: caseCNN1StepBPAllDTypes},
		{Name: "ConvT1 StepBP 1-hop Float32 — exact bits", Run: caseConvT1StepBPF32},
		{Name: "ConvT1 StepBP 1-hop × all 34 dtypes — spectrum", Run: caseConvT1StepBPAllDTypes},
		{Name: "RNN StepBP 1-hop Float32 — exact bits", Run: caseRNNStepBPF32},
		{Name: "RNN StepBP 1-hop × all 34 dtypes — spectrum", Run: caseRNNStepBPAllDTypes},
		{Name: "LSTM StepBP 1-hop Float32 — exact bits", Run: caseLSTMStepBPF32},
		{Name: "LSTM StepBP 1-hop × all 34 dtypes — spectrum", Run: caseLSTMStepBPAllDTypes},
		{Name: "Dense MeshBP 1×1×1×1 Float32 — exact bits", Run: caseDenseMeshBP111F32},
		{Name: "Dense MeshBP 2×1×1×1 chain Float32 — exact bits", Run: caseDenseMeshBP211F32},
		{Name: "Dense MeshBP 2×2×2×1 Float32 — exact bits", Run: caseDenseMeshBP222F32},
		{Name: "Dense MeshBP volume ModeMeshBP 2×2×2×1 Float32 — exact bits", Run: caseDenseNBPVol222F32},
		{Name: "RMSNorm MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseRMSNormNBPVol222F32},
		{Name: "LayerNorm MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseLayerNormNBPVol222F32},
		{Name: "Softmax MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseSoftmaxNBPVol222F32},
		{Name: "SwiGLU MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseSwiGLUNBPVol222F32},
		{Name: "Sequential MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseSeqNBPVol222F32},
		{Name: "Residual MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseReNBPVol222F32},
		{Name: "Mamba MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseMbNBPVol222F32},
		{Name: "Metacognition MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseMcNBPVol222F32},
		{Name: "KMeans MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseKmNBPVol222F32},
		{Name: "GDN MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseGdNBPVol222F32},
		{Name: "NestedStack MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseStNBPVol222F32},
		{Name: "Embedding MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseEmbeddingNBPVol222F32},
		{Name: "MHA MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseMHANBPVol222F32},
		{Name: "CNN1 MeshBP volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNNBPVol222F32(1) }},
		{Name: "CNN2 MeshBP volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNNBPVol222F32(2) }},
		{Name: "CNN3 MeshBP volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseCNNNBPVol222F32(3) }},
		{Name: "ConvT1 MeshBP volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTNBPVol222F32(1) }},
		{Name: "ConvT2 MeshBP volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTNBPVol222F32(2) }},
		{Name: "ConvT3 MeshBP volume 2×2×2×1 Float32 — exact bits", Run: func() error { return caseConvTNBPVol222F32(3) }},
		{Name: "RNN MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseRNNNBPVol222F32},
		{Name: "LSTM MeshBP volume 2×2×2×1 Float32 — exact bits", Run: caseLSTMNBPVol222F32},
		{Name: "Parallel MeshBP volume 2×2×2×1 Float32 — exact bits", Run: casePnNBPVol222F32},
	}
}

func RunStepBPTrain() error {
	var fails []string
	cs := CasesStepBPTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [stepbp %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_stepbp_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_stepbp_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp stepbp_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

type stepbpOut struct {
	loss   float64
	WAfter [][]float32
}

func disableDense(l *dense.Layer) {
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
}

func runCppSTB1(payload []byte, benchIters int) (stepbpOut, time.Duration, error) {
	var z stepbpOut
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
	if len(raw) < 8 || string(raw[:4]) != "STBO" {
		return z, bench, fmt.Errorf("cpp stepbp bad magic")
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

func compareStepBP(goO, cppO stepbpOut, requireExact bool) error {
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

// runStepBPSpectrum prints Dense-style dtype×speed table then requires exact_bit on each.
func runStepBPSpectrum(label string, build func(core.DType) (stepbpOut, []byte, error)) error {
	const benchIters = defaultGoBenchIters
	fmt.Printf("\n      [%s StepBP ×34]\n", label)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	for _, dt := range core.AllDTypes {
		goO, payload, err := build(dt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		_, cppDur, err := runCppSTB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp-bench:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—", "—", "—", "—", err)
			continue
		}
		cppO, _, err := runCppSTB1(payload, 0)
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
		if err := compareStepBP(goO, cppO, true); err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("dtype fails: %s", strings.Join(fails, "; "))
	}
	return nil
}

// --- Dense StepBP ---

type denseStepChild struct {
	in, out int
	act     core.ActivationType
	W       []float32
	bias    []float64
}

func encodeDenseSTB1(dtype core.DType, lr float64, children []denseStepChild, batch int, xShape, tShape []int32, ticksX, ticksT [][]float32) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("STB1"))
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

func runGoDenseStepBP(dtype core.DType, lr float64, children []denseStepChild, batch int, ticksX, ticksT [][]float32) (stepbpOut, error) {
	var z stepbpOut
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
		loss, err = parallel.TrainStackMSE(st, xin, tgt, parallel.ModeStepBP, lr)
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

func caseDenseStepBPMultiHopF32() error {
	const batch = 2
	lr := 1e-2
	children := []denseStepChild{
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
	goO, err := runGoDenseStepBP(core.DTypeFloat32, lr, children, batch, ticksX, ticksT)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodeDenseSTB1(core.DTypeFloat32, lr, children, batch, []int32{batch, 4}, []int32{batch, 2}, ticksX, ticksT)
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, true)
}

func caseDenseStepBP1HopAllDTypes() error {
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
		goO, err := runGoDenseStepBP(dt, 1e-2, ch, batch, tx, tt)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s go:%v", dt, err))
			continue
		}
		payload := encodeDenseSTB1(dt, 1e-2, ch, batch, []int32{batch, in}, []int32{batch, out}, tx, tt)
		cppO, cppDur, err := runCppSTB1(payload, benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp:%v", dt, err))
			continue
		}
		// fresh cpp once for compare (bench mutated)
		cppO, _, err = runCppSTB1(payload, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s cpp2:%v", dt, err))
			continue
		}
		m := suites.CompareF32Spectrum(goO.WAfter[0], cppO.WAfter[0])
		goUS, berr := benchGoUS(func() error {
			_, err := runGoDenseStepBP(dt, 1e-2, ch, batch, tx, tt)
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
		if err := compareStepBP(goO, cppO, false); err != nil {
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

func encodeSTB1Header(nChildren, nTicks, batch int, lr float64) *bytes.Buffer {
	var in bytes.Buffer
	_, _ = in.Write([]byte("STB1"))
	_ = binary.Write(&in, binary.LittleEndian, int32(nChildren))
	_ = binary.Write(&in, binary.LittleEndian, int32(nTicks))
	_ = binary.Write(&in, binary.LittleEndian, int32(batch))
	_ = binary.Write(&in, binary.LittleEndian, lr)
	return &in
}

func writeShapesTicks(in *bytes.Buffer, xShape, tShape []int32, ticksX, ticksT [][]float32) {
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

func caseRMSNormStepBPF32() error {
	return runRMSNormStepBP(core.DTypeFloat32, true)
}

func caseRMSNormStepBPAllDTypes() error {
	return runStepBPSpectrum("RMSNorm", buildRMSNormStepBP)
}

func buildRMSNormStepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ga, err := weights.MatrixF32(st.Children[0].(*rmsnorm.Layer).Gamma)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{ga}}

	in := encodeSTB1Header(1, 1, batch, lr)
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

func runRMSNormStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildRMSNormStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseLayerNormStepBPF32() error {
	return runLayerNormStepBP(core.DTypeFloat32, true)
}

func caseLayerNormStepBPAllDTypes() error {
	return runStepBPSpectrum("LayerNorm", buildLayerNormStepBP)
}

func buildLayerNormStepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ll := st.Children[0].(*layernorm.Layer)
	gA, err := weights.MatrixF32(ll.Gamma)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	bA, err := weights.MatrixF32(ll.Beta)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{append(append([]float32(nil), gA...), bA...)}}

	in := encodeSTB1Header(1, 1, batch, lr)
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

func runLayerNormStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildLayerNormStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseSoftmaxStepBPF32() error {
	return runSoftmaxStepBP(core.DTypeFloat32, true)
}

func caseSoftmaxStepBPAllDTypes() error {
	return runStepBPSpectrum("Softmax", buildSoftmaxStepBP)
}

func buildSoftmaxStepBP(dtype core.DType) (stepbpOut, []byte, error) {

	const batch, dim = 2, 4
	lr := 1e-2
	x := []float32{0.5, -0.2, 0.1, 0.3, -0.4, 0.6, 0.2, -0.1}
	tgt := []float32{0.25, 0.25, 0.25, 0.25, 0.1, 0.2, 0.3, 0.4}
	cfg := softmax.Config{Dim: dim, Kind: softmax.KindStandard, Temperature: 1}
	l, err := softmax.New(cfg)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	_ = l.SetDType(dtype)
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{{}}}

	in := encodeSTB1Header(1, 1, batch, lr)
	_ = binary.Write(in, binary.LittleEndian, int32(5)) // Softmax
	_ = binary.Write(in, binary.LittleEndian, int32(dim))
	_ = binary.Write(in, binary.LittleEndian, int32(dtype))
	_ = binary.Write(in, binary.LittleEndian, int32(0)) // Standard
	_ = binary.Write(in, binary.LittleEndian, 1.0)
	writeShapesTicks(in, []int32{batch, dim}, []int32{batch, dim}, [][]float32{x}, [][]float32{tgt})
	return goO, in.Bytes(), nil
}

func runSoftmaxStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildSoftmaxStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseSwiGLUStepBPF32() error {
	return runSwiGLUStepBP(core.DTypeFloat32, true)
}

func caseSwiGLUStepBPAllDTypes() error {
	return runStepBPSpectrum("SwiGLU", buildSwiGLUStepBP)
}

func buildSwiGLUStepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, inDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, inDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	sl := st.Children[0].(*swiglu.Layer)
	gW, err := weights.MatrixF32(sl.Gate.Weights)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	uW, err := weights.MatrixF32(sl.Up.Weights)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	dW, err := weights.MatrixF32(sl.Down.Weights)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	wa := append(append(append([]float32(nil), gW...), uW...), dW...)
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runSwiGLUStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildSwiGLUStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseSequentialStepBPF32() error {
	return runSeqResStepBP(true, core.DTypeFloat32, true)
}

func caseSequentialStepBPAllDTypes() error {
	return runStepBPSpectrum("Sequential", func(dt core.DType) (stepbpOut, []byte, error) {
		return buildSeqResStepBP(true, dt)
	})
}

func caseResidualStepBPF32() error {
	return runSeqResStepBP(false, core.DTypeFloat32, true)
}

func caseResidualStepBPAllDTypes() error {
	return runStepBPSpectrum("Residual", func(dt core.DType) (stepbpOut, []byte, error) {
		return buildSeqResStepBP(false, dt)
	})
}

func buildSeqResStepBP(isSeq bool, dtype core.DType) (stepbpOut, []byte, error) {

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
			return stepbpOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	} else {
		cfg := residual.Config{Dim: dim, SeqLen: seq, Depth: depth}
		l, err := residual.NewConfigured(cfg, dtype, quant.FormatNone, packed)
		if err != nil {
			return stepbpOut{}, nil, err
		}
		l.Exec.Backend = core.BackendCPUTiled
		l.Exec.MultiCore = false
		op = l
	}
	st, err := parallel.NewStack(op)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	var wa []float32
	if isSeq {
		sl := st.Children[0].(*sequential.Layer)
		for _, ch := range sl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return stepbpOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	} else {
		rl := st.Children[0].(*residual.Layer)
		for _, ch := range rl.Children {
			w, err := weights.MatrixF32(ch.Weights)
			if err != nil {
				return stepbpOut{}, nil, err
			}
			wa = append(wa, w...)
		}
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}

	kind := int32(16)
	if !isSeq {
		kind = 17
	}
	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runSeqResStepBP(isSeq bool, dtype core.DType, exact bool) error {
	goO, payload, err := buildSeqResStepBP(isSeq, dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseMambaStepBPF32() error {
	return runMambaStepBP(core.DTypeFloat32, true)
}

func caseMambaStepBPAllDTypes() error {
	return runStepBPSpectrum("Mamba", buildMambaStepBP)
}

func buildMambaStepBP(dtype core.DType) (stepbpOut, []byte, error) {

	cfg := mamba.Config{DModel: 4, DState: 2, Expand: 2, SeqLen: 2}
	const batch = 2
	lr := 1e-2
	// Same init family as Mamba NormalBP leaf spectrum (avoids pathological tiny-dtype packs).
	fc, err := makeMbLeafCase(cfg, batch, dtype, 0x53545042^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	packed := fc.W
	x := fc.x
	tgt := fc.target
	lr = fc.lr
	inInit, outInit, aLog, dSkip, err := splitMambaPacked(cfg, packed)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	l, err := mamba.NewConfigured(cfg, dtype, quant.FormatNone, inInit, outInit, aLog, dSkip)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	disableMambaTrain(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.DModel)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*mamba.Layer)
	wa, err := packMambaWeightsF32(ml)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}

	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runMambaStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildMambaStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseEmbeddingStepBPF32() error { return runEmbeddingStepBP(core.DTypeFloat32, true) }
func caseEmbeddingStepBPAllDTypes() error {
	return runStepBPSpectrum("Embedding", buildEmbeddingStepBP)
}

func buildEmbeddingStepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, emb)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*embedding.Layer).Weights)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runEmbeddingStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildEmbeddingStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseMHAStepBPF32() error { return runMHAStepBP(core.DTypeFloat32, true) }
func caseMHAStepBPAllDTypes() error {
	return runStepBPSpectrum("MHA", buildMHAStepBP)
}

func buildMHAStepBP(dtype core.DType) (stepbpOut, []byte, error) {

	cfg := mha.DecoderCausal(8, 2, 2)
	cfg.MaxSeqLen = 16
	const batch, seq = 2, 4
	fc, err := makeMHANBPCase(cfg, batch, seq, dtype, 0x534d4841, 1e-2)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	c := fc.cfg // validated
	l, err := mha.NewConfigured(c, dtype, quant.FormatNone, fc.q, fc.k, fc.v, fc.o)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, c.DModel)
	copy(xin.Data, fc.x)
	target := core.NewTensor[float32](batch, seq, c.DModel)
	copy(target.Data, fc.target)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, fc.lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := exportMHAWeights(st.Children[0].(*mha.Layer))
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, fc.lr)
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

func runMHAStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildMHAStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseCNN1StepBPF32() error { return runCNN1StepBP(core.DTypeFloat32, true) }
func caseCNN1StepBPAllDTypes() error {
	return runStepBPSpectrum("CNN1", buildCNN1StepBP)
}

func buildCNN1StepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*cnn1.Layer).Proj.Weights)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runCNN1StepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildCNN1StepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseConvT1StepBPF32() error { return runConvT1StepBP(core.DTypeFloat32, true) }
func caseConvT1StepBPAllDTypes() error {
	return runStepBPSpectrum("ConvT1", buildConvT1StepBP)
}

func buildConvT1StepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.InChannels, cfg.SeqLen)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.Filters, outLen)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := weights.MatrixF32(st.Children[0].(*convt1.Layer).Proj.Weights)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runConvT1StepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildConvT1StepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseRNNStepBPF32() error { return runRNNStepBP(core.DTypeFloat32, true) }
func caseRNNStepBPAllDTypes() error {
	return runStepBPSpectrum("RNN", buildRNNStepBP)
}

func buildRNNStepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	disableRNN(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packRNNWeightsF32(st.Children[0].(*rnn.Layer))
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runRNNStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildRNNStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseLSTMStepBPF32() error { return runLSTMStepBP(core.DTypeFloat32, true) }
func caseLSTMStepBPAllDTypes() error {
	return runStepBPSpectrum("LSTM", buildLSTMStepBP)
}

func buildLSTMStepBP(dtype core.DType) (stepbpOut, []byte, error) {

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
		return stepbpOut{}, nil, err
	}
	disableLSTMTrainGates(l)
	st, err := parallel.NewStack(l)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, cfg.SeqLen, cfg.InputSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, cfg.SeqLen, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packLSTMWeightsF32(st.Children[0].(*lstm.Layer))
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
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

func runLSTMStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildLSTMStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

// --- Dense MeshBP (runtime/step clock + FamilyBP) ---

type meshbpOut struct {
	loss float64
	Ws   [][]float32
}

func encodeDenseMBP1(depth, rows, cols, lpc, batch, ticks int, cells []volCell, x, target []float32, lr float64) ([]byte, error) {
	var in bytes.Buffer
	_, _ = in.Write([]byte("MBP1"))
	for _, v := range []int32{int32(depth), int32(rows), int32(cols), int32(lpc), int32(batch), 0 /*cpu*/, 0 /*n_links*/, int32(ticks)} {
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

func runCppMBP1(payload []byte, benchIters int) (meshbpOut, time.Duration, error) {
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
		return z, 0, fmt.Errorf("cpp mbp1: %v: %s", err, stderr.String())
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
	if len(raw) < 8 || string(raw[:4]) != "MBPO" {
		return z, bench, fmt.Errorf("cpp meshbp bad magic")
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
		if err := binary.Read(r, binary.LittleEndian, &o); err != nil {
			return z, bench, err
		}
		if err := binary.Read(r, binary.LittleEndian, &inn); err != nil {
			return z, bench, err
		}
		w := make([]float32, o*inn)
		for j := range w {
			if err := binary.Read(r, binary.LittleEndian, &w[j]); err != nil {
				return z, bench, err
			}
		}
		z.Ws = append(z.Ws, w)
	}
	return z, bench, nil
}

func runGoDenseMeshBP(depth, rows, cols, lpc, batch, ticks int, cells []volCell, x, target []float32, lr float64) (meshbpOut, error) {
	var z meshbpOut
	g := architecture.NewGrid(depth, rows, cols, lpc)
	g.Exec.MultiCore = false
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
	tgt := core.NewTensor[float32](batch, len(target)/batch)
	copy(tgt.Data, target)

	st := step.New[float32](g)
	for t := 0; t < ticks; t++ {
		st.SetInput(xin)
		if _, err := step.Forward(g, st, true); err != nil {
			return z, fmt.Errorf("mesh fwd tick %d: %w", t, err)
		}
	}
	if len(st.LayerData) == 0 || st.LayerData[len(st.LayerData)-1] == nil {
		return z, fmt.Errorf("empty mesh output")
	}
	out := st.LayerData[len(st.LayerData)-1]
	if out.Len() != tgt.Len() {
		tgt = core.NewTensor[float32](out.Shape...)
		copy(tgt.Data, target)
	}
	loss, err := training.MSE(out, tgt)
	if err != nil {
		return z, err
	}
	gy, err := training.MSEGrad(out, tgt)
	if err != nil {
		return z, err
	}
	_, grads, err := step.Backward(g, st, gy)
	if err != nil {
		return z, err
	}
	for i, pair := range grads {
		if pair[1] == nil {
			continue
		}
		cell := &g.Cells[i]
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return z, fmt.Errorf("expected dense at idx %d", i)
		}
		if err := dense.ApplyGradSGD(dl, pair[1], lr); err != nil {
			return z, err
		}
	}
	z.loss = loss
	for i, c := range hops {
		if !cells[i].enabled {
			continue
		}
		cell := g.At(c.Z, c.Y, c.X, c.L)
		dl, ok := cell.Op.(*dense.Layer)
		if !ok {
			return z, fmt.Errorf("expected dense at %v", c)
		}
		Wa, err := weights.MatrixF32(dl.Weights)
		if err != nil {
			return z, err
		}
		z.Ws = append(z.Ws, append([]float32(nil), Wa...))
	}
	return z, nil
}

func compareMeshBP(goO, cppO meshbpOut, requireExact bool) error {
	if len(goO.Ws) != len(cppO.Ws) {
		return fmt.Errorf("n_cells go=%d cpp=%d", len(goO.Ws), len(cppO.Ws))
	}
	for i := range goO.Ws {
		m := suites.CompareF32Spectrum(goO.Ws[i], cppO.Ws[i])
		if requireExact && m.Band != suites.DetExactBit {
			return fmt.Errorf("W[%d]: %s", i, suites.FormatSpectrum(m))
		}
	}
	if !lossWithin1Ulp(goO.loss, cppO.loss) {
		rel := math.Abs(goO.loss-cppO.loss) / math.Max(1e-30, math.Abs(goO.loss))
		if requireExact && rel > 0 {
			return fmt.Errorf("loss go=%g cpp=%g", goO.loss, cppO.loss)
		}
	}
	return nil
}

func runDenseMeshBPCase(depth, rows, cols, lpc, batch, ticks, benchIters int) error {
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
	goO, err := runGoDenseMeshBP(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload, err := encodeDenseMBP1(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
	if err != nil {
		return err
	}
	_, cppDur, err := runCppMBP1(payload, benchIters)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	cppO, _, err := runCppMBP1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp2: %w", err)
	}
	if err := compareMeshBP(goO, cppO, true); err != nil {
		return err
	}
	goUS, err := benchGoUS(func() error {
		_, _ = runGoDenseMeshBP(depth, rows, cols, lpc, batch, ticks, cells, x, target, 1e-2)
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
	fmt.Printf("\n      MeshBP %dx%dx%dx%d ticks=%d loss=%.6g exact_bit  go=%s  cpp=%s  spd%%Δ=%+.1f%%",
		depth, rows, cols, lpc, ticks, goO.loss, FormatUS(goUS), FormatUS(cppUS), spd)
	return nil
}

func caseDenseMeshBP111F32() error {
	return runDenseMeshBPCase(1, 1, 1, 1, 1, 1, defaultGoBenchIters)
}
func caseDenseMeshBP211F32() error {
	return runDenseMeshBPCase(2, 1, 1, 1, 1, 2, defaultGoBenchIters)
}
func caseDenseMeshBP222F32() error {
	return runDenseMeshBPCase(2, 2, 2, 1, 1, 8, defaultGoBenchIters)
}

// --- CASCADE Meta/GDN/Stack leaf (auto) ---

func caseMetacognitionStepBPF32() error {
	return runMetacognitionStepBP(core.DTypeFloat32, true)
}

func caseMetacognitionStepBPAllDTypes() error {
	return runStepBPSpectrum("Metacognition", buildMetacognitionStepBP)
}

func buildMetacognitionStepBP(dtype core.DType) (stepbpOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeMcModeLeaf(dtype, 0x4d435450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	st, err := newMcStack(cfg, dtype, packed)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.Dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	ml := st.Children[0].(*metacognition.Layer)
	wa, err := packMetacognitionWeightsF32(ml)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
	writeMcChild(buf, cfg.Dim, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.Dim)},
		[]int32{int32(batch), int32(seq), int32(cfg.Dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runMetacognitionStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildMetacognitionStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseGDNStepBPF32() error {
	return runGDNStepBP(true)
}

func buildGDNStepBP() (stepbpOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeGdModeLeaf(0x47444e50)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	st, err := newGdStack(cfg, packed)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, cfg.HiddenSize)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	gl := st.Children[0].(*gdn.Layer)
	wa, err := packGDNWeightsF32(gl)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
	writeGdChild(buf, cfg, seq, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runGDNStepBP(exact bool) error {
	goO, payload, err := buildGDNStepBP()
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

func caseNestedStackStepBPF32() error {
	return runNestedStackStepBP(core.DTypeFloat32, true)
}

func caseNestedStackStepBPAllDTypes() error {
	return runStepBPSpectrum("NestedStack", buildNestedStackStepBP)
}

func buildNestedStackStepBP(dtype core.DType) (stepbpOut, []byte, error) {
	dim, nChildren, batch, seq, packed, x, tgt, lr, err := makeStModeLeaf(dtype, 0x53544b50^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	st, err := buildNestedStackDense(dim, nChildren, dtype, packed)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	xin := core.NewTensor[float32](batch*seq, dim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch*seq, dim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	wa, err := packStackWeightsF32(st)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
	if err := writeNestedStackDenseChild(buf, dim, nChildren, dtype, packed); err != nil {
		return stepbpOut{}, nil, err
	}
	writeShapesTicks(buf, []int32{int32(batch * seq), int32(dim)},
		[]int32{int32(batch * seq), int32(dim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runNestedStackStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildNestedStackStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}

// --- CASCADE KMeans leaf (auto) ---

func caseKMeansStepBPF32() error {
	return runKMeansStepBP(core.DTypeFloat32, true)
}

func caseKMeansStepBPAllDTypes() error {
	return runStepBPSpectrum("KMeans", buildKMeansStepBP)
}

func buildKMeansStepBP(dtype core.DType) (stepbpOut, []byte, error) {
	cfg, batch, seq, packed, x, tgt, lr, err := makeKmModeLeaf(dtype, 0x4b4d5450^uint32(dtype)*0x9e3779b9)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	st, err := newKmStack(cfg, dtype, packed)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	outDim := cfg.OutDim()
	xin := core.NewTensor[float32](batch, seq, cfg.FeatureDim)
	copy(xin.Data, x)
	target := core.NewTensor[float32](batch, seq, outDim)
	copy(target.Data, tgt)
	loss, err := parallel.TrainStackMSE(st, xin, target, parallel.ModeStepBP, lr)
	if err != nil {
		return stepbpOut{}, nil, fmt.Errorf("go: %w", err)
	}
	kl := st.Children[0].(*kmeans.Layer)
	wa, err := packKMeansWeightsF32(kl)
	if err != nil {
		return stepbpOut{}, nil, err
	}
	goO := stepbpOut{loss: loss, WAfter: [][]float32{wa}}
	buf := encodeSTB1Header(1, 1, batch, lr)
	writeKmChild(buf, cfg, seq, dtype, packed)
	writeShapesTicks(buf, []int32{int32(batch), int32(seq), int32(cfg.FeatureDim)},
		[]int32{int32(batch), int32(seq), int32(outDim)}, [][]float32{x}, [][]float32{tgt})
	return goO, buf.Bytes(), nil
}

func runKMeansStepBP(dtype core.DType, exact bool) error {
	goO, payload, err := buildKMeansStepBP(dtype)
	if err != nil {
		return err
	}
	cppO, _, err := runCppSTB1(payload, 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareStepBP(goO, cppO, exact)
}


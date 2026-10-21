package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strings"

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
	"github.com/openfluke/welvet/layers/kmeans"
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
	"github.com/openfluke/welvet/systems/dna"
)

// C++ stack::Kind values.
const (
	skDense         int32 = 1
	skRMSNorm       int32 = 2
	skLayerNorm     int32 = 3
	skSwiGLU        int32 = 4
	skSoftmax       int32 = 5
	skEmbedding     int32 = 6
	skMHA           int32 = 7
	skCNN1          int32 = 8
	skCNN2          int32 = 9
	skCNN3          int32 = 10
	skConvT1        int32 = 11
	skConvT2        int32 = 12
	skConvT3        int32 = 13
	skRNN           int32 = 14
	skLSTM          int32 = 15
	skSequential    int32 = 16
	skResidual      int32 = 17
	skMamba         int32 = 18
	skNestedStack   int32 = 19
	skMetacognition int32 = 20
	skGDN           int32 = 21
	skKMeans        int32 = 22
	skParallel      int32 = 23
)

type polyBranchEnc struct {
	bytes []byte
	op    any
}

type polyCase struct {
	name       string
	cfgDim     int
	cfgOutFeat int
	cfgSeq     int
	make       func(dt core.DType) (branches []polyBranchEnc, x, target []float32, xShape, tShape []int32, err error)
	spectrum   bool
}

// CasesParallelPolyTrain — Phase E polymorphic Parallel ModeNormalBP Add.
func CasesParallelPolyTrain() []Case {
	var out []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		out = append(out, Case{
			Name: fmt.Sprintf("Parallel poly %s Float32 — exact bits", pc.name),
			Run:  func() error { return runPolyKind(pc, core.DTypeFloat32, true, 20) },
		})
		if pc.spectrum {
			out = append(out, Case{
				Name: fmt.Sprintf("Parallel poly %s ×34 dtypes — spectrum", pc.name),
				Run:  func() error { return runPolyKindSpectrum(pc) },
			})
		}
	}
	return out
}

func RunParallelPolyTrain() error {
	var fails []string
	cs := CasesParallelPolyTrain()
	for i, c := range cs {
		suites.BeginCase()
		fmt.Printf("  [parallel-poly %d] %s … ", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_cpp_parallel_poly_train", c.Name, "FAIL", err.Error())
			fmt.Printf("FAIL\n      %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_cpp_parallel_poly_train", c.Name, "PASS", "")
		fmt.Println("PASS")
	}
	if len(fails) > 0 {
		return fmt.Errorf("go_v_cpp parallel_poly_train: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func polyTrainKinds() []polyCase {
	const batch, dim, seq = 2, 8, 4
	return []polyCase{
		{name: "dense", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeDensePair(dt, batch, dim)
		}},
		{name: "rmsnorm", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeRMSPair(dt, batch, dim)
		}},
		{name: "layernorm", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeLNPair(dt, batch, dim)
		}},
		{name: "swiglu", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeSwiPair(dt, batch, dim)
		}},
		{name: "softmax", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeSoftmaxPair(dt, batch, dim)
		}},
		{name: "sequential", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeSeqResPair(dt, batch, dim, 2, true)
		}},
		{name: "residual", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeSeqResPair(dt, batch, dim, 1, false)
		}},
		{name: "metacognition", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeMetaPair(dt, batch, dim)
		}},
		{name: "mha", cfgDim: dim, cfgOutFeat: dim, cfgSeq: seq, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeMHAPair(dt, batch, seq, dim)
		}},
		{name: "rnn", cfgDim: dim, cfgOutFeat: dim, cfgSeq: seq, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeRNNPair(dt, batch, seq, dim, false)
		}},
		{name: "embedding", cfgDim: dim, cfgOutFeat: 0, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeEmbedPair(dt, batch, seq, dim)
		}},
		{name: "cnn1", cfgDim: 8, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeCNN1Pair(dt, batch)
		}},
		{name: "stack", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeStackPair(dt, batch, dim)
		}},
		{name: "lstm", cfgDim: dim, cfgOutFeat: dim, cfgSeq: seq, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeRNNPair(dt, batch, seq, dim, true)
		}},
		{name: "kmeans", cfgDim: dim, cfgOutFeat: dim, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeKMeansPair(dt, batch, dim)
		}},
		{name: "mamba", cfgDim: 6, cfgOutFeat: 6, cfgSeq: 3, spectrum: true, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeMambaPair(dt, batch)
		}},
		{name: "gdn", cfgDim: 8, cfgOutFeat: 8, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeGDNPair(dt, batch)
		}},
		{name: "cnn2", cfgDim: 8, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeCNN2Pair(dt, batch)
		}},
		{name: "cnn3", cfgDim: 8, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeCNN3Pair(dt, batch)
		}},
		{name: "convt1", cfgDim: 8, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeConvT1Pair(dt, batch)
		}},
		{name: "convt2", cfgDim: 8, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeConvT2Pair(dt, batch)
		}},
		{name: "convt3", cfgDim: 8, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeConvT3Pair(dt, batch)
		}},
		{name: "nested_parallel", cfgDim: dim, cfgOutFeat: dim, spectrum: false, make: func(dt core.DType) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
			return makeNestedParallelPair(dt, batch, dim)
		}},
	}
}

func writeI32(b *bytes.Buffer, vs ...int32) {
	for _, v := range vs {
		_ = binary.Write(b, binary.LittleEndian, v)
	}
}
func writeF32s(b *bytes.Buffer, vs []float32) {
	for _, v := range vs {
		_ = binary.Write(b, binary.LittleEndian, v)
	}
}
func writeF64(b *bytes.Buffer, v float64) { _ = binary.Write(b, binary.LittleEndian, v) }

func seedPat(n int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*1664525 + 1013904223
		out[i] = 0.05 * float32(int32(s%2001)-1000) / 100.0
	}
	return out
}

func polyEncDense(in, out int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skDense, int32(in), int32(out), -1, int32(dt))
	writeF32s(&b, W)
	_ = binary.Write(&b, binary.LittleEndian, uint32(0))
	return b.Bytes()
}

func makeDensePair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	Wa, Wb := seedPat(dim*dim, 0xD001), seedPat(dim*dim, 0xD002)
	a, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0xD010), seedPat(batch*dim, 0xD020)
	return []polyBranchEnc{{polyEncDense(dim, dim, dt, Wa), a}, {polyEncDense(dim, dim, dt, Wb), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncRMS(dim int, dt core.DType, eps float64, g []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skRMSNorm, int32(dim), int32(dt))
	writeF64(&b, eps)
	writeF32s(&b, g)
	return b.Bytes()
}

func makeRMSPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	ga, gb := seedPat(dim, 0xA001), seedPat(dim, 0xA002)
	for i := range ga {
		ga[i] = 0.5 + ga[i]*0.1
		gb[i] = 0.5 + gb[i]*0.1
	}
	a, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim, Eps: 1e-6}, dt, quant.FormatNone, ga)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim, Eps: 1e-6}, dt, quant.FormatNone, gb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0xA010), seedPat(batch*dim, 0xA020)
	return []polyBranchEnc{{polyEncRMS(dim, dt, 1e-6, ga), a}, {polyEncRMS(dim, dt, 1e-6, gb), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncLN(dim int, dt core.DType, eps float64, g, beta []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skLayerNorm, int32(dim), int32(dt))
	writeF64(&b, eps)
	writeF32s(&b, g)
	writeF32s(&b, beta)
	return b.Bytes()
}

func makeLNPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	ga, ba := seedPat(dim, 0xB001), seedPat(dim, 0xB011)
	gb, bb := seedPat(dim, 0xB002), seedPat(dim, 0xB012)
	for i := range ga {
		ga[i] = 0.5 + ga[i]*0.1
		gb[i] = 0.5 + gb[i]*0.1
	}
	a, err := layernorm.NewConfigured(layernorm.Config{Dim: dim, Eps: 1e-5}, dt, quant.FormatNone, ga, ba)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := layernorm.NewConfigured(layernorm.Config{Dim: dim, Eps: 1e-5}, dt, quant.FormatNone, gb, bb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0xB010), seedPat(batch*dim, 0xB020)
	return []polyBranchEnc{{polyEncLN(dim, dt, 1e-5, ga, ba), a}, {polyEncLN(dim, dt, 1e-5, gb, bb), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncSwi(in, inter int, dt core.DType, packed []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skSwiGLU, int32(in), int32(inter), int32(dt))
	writeF32s(&b, packed)
	return b.Bytes()
}

func makeSwiPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	inter := dim * 2
	n := inter*dim*2 + dim*inter
	pa, pb := seedPat(n, 0xC001), seedPat(n, 0xC002)
	cfg := swiglu.Config{InputDim: dim, IntermediateDim: inter}
	ng := inter * dim
	a, err := swiglu.NewConfigured(cfg, dt, quant.FormatNone, pa[:ng], pa[ng:2*ng], pa[2*ng:])
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := swiglu.NewConfigured(cfg, dt, quant.FormatNone, pb[:ng], pb[ng:2*ng], pb[2*ng:])
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0xC010), seedPat(batch*dim, 0xC020)
	return []polyBranchEnc{{polyEncSwi(dim, inter, dt, pa), a}, {polyEncSwi(dim, inter, dt, pb), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncSoftmax(dim int, dt core.DType) []byte {
	var b bytes.Buffer
	writeI32(&b, skSoftmax, int32(dim), int32(dt), 0)
	writeF64(&b, 1.0)
	return b.Bytes()
}

func makeSoftmaxPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	a, err := softmax.New(softmax.Config{Dim: dim, SeqLen: 1})
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	_ = a.SetDType(dt)
	b, err := softmax.New(softmax.Config{Dim: dim, SeqLen: 1})
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	_ = b.SetDType(dt)
	x, tgt := seedPat(batch*dim, 0xF010), seedPat(batch*dim, 0xF020)
	return []polyBranchEnc{{polyEncSoftmax(dim, dt), a}, {polyEncSoftmax(dim, dt), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncSeqRes(kind int32, dim, seq, depth int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, kind, int32(dim), int32(seq), int32(depth), -1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeSeqResPair(dt core.DType, batch, dim, depth int, isSeq bool) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	Wa, Wb := seedPat(depth*dim*dim, 0xE001), seedPat(depth*dim*dim, 0xE002)
	var a, b any
	var err error
	kind := skResidual
	if isSeq {
		kind = skSequential
		a, err = sequential.NewConfigured(sequential.Config{Dim: dim, Depth: depth}, dt, quant.FormatNone, Wa)
		if err != nil {
			return nil, nil, nil, nil, nil, err
		}
		b, err = sequential.NewConfigured(sequential.Config{Dim: dim, Depth: depth}, dt, quant.FormatNone, Wb)
	} else {
		a, err = residual.NewConfigured(residual.Config{Dim: dim, Depth: depth}, dt, quant.FormatNone, Wa)
		if err != nil {
			return nil, nil, nil, nil, nil, err
		}
		b, err = residual.NewConfigured(residual.Config{Dim: dim, Depth: depth}, dt, quant.FormatNone, Wb)
	}
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0xE010), seedPat(batch*dim, 0xE020)
	return []polyBranchEnc{{polyEncSeqRes(kind, dim, 1, depth, dt, Wa), a}, {polyEncSeqRes(kind, dim, 1, depth, dt, Wb), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncKMeans(k, d int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	// C++: k, d, seq, output_mode, act, temperature, dtype — OutputFeatures=1
	writeI32(&b, skKMeans, int32(k), int32(d), 1, 1, -1)
	writeF64(&b, 1.0)
	writeI32(&b, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeKMeansPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	k := 4
	Wa, Wb := seedPat(k*dim, 0x21001), seedPat(k*dim, 0x21002)
	// ActivationLinear must match polyEncKMeans act=-1 (Go zero-value Activation is ReLU=0).
	cfg := kmeans.Config{
		NumClusters: k, FeatureDim: dim, Temperature: 1,
		OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear,
	}
	a, err := kmeans.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := kmeans.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0x21010), seedPat(batch*dim, 0x21020)
	return []polyBranchEnc{{polyEncKMeans(k, dim, dt, Wa), a}, {polyEncKMeans(k, dim, dt, Wb), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncMeta(dim, seq int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skMetacognition, int32(dim), int32(seq), int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeMetaPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := metacognition.Config{Dim: dim}
	Wa, Wb := seedPat(dim*dim, 0x31001), seedPat(dim*dim, 0x31002)
	a, err := metacognition.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := metacognition.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0x31010), seedPat(batch*dim, 0x31020)
	return []polyBranchEnc{{polyEncMeta(dim, 1, dt, Wa), a}, {polyEncMeta(dim, 1, dt, Wb), b}},
		x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncMHA(seq, dModel, heads, kv, hd, msl int, dt core.DType, Q, K, V, O []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skMHA, int32(seq), int32(dModel), int32(heads), int32(kv), int32(hd), int32(msl), 1, 0, int32(dt))
	writeF32s(&b, Q)
	writeF32s(&b, K)
	writeF32s(&b, V)
	writeF32s(&b, O)
	return b.Bytes()
}

func makeMHAPair(dt core.DType, batch, seq, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	heads := 2
	hd := dim / heads
	q := heads * hd
	nq := q * dim
	cfg := mha.Config{DModel: dim, NumHeads: heads, MaxSeqLen: seq, Causal: true}
	Qa, Ka, Va, Oa := seedPat(nq, 0x41001), seedPat(nq, 0x41011), seedPat(nq, 0x41021), seedPat(nq, 0x41031)
	Qb, Kb, Vb, Ob := seedPat(nq, 0x41002), seedPat(nq, 0x41012), seedPat(nq, 0x41022), seedPat(nq, 0x41032)
	a, err := mha.NewConfigured(cfg, dt, quant.FormatNone, Qa, Ka, Va, Oa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := mha.NewConfigured(cfg, dt, quant.FormatNone, Qb, Kb, Vb, Ob)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*seq*dim, 0x41010), seedPat(batch*seq*dim, 0x41020)
	return []polyBranchEnc{
		{polyEncMHA(seq, dim, heads, heads, hd, seq, dt, Qa, Ka, Va, Oa), a},
		{polyEncMHA(seq, dim, heads, heads, hd, seq, dt, Qb, Kb, Vb, Ob), b},
	}, x, tgt, []int32{int32(batch), int32(seq), int32(dim)}, []int32{int32(batch), int32(seq), int32(dim)}, nil
}

func polyEncRNN(kind int32, in, hid, seq int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, kind, int32(in), int32(hid), int32(seq), int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeRNNPair(dt core.DType, batch, seq, dim int, isLSTM bool) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	if isLSTM {
		cfg := lstm.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq}
		nw := cfg.WeightCount()
		Wa, Wb := seedPat(nw, 0x51001), seedPat(nw, 0x51002)
		a, err := lstm.NewConfigured(cfg, dt, quant.FormatNone, Wa)
		if err != nil {
			return nil, nil, nil, nil, nil, err
		}
		b, err := lstm.NewConfigured(cfg, dt, quant.FormatNone, Wb)
		if err != nil {
			return nil, nil, nil, nil, nil, err
		}
		x, tgt := seedPat(batch*seq*dim, 0x51010), seedPat(batch*seq*dim, 0x51020)
		return []polyBranchEnc{{polyEncRNN(skLSTM, dim, dim, seq, dt, Wa), a}, {polyEncRNN(skLSTM, dim, dim, seq, dt, Wb), b}},
			x, tgt, []int32{int32(batch), int32(seq), int32(dim)}, []int32{int32(batch), int32(seq), int32(dim)}, nil
	}
	cfg := rnn.Config{InputSize: dim, HiddenSize: dim, SeqLen: seq}
	nw := cfg.WeightCount()
	Wa, Wb := seedPat(nw, 0x61001), seedPat(nw, 0x61002)
	a, err := rnn.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := rnn.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*seq*dim, 0x61010), seedPat(batch*seq*dim, 0x61020)
	return []polyBranchEnc{{polyEncRNN(skRNN, dim, dim, seq, dt, Wa), a}, {polyEncRNN(skRNN, dim, dim, seq, dt, Wb), b}},
		x, tgt, []int32{int32(batch), int32(seq), int32(dim)}, []int32{int32(batch), int32(seq), int32(dim)}, nil
}

func polyEncEmbed(vocab, emb, seq int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skEmbedding, int32(vocab), int32(emb), int32(seq), -1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeEmbedPair(dt core.DType, batch, seq, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	vocab := 16
	cfg := embedding.Config{VocabSize: vocab, EmbeddingDim: dim, SeqLen: seq}
	Wa, Wb := seedPat(vocab*dim, 0x81001), seedPat(vocab*dim, 0x81002)
	a, err := embedding.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := embedding.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x := make([]float32, batch*seq)
	for i := range x {
		x[i] = float32(i % vocab)
	}
	tgt := seedPat(batch*seq*dim, 0x81020)
	return []polyBranchEnc{{polyEncEmbed(vocab, dim, seq, dt, Wa), a}, {polyEncEmbed(vocab, dim, seq, dt, Wb), b}},
		x, tgt, []int32{int32(batch), int32(seq)}, []int32{int32(batch), int32(seq), int32(dim)}, nil
}

func polyEncCNN1(inCh, filters, seq, kernel, stride, pad int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skCNN1, int32(inCh), int32(filters), int32(seq), int32(kernel), int32(stride), int32(pad), -1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeCNN1Pair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := cnn1.Config{InChannels: 4, Filters: 4, SeqLen: 8, Kernel: 3, Padding: 1, Stride: 1, Activation: core.ActivationLinear}
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel
	Wa, Wb := seedPat(nw, 0x9101), seedPat(nw, 0x9102)
	a, err := cnn1.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := cnn1.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	outLen := cfg.OutLen()
	x, tgt := seedPat(batch*4*8, 0x9110), seedPat(batch*4*outLen, 0x9120)
	return []polyBranchEnc{
		{polyEncCNN1(4, 4, 8, 3, 1, 1, dt, Wa), a},
		{polyEncCNN1(4, 4, 8, 3, 1, 1, dt, Wb), b},
	}, x, tgt, []int32{int32(batch), 4, 8}, []int32{int32(batch), 4, int32(outLen)}, nil
}

func makeStackPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	Wa, Wb := seedPat(dim*dim, 0xB1001), seedPat(dim*dim, 0xB1002)
	da, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	db, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	enc := func(W []float32) []byte {
		var b bytes.Buffer
		writeI32(&b, skNestedStack, 1)
		b.Write(polyEncDense(dim, dim, dt, W))
		return b.Bytes()
	}
	x, tgt := seedPat(batch*dim, 0xB1010), seedPat(batch*dim, 0xB1020)
	return []polyBranchEnc{
		{enc(Wa), &parallel.Stack{Children: []any{da}}},
		{enc(Wb), &parallel.Stack{Children: []any{db}}},
	}, x, tgt, []int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func polyEncMamba(dModel, dState, expand, seq int, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skMamba, int32(dModel), int32(dState), int32(expand), int32(seq), int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeMambaPair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := mamba.Config{DModel: 6, DState: 4, Expand: 2, SeqLen: 3}
	nw := mambaWeightCount(cfg)
	Wa, Wb := seedPat(nw, 0xA1001), seedPat(nw, 0xA1002)
	inA, outA, aLogA, dSkipA, err := splitMambaPacked(cfg, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	inB, outB, aLogB, dSkipB, err := splitMambaPacked(cfg, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	a, err := mamba.NewConfigured(cfg, dt, quant.FormatNone, inA, outA, aLogA, dSkipA)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := mamba.NewConfigured(cfg, dt, quant.FormatNone, inB, outB, aLogB, dSkipB)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*cfg.SeqLen*cfg.DModel, 0xA1010), seedPat(batch*cfg.SeqLen*cfg.DModel, 0xA1020)
	return []polyBranchEnc{
		{polyEncMamba(cfg.DModel, cfg.DState, cfg.Expand, cfg.SeqLen, dt, Wa), a},
		{polyEncMamba(cfg.DModel, cfg.DState, cfg.Expand, cfg.SeqLen, dt, Wb), b},
	}, x, tgt, []int32{int32(batch), int32(cfg.SeqLen), int32(cfg.DModel)},
		[]int32{int32(batch), int32(cfg.SeqLen), int32(cfg.DModel)}, nil
}

func polyEncGDN(cfg gdn.Config, seq int, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skGDN, int32(cfg.HiddenSize), int32(seq), int32(cfg.NumKeyHeads), int32(cfg.NumValueHeads),
		int32(cfg.KeyHeadDim), int32(cfg.ValueHeadDim), int32(cfg.ConvKernel), 1)
	writeF32s(&b, W)
	return b.Bytes()
}

func makeGDNPair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	_ = dt // GDN f32-only
	cfg := smokeGdCfg()
	seq := 4
	nw := gdnWeightCount(cfg)
	Wa, Wb := seedPat(nw, 0xC1001), seedPat(nw, 0xC1002)
	split := func(W []float32) (*gdn.Layer, error) {
		inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(cfg, W)
		if err != nil {
			return nil, err
		}
		return gdn.NewConfigured(cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	}
	a, err := split(Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := split(Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*seq*cfg.HiddenSize, 0xC1010), seedPat(batch*seq*cfg.HiddenSize, 0xC1020)
	return []polyBranchEnc{
		{polyEncGDN(cfg, seq, Wa), a},
		{polyEncGDN(cfg, seq, Wb), b},
	}, x, tgt, []int32{int32(batch), int32(seq), int32(cfg.HiddenSize)},
		[]int32{int32(batch), int32(seq), int32(cfg.HiddenSize)}, nil
}

func polyEncCNN2(cfg cnn2.Config, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skCNN2, int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height), int32(cfg.Width),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), -1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeCNN2Pair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := preservingCNN2()
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel
	Wa, Wb := seedPat(nw, 0xD2001), seedPat(nw, 0xD2002)
	a, err := cnn2.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := cnn2.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	xN := batch * cfg.InChannels * cfg.Height * cfg.Width
	tN := batch * cfg.Filters * cfg.OutH() * cfg.OutW()
	x, tgt := seedPat(xN, 0xD2010), seedPat(tN, 0xD2020)
	return []polyBranchEnc{
		{polyEncCNN2(cfg, dt, Wa), a},
		{polyEncCNN2(cfg, dt, Wb), b},
	}, x, tgt, []int32{int32(batch), int32(cfg.InChannels), int32(cfg.Height), int32(cfg.Width)},
		[]int32{int32(batch), int32(cfg.Filters), int32(cfg.OutH()), int32(cfg.OutW())}, nil
}

func polyEncCNN3(cfg cnn3.Config, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skCNN3, int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth), int32(cfg.Height),
		int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), -1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeCNN3Pair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := preservingCNN3()
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel * cfg.Kernel
	Wa, Wb := seedPat(nw, 0xD3001), seedPat(nw, 0xD3002)
	a, err := cnn3.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := cnn3.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	xN := batch * cfg.InChannels * cfg.Depth * cfg.Height * cfg.Width
	tN := batch * cfg.Filters * cfg.OutD() * cfg.OutH() * cfg.OutW()
	x, tgt := seedPat(xN, 0xD3010), seedPat(tN, 0xD3020)
	return []polyBranchEnc{
		{polyEncCNN3(cfg, dt, Wa), a},
		{polyEncCNN3(cfg, dt, Wb), b},
	}, x, tgt,
		[]int32{int32(batch), int32(cfg.InChannels), int32(cfg.Depth), int32(cfg.Height), int32(cfg.Width)},
		[]int32{int32(batch), int32(cfg.Filters), int32(cfg.OutD()), int32(cfg.OutH()), int32(cfg.OutW())}, nil
}

func polyEncConvT1(cfg convt1.Config, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skConvT1, int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.SeqLen), int32(cfg.Kernel),
		int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding), -1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeConvT1Pair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := preservingConvT1()
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel
	Wa, Wb := seedPat(nw, 0xE1001), seedPat(nw, 0xE1002)
	a, err := convt1.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := convt1.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	xN := batch * cfg.InChannels * cfg.SeqLen
	tN := batch * cfg.Filters * cfg.OutLen()
	x, tgt := seedPat(xN, 0xE1010), seedPat(tN, 0xE1020)
	return []polyBranchEnc{
		{polyEncConvT1(cfg, dt, Wa), a},
		{polyEncConvT1(cfg, dt, Wb), b},
	}, x, tgt, []int32{int32(batch), int32(cfg.InChannels), int32(cfg.SeqLen)},
		[]int32{int32(batch), int32(cfg.Filters), int32(cfg.OutLen())}, nil
}

func polyEncConvT2(cfg convt2.Config, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skConvT2, int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Height), int32(cfg.Width),
		int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding), -1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeConvT2Pair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := preservingConvT2()
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel
	Wa, Wb := seedPat(nw, 0xE2001), seedPat(nw, 0xE2002)
	a, err := convt2.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := convt2.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	xN := batch * cfg.InChannels * cfg.Height * cfg.Width
	tN := batch * cfg.Filters * cfg.OutH() * cfg.OutW()
	x, tgt := seedPat(xN, 0xE2010), seedPat(tN, 0xE2020)
	return []polyBranchEnc{
		{polyEncConvT2(cfg, dt, Wa), a},
		{polyEncConvT2(cfg, dt, Wb), b},
	}, x, tgt, []int32{int32(batch), int32(cfg.InChannels), int32(cfg.Height), int32(cfg.Width)},
		[]int32{int32(batch), int32(cfg.Filters), int32(cfg.OutH()), int32(cfg.OutW())}, nil
}

func polyEncConvT3(cfg convt3.Config, dt core.DType, W []float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skConvT3, int32(cfg.InChannels), int32(cfg.Filters), int32(cfg.Depth), int32(cfg.Height),
		int32(cfg.Width), int32(cfg.Kernel), int32(cfg.Stride), int32(cfg.Padding), int32(cfg.OutputPadding),
		-1, int32(dt))
	writeF32s(&b, W)
	return b.Bytes()
}

func makeConvT3Pair(dt core.DType, batch int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	cfg := preservingConvT3()
	nw := cfg.Filters * cfg.InChannels * cfg.Kernel * cfg.Kernel * cfg.Kernel
	Wa, Wb := seedPat(nw, 0xE3001), seedPat(nw, 0xE3002)
	a, err := convt3.NewConfigured(cfg, dt, quant.FormatNone, Wa)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	b, err := convt3.NewConfigured(cfg, dt, quant.FormatNone, Wb)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	xN := batch * cfg.InChannels * cfg.Depth * cfg.Height * cfg.Width
	tN := batch * cfg.Filters * cfg.OutD() * cfg.OutH() * cfg.OutW()
	x, tgt := seedPat(xN, 0xE3010), seedPat(tN, 0xE3020)
	return []polyBranchEnc{
		{polyEncConvT3(cfg, dt, Wa), a},
		{polyEncConvT3(cfg, dt, Wb), b},
	}, x, tgt,
		[]int32{int32(batch), int32(cfg.InChannels), int32(cfg.Depth), int32(cfg.Height), int32(cfg.Width)},
		[]int32{int32(batch), int32(cfg.Filters), int32(cfg.OutD()), int32(cfg.OutH()), int32(cfg.OutW())}, nil
}

func polyEncParallelNest(dim, outFeat, branches int, combine parallel.CombineMode, dt core.DType, Ws [][]float32) []byte {
	var b bytes.Buffer
	writeI32(&b, skParallel, parallelCombineCode(combine), int32(dim), int32(outFeat), 0, 0)
	writeF64(&b, 0)
	writeI32(&b, int32(branches))
	for _, W := range Ws {
		b.Write(polyEncDense(dim, outFeat, dt, W))
	}
	writeI32(&b, 0) // has_gate
	return b.Bytes()
}

func makeNestedParallelPair(dt core.DType, batch, dim int) ([]polyBranchEnc, []float32, []float32, []int32, []int32, error) {
	mkInner := func(seed uint32) (*parallel.Layer, []byte, error) {
		Wa, Wb := seedPat(dim*dim, seed), seedPat(dim*dim, seed^0x1111)
		a, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, quant.FormatNone, Wa)
		if err != nil {
			return nil, nil, err
		}
		b, err := dense.NewConfigured(dim, dim, core.ActivationLinear, dt, quant.FormatNone, Wb)
		if err != nil {
			return nil, nil, err
		}
		l, err := parallel.NewFromBranches(parallel.Config{
			Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
		}, []any{a, b}, nil)
		if err != nil {
			return nil, nil, err
		}
		return l, polyEncParallelNest(dim, dim, 2, parallel.CombineAdd, dt, [][]float32{Wa, Wb}), nil
	}
	la, ea, err := mkInner(0xF1001)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	lb, eb, err := mkInner(0xF2001)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	x, tgt := seedPat(batch*dim, 0xF1010), seedPat(batch*dim, 0xF1020)
	return []polyBranchEnc{{ea, la}, {eb, lb}}, x, tgt,
		[]int32{int32(batch), int32(dim)}, []int32{int32(batch), int32(dim)}, nil
}

func encodePPB1(pc polyCase, brs []polyBranchEnc, x, tgt []float32, xShape, tShape []int32, lr float64) []byte {
	var in bytes.Buffer
	_, _ = in.Write([]byte("PPB1"))
	writeI32(&in, 1, int32(pc.cfgDim), int32(pc.cfgOutFeat), int32(pc.cfgSeq), int32(len(brs)))
	for _, br := range brs {
		in.Write(br.bytes)
	}
	writeI32(&in, int32(len(xShape)))
	writeI32(&in, xShape...)
	writeF32s(&in, x)
	writeI32(&in, int32(len(tShape)))
	writeI32(&in, tShape...)
	writeF32s(&in, tgt)
	writeF64(&in, lr)
	return in.Bytes()
}

func ints(s []int32) []int {
	o := make([]int, len(s))
	for i, v := range s {
		o[i] = int(v)
	}
	return o
}

func runGoPoly(pc polyCase, brs []polyBranchEnc, x, tgt []float32, xShape, tShape []int32, lr float64) (cameralModesOut, error) {
	var z cameralModesOut
	ops := make([]any, len(brs))
	for i, br := range brs {
		ops[i] = br.op
	}
	cfg := parallel.Config{
		Dim: pc.cfgDim, OutFeat: pc.cfgOutFeat, Branches: len(ops),
		Combine: parallel.CombineAdd, SeqLen: pc.cfgSeq,
	}
	l, err := parallel.NewFromBranches(cfg, ops, nil)
	if err != nil {
		return z, err
	}
	l.Exec.Backend = core.BackendCPUTiled
	l.Exec.MultiCore = false
	xin := core.NewTensor[float32](ints(xShape)...)
	copy(xin.Data, x)
	target := core.NewTensor[float32](ints(tShape)...)
	copy(target.Data, tgt)
	loss, err := parallel.TrainMSE(l, xin, target, parallel.ModeNormalBP, lr)
	if err != nil {
		return z, err
	}
	z.loss = loss
	z.WAfter = make([][]float32, len(l.Branches))
	for i, op := range l.Branches {
		flat, err := packPolyBranchW(op)
		if err != nil {
			return z, err
		}
		z.WAfter[i] = flat
	}
	return z, nil
}

// packPolyBranchW matches C++ branch_export_f32 / GradWSize loom layout.
func packPolyBranchW(op any) ([]float32, error) {
	switch v := op.(type) {
	case *mamba.Layer:
		return packMambaWeightsF32(v)
	case *lstm.Layer:
		return packLSTMWeightsF32(v)
	default:
		return dna.FlattenOp(op)
	}
}

func runPolyKind(pc polyCase, dt core.DType, exact bool, benchIters int) error {
	const lr = 1e-2
	brs, x, tgt, xShape, tShape, err := pc.make(dt)
	if err != nil {
		return fmt.Errorf("build: %w", err)
	}
	goO, err := runGoPoly(pc, brs, x, tgt, xShape, tShape, lr)
	if err != nil {
		return fmt.Errorf("go: %w", err)
	}
	payload := encodePPB1(pc, brs, x, tgt, xShape, tShape, lr)
	if benchIters > 0 {
		if _, _, err := runCppCameralModes(payload, "PPBO", benchIters); err != nil {
			return fmt.Errorf("cpp-bench: %w", err)
		}
	}
	cppO, _, err := runCppCameralModes(payload, "PPBO", 0)
	if err != nil {
		return fmt.Errorf("cpp: %w", err)
	}
	return compareCameralModes(goO, cppO, exact)
}

func runPolyKindSpectrum(pc polyCase) error {
	fmt.Printf("\n      [Parallel poly %s ×34]\n", pc.name)
	fmt.Printf("      %-12s %-12s %8s %10s %10s %9s\n", "dtype", "band", "exact%W", "goµs", "cppµs", "spd%Δ")
	var fails []string
	const benchIters = 10
	for _, dt := range core.AllDTypes {
		err := runPolyKind(pc, dt, true, 0)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s:%v", dt, err))
			fmt.Printf("      %-12s %-12s %7s %10s %10s %8s  %v\n", dt, suites.DetBroken, "—",
				"—", "—", "—", err)
			continue
		}
		brs, x, tgt, xShape, tShape, _ := pc.make(dt)
		payload := encodePPB1(pc, brs, x, tgt, xShape, tShape, 1e-2)
		_, cppDur, err := runCppCameralModes(payload, "PPBO", benchIters)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s bench:%v", dt, err))
			continue
		}
		goUS, berr := benchGoUS(func() error {
			return runPolyKind(pc, dt, true, 0)
		}, benchIters)
		if berr != nil {
			fails = append(fails, fmt.Sprintf("%s go-bench:%v", dt, berr))
			continue
		}
		cppUS := perIterUS(cppDur, benchIters)
		spd := suites.SpeedPctDiff(goUS, cppUS)
		fmt.Printf("      %-12s %-12s %7s %10s %10s %8.1f%%\n", dt, suites.DetExactBit, "100%", FormatUS(goUS), FormatUS(cppUS), spd)
	}
	if len(fails) > 0 {
		n := len(fails)
		if n > 6 {
			n = 6
		}
		return fmt.Errorf("%d dtype fails: %s", len(fails), strings.Join(fails[:n], "; "))
	}
	return nil
}

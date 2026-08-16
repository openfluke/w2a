package entity_test

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/model/entity"
	"github.com/openfluke/welvet/quant"
)

func fillDense(d *dense.Layer, v float32) {
	if d == nil || d.Weights == nil {
		return
	}
	f, err := d.Weights.FlattenF32()
	if err != nil {
		return
	}
	for i := range f {
		f[i] = v
	}
	_ = d.Weights.SetFromF32(f)
}

func fillOp(op any, v float32) {
	switch x := op.(type) {
	case *dense.Layer:
		fillDense(x, v)
	case *parallel.Layer:
		if x == nil {
			return
		}
		for _, b := range x.Branches {
			fillOp(b, v)
		}
		fillDense(x.Gate, v)
	case *parallel.Stack:
		if x == nil {
			return
		}
		for _, c := range x.Children {
			fillOp(c, v)
		}
	case *metacognition.Layer:
		if x != nil {
			fillDense(x.Observed, v)
		}
	case *cnn1.Layer:
		if x != nil {
			fillDense(x.Proj, v)
		}
	}
}

func denseFlat(t *testing.T, d *dense.Layer) []float32 {
	t.Helper()
	if d == nil || d.Weights == nil {
		t.Fatal("nil dense")
	}
	f, err := d.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(f))
	copy(out, f)
	return out
}

func sameF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func fwd(t *testing.T, s *parallel.Stack, x *core.Tensor[float32]) []float32 {
	t.Helper()
	_, post, err := parallel.ForwardStack(s, x)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(post.Data))
	copy(out, post.Data)
	return out
}

func TestCameralBicameralRoundTrip(t *testing.T) {
	const in, hidden, out = 8, 6, 4
	s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	fillOp(s, 0.07)
	path := filepath.Join(t.TempDir(), "bi.entity")
	if err := entity.WriteCameralFile(path, s, parallel.ModeTweenSplitFastProxy); err != nil {
		t.Fatal(err)
	}
	if !entity.IsEntity(path) {
		t.Fatal("IsEntity false")
	}
	info, err := entity.Inspect(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Cameral == nil || info.Cameral.Kind != "bicameral" {
		t.Fatalf("cameral=%+v", info.Cameral)
	}
	if info.Cameral.TrainMode != "TweenSplitFastProxy" {
		t.Fatalf("train_mode=%q", info.Cameral.TrainMode)
	}
	if info.Cameral.In != in || info.Cameral.Hidden != hidden || info.Cameral.Out != out {
		t.Fatalf("dims %+v", info.Cameral)
	}

	got, mode, err := entity.LoadCameral(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode != parallel.ModeTweenSplitFastProxy {
		t.Fatalf("mode=%s", mode)
	}
	x := core.NewTensor[float32](1, in)
	for i := range x.Data {
		x.Data[i] = 0.2 + float32(i)*0.01
	}
	if !sameF32(fwd(t, s, x), fwd(t, got, x)) {
		t.Fatal("forward mismatch after cameral round-trip")
	}
	stem0 := s.Children[0].(*dense.Layer)
	stem1 := got.Children[0].(*dense.Layer)
	if !sameF32(denseFlat(t, stem0), denseFlat(t, stem1)) {
		t.Fatal("stem weights mismatch")
	}
}

func TestCameralBranchModesPersist(t *testing.T) {
	s, err := parallel.Bicameral(8, 8, 8, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	hemi := s.Children[1].(*parallel.Layer)
	hemi.SetBranchModes(parallel.ModeTweenSplitFastProxy, parallel.ModeStepBP)
	s.SetChildModes(parallel.ModeStepTween, parallel.ModeInherit, parallel.ModeTweenSplitSparse)
	s.AltTimes = 3
	path := filepath.Join(t.TempDir(), "modes.entity")
	if err := entity.WriteCameralFile(path, s, parallel.ModeStepBP); err != nil {
		t.Fatal(err)
	}
	got, mode, err := entity.LoadCameral(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode != parallel.ModeStepBP {
		t.Fatalf("parent mode %s", mode)
	}
	if got.AltTimes != 3 {
		t.Fatalf("AltTimes %d", got.AltTimes)
	}
	if len(got.ChildModes) != 3 ||
		got.ChildModes[0] != parallel.ModeStepTween ||
		got.ChildModes[2] != parallel.ModeTweenSplitSparse {
		t.Fatalf("ChildModes %v", got.ChildModes)
	}
	gh := got.Children[1].(*parallel.Layer)
	if len(gh.BranchModes) != 2 ||
		gh.BranchModes[0] != parallel.ModeTweenSplitFastProxy ||
		gh.BranchModes[1] != parallel.ModeStepBP {
		t.Fatalf("BranchModes %v", gh.BranchModes)
	}
}

func TestCameralTrainAfterLoad(t *testing.T) {
	const in, hidden, out = 8, 6, 8
	s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	fillOp(s, 0.05)
	path := filepath.Join(t.TempDir(), "train.entity")
	if err := entity.WriteCameralFile(path, s, parallel.ModeTweenSplitFastProxy); err != nil {
		t.Fatal(err)
	}
	got, mode, err := entity.LoadCameral(path)
	if err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.3
		y.Data[i] = 0.9
	}
	before := denseFlat(t, got.Children[0].(*dense.Layer))
	if _, err := parallel.TrainStackMSE(got, x, y, mode, 0.05); err != nil {
		t.Fatal(err)
	}
	after := denseFlat(t, got.Children[0].(*dense.Layer))
	moved := false
	for i := range before {
		if before[i] != after[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatal("FastProxy load did not update stem")
	}
}

func TestCameralMetaSandwich(t *testing.T) {
	const dim = 8
	stem, err := dense.NewConfigured[float32](dim, dim, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	mid, err := metacognition.New(metacognition.Config{Dim: dim})
	if err != nil {
		t.Fatal(err)
	}
	head, err := dense.NewConfigured[float32](dim, dim, core.ActivationSigmoid, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	s, err := parallel.Sandwich(stem, mid, head)
	if err != nil {
		t.Fatal(err)
	}
	fillOp(s, 0.04)
	path := filepath.Join(t.TempDir(), "meta.entity")
	if err := entity.WriteCameralFile(path, s, parallel.ModeTweenSplitLinear); err != nil {
		t.Fatal(err)
	}
	got, mode, err := entity.LoadCameral(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode != parallel.ModeTweenSplitLinear {
		t.Fatalf("mode=%s", mode)
	}
	x := core.NewTensor[float32](1, dim)
	y := core.NewTensor[float32](1, dim)
	for i := range x.Data {
		x.Data[i] = 0.2
		y.Data[i] = 0.8
	}
	if _, err := parallel.TrainStackMSE(got, x, y, mode, 0.05); err != nil {
		t.Fatal(err)
	}
}

func TestCameralWrappedCNN(t *testing.T) {
	const in, hidden, out = 16, 32, 1
	stem, err := dense.NewConfigured[float32](in, hidden, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	cnn, err := cnn1.New(cnn1.Config{
		InChannels: 1, Filters: 1, SeqLen: hidden, Kernel: 3, Stride: 1, Padding: 1,
		Activation: core.ActivationReLU,
	})
	if err != nil {
		t.Fatal(err)
	}
	inV, err := parallel.NewView(1, 1, hidden)
	if err != nil {
		t.Fatal(err)
	}
	outV, err := parallel.NewView(1, hidden)
	if err != nil {
		t.Fatal(err)
	}
	mid, err := parallel.NewStack(inV, cnn, outV)
	if err != nil {
		t.Fatal(err)
	}
	head, err := dense.NewConfigured[float32](hidden, out, core.ActivationSigmoid, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	s, err := parallel.Sandwich(stem, mid, head)
	if err != nil {
		t.Fatal(err)
	}
	fillOp(s, 0.05)
	path := filepath.Join(t.TempDir(), "cnn.entity")
	if err := entity.WriteCameralFile(path, s, parallel.ModeTweenSplitFastProxy); err != nil {
		t.Fatal(err)
	}
	got, mode, err := entity.LoadCameral(path)
	if err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](1, in)
	for i := range x.Data {
		x.Data[i] = 0.15
	}
	want := fwd(t, s, x)
	have := fwd(t, got, x)
	if len(want) != len(have) {
		t.Fatalf("fwd len %d vs %d", len(want), len(have))
	}
	for i := range want {
		if math.Abs(float64(want[i]-have[i])) > 1e-6 {
			t.Fatalf("fwd[%d] %v vs %v", i, want[i], have[i])
		}
	}
	y := core.NewTensor[float32](1, out)
	y.Data[0] = 1
	if _, err := parallel.TrainStackMSE(got, x, y, mode, 0.05); err != nil {
		t.Fatal(err)
	}
}

func TestParseTrainModeAliases(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want parallel.TrainMode
	}{
		{"TweenSplitFastProxy", parallel.ModeTweenSplitFastProxy},
		{"fastproxy", parallel.ModeTweenSplitFastProxy},
		{"sparse", parallel.ModeTweenSplitSparse},
		{"sgd", parallel.ModeNormalBP},
		{"", parallel.ModeInherit},
	} {
		got, err := parallel.ParseTrainMode(tc.in)
		if err != nil {
			t.Fatalf("%q: %v", tc.in, err)
		}
		if got != tc.want {
			t.Fatalf("%q → %s want %s", tc.in, got, tc.want)
		}
	}
}

func TestLoadCameralRejectsTransformer(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lm.entity")
	spec := &entity.TransformerSpec{Architecture: "toy", HiddenSize: 4, VocabSize: 8, Engine: "welvet"}
	if err := entity.WriteTransformerFile(path, spec, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := entity.LoadCameral(path); err == nil {
		t.Fatal("expected error")
	}
}

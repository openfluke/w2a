package parallel_test

import (
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func fillOpWeights(op any, v float32) {
	switch x := op.(type) {
	case *dense.Layer:
		if x == nil || x.Weights == nil {
			return
		}
		f, err := x.Weights.FlattenF32()
		if err != nil {
			return
		}
		for i := range f {
			f[i] = v
		}
		_ = x.Weights.SetFromF32(f)
	case *parallel.Layer:
		if x == nil {
			return
		}
		for _, b := range x.Branches {
			fillOpWeights(b, v)
		}
	case *cnn1.Layer:
		if x != nil {
			fillOpWeights(x.Proj, v)
		}
	case *sequential.Layer:
		if x == nil {
			return
		}
		for _, c := range x.ChildOps() {
			fillOpWeights(c, v)
		}
	case *residual.Layer:
		if x == nil {
			return
		}
		for _, c := range x.ChildOps() {
			fillOpWeights(c, v)
		}
	case *parallel.ResidualSkip:
		if x != nil {
			fillOpWeights(x.F, v)
		}
	case *parallel.Stack:
		if x == nil {
			return
		}
		for _, c := range x.Children {
			fillOpWeights(c, v)
		}
	case *swiglu.Layer:
		if x == nil {
			return
		}
		fillOpWeights(x.Gate, v)
		fillOpWeights(x.Up, v)
		fillOpWeights(x.Down, v)
	}
}

func denseSnap(t *testing.T, d *dense.Layer) []float32 {
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

func moved(a, b []float32) bool {
	if len(a) != len(b) {
		return true
	}
	for i := range a {
		if a[i] != b[i] {
			return true
		}
	}
	return false
}

func TestTweenSplitUpdatesWholeSandwich(t *testing.T) {
	const in, hidden, out = 8, 6, 8
	s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	stem := s.Children[0].(*dense.Layer)
	head := s.Children[2].(*dense.Layer)
	beforeStem := denseSnap(t, stem)
	beforeHead := denseSnap(t, head)
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
		y.Data[i] = 0.8
	}
	loss, err := parallel.TrainStackMSE(s, x, y, parallel.ModeStepTweenSplit, 0.05)
	if err != nil {
		t.Fatalf("TrainStackMSE: %v", err)
	}
	if loss != loss {
		t.Fatal("NaN loss")
	}
	if !moved(beforeStem, denseSnap(t, stem)) {
		t.Fatal("stem weights did not move")
	}
	if !moved(beforeHead, denseSnap(t, head)) {
		t.Fatal("head weights did not move")
	}
}

func TestTweenSplitWeightedModesMove(t *testing.T) {
	const in, hidden, out = 8, 6, 8
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
		y.Data[i] = 0.8
	}
	modes := []parallel.TrainMode{
		parallel.ModeTweenSplitHeadProxy, parallel.ModeTweenSplitLinear,
		parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitLinearCache,
		parallel.ModeTweenSplitHeadProxyAsync, parallel.ModeTweenSplitSparse,
	}
	for _, mode := range modes {
		s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			t.Fatal(err)
		}
		fillOpWeights(s, 0.05)
		stem := s.Children[0].(*dense.Layer)
		before := denseSnap(t, stem)
		if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		if !moved(before, denseSnap(t, stem)) {
			t.Fatalf("%s did not move stem", mode)
		}
	}
}

func TestSplitTapeScoresThenTrains(t *testing.T) {
	const in, hidden, out = 8, 6, 8
	s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	stem := s.Children[0].(*dense.Layer)
	before := denseSnap(t, stem)
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
		y.Data[i] = 0.8
	}
	tape, err := parallel.OpenSplitTape(s, x)
	if err != nil {
		t.Fatal(err)
	}
	if tape.Post == nil || tape.Post.Len() != out {
		t.Fatalf("tape post %v", tape.Post)
	}
	if _, err := tape.Train(y, parallel.ModeTweenSplitFastProxy, 0.05); err != nil {
		t.Fatal(err)
	}
	if !moved(before, denseSnap(t, stem)) {
		t.Fatal("OpenSplitTape.Train did not move stem")
	}
}

func TestLinearCacheAndAsyncSecondStep(t *testing.T) {
	const in, hidden, out = 8, 6, 8
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
		y.Data[i] = 0.8
	}
	for _, mode := range []parallel.TrainMode{parallel.ModeTweenSplitLinearCache, parallel.ModeTweenSplitHeadProxyAsync} {
		s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			t.Fatal(err)
		}
		fillOpWeights(s, 0.05)
		stem := s.Children[0].(*dense.Layer)
		if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
			t.Fatal(err)
		}
		mid := denseSnap(t, stem)
		if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
			t.Fatal(err)
		}
		if !moved(mid, denseSnap(t, stem)) {
			t.Fatalf("%s second step did not move stem", mode)
		}
	}
}

func TestSplitFamilyModes(t *testing.T) {
	for _, m := range []parallel.TrainMode{
		parallel.ModeTweenSplit, parallel.ModeStepTweenSplit,
		parallel.ModeTweenSplitHeadProxy, parallel.ModeTweenSplitLinear,
		parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitLinearCache,
		parallel.ModeTweenSplitHeadProxyAsync, parallel.ModeTweenSplitSparse,
		parallel.ModeMeshTweenSplit, parallel.ModeMeshTweenSplitFastProxy,
		parallel.ModeMeshTweenSplitSparse,
	} {
		if !m.IsSplitFamily() {
			t.Fatalf("%s should be split family", m)
		}
	}
	if parallel.ModeTweenAlt.IsSplitFamily() {
		t.Fatal("Alt is not split family")
	}
}

func TestTweenAltUpdatesWholeSandwich(t *testing.T) {
	const in, hidden, out = 8, 6, 8
	s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	stem := s.Children[0].(*dense.Layer)
	head := s.Children[2].(*dense.Layer)
	beforeStem := denseSnap(t, stem)
	beforeHead := denseSnap(t, head)
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
		y.Data[i] = 0.8
	}
	s.AltTimes = 2
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeTweenAlt, 0.05); err != nil {
		t.Fatalf("TrainStackMSE: %v", err)
	}
	if !moved(beforeStem, denseSnap(t, stem)) {
		t.Fatal("stem weights did not move")
	}
	if !moved(beforeHead, denseSnap(t, head)) {
		t.Fatal("head weights did not move")
	}
}

func TestTweenAltFamily(t *testing.T) {
	if parallel.ModeTweenAlt.Family() != parallel.ModeStepTweenAlt.Family() {
		t.Fatal("TweenAlt and StepTweenAlt should share a family")
	}
	if parallel.ModeTweenAlt.Family() == parallel.ModeTweenSplit.Family() {
		t.Fatal("TweenAlt family must differ from TweenSplit")
	}
}

func TestLinearMetacognitionSandwichNoPanic(t *testing.T) {
	const in, hidden, out = 8, 8, 8
	stem, err := dense.NewConfigured[float32](in, hidden, core.ActivationLeakyReLU,
		core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	mid, err := metacognition.New(metacognition.Config{Dim: hidden})
	if err != nil {
		t.Fatal(err)
	}
	head, err := dense.NewConfigured[float32](hidden, out, core.ActivationSigmoid,
		core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	s, err := parallel.Sandwich(stem, mid, head)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
		y.Data[i] = 0.8
	}
	for _, mode := range []parallel.TrainMode{
		parallel.ModeTweenSplitLinear, parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitSparse,
	} {
		if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
	}
}

func TestTweenTrainsWrappedCNN(t *testing.T) {
	const in, hidden, out = 16, 32, 1
	stem, err := dense.NewConfigured[float32](in, hidden, core.ActivationLeakyReLU,
		core.DTypeFloat32, quant.FormatNone, nil)
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
	head, err := dense.NewConfigured[float32](hidden, out, core.ActivationSigmoid,
		core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	s, err := parallel.Sandwich(stem, mid, head)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
	}
	y.Data[0] = 1
	for _, mode := range []parallel.TrainMode{
		parallel.ModeStepTween, parallel.ModeStepTweenSplit,
		parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitLinear,
	} {
		if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
	}
}

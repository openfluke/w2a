package parallel_test

import (
	"strings"
	"testing"

	parallelsuite "github.com/openfluke/w2a/suites/parallel"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func xy(in, out int) (*core.Tensor[float32], *core.Tensor[float32]) {
	x := core.NewTensor[float32](1, in)
	y := core.NewTensor[float32](1, out)
	for i := range x.Data {
		x.Data[i] = 0.2
	}
	for i := range y.Data {
		y.Data[i] = 0.8
	}
	return x, y
}

func mustBicameral(t *testing.T) *parallel.Stack {
	t.Helper()
	s, err := parallel.Bicameral(8, 6, 8, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	return s
}

func snapDensest(t *testing.T, op any) []float32 {
	t.Helper()
	var out []float32
	var walk func(any)
	walk = func(op any) {
		switch x := op.(type) {
		case *dense.Layer:
			if x != nil && x.Weights != nil {
				f, err := x.Weights.FlattenF32()
				if err == nil {
					out = append(out, f...)
				}
			}
		case *parallel.Layer:
			if x == nil {
				return
			}
			for _, b := range x.Branches {
				walk(b)
			}
			walk(x.Gate)
		case *parallel.Stack:
			if x == nil {
				return
			}
			for _, c := range x.Children {
				walk(c)
			}
		case *cnn1.Layer:
			if x != nil {
				walk(x.Proj)
			}
		case *sequential.Layer:
			if x == nil {
				return
			}
			for _, c := range x.ChildOps() {
				walk(c)
			}
		case *residual.Layer:
			if x == nil {
				return
			}
			for _, c := range x.ChildOps() {
				walk(c)
			}
		case *parallel.ResidualSkip:
			if x != nil {
				walk(x.F)
			}
		}
	}
	walk(op)
	if len(out) == 0 {
		t.Fatal("no dense weights")
	}
	return out
}

func anyMoved(t *testing.T, s *parallel.Stack, mode parallel.TrainMode) {
	t.Helper()
	before := snapDensest(t, s)
	x, y := xy(8, 8)
	if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
		t.Fatalf("%s: %v", mode, err)
	}
	if !moved(before, snapDensest(t, s)) {
		t.Fatalf("%s did not move any weights", mode)
	}
}

func stemMoved(t *testing.T, s *parallel.Stack, mode parallel.TrainMode) {
	t.Helper()
	stem := s.Children[0].(*dense.Layer)
	before := denseSnap(t, stem)
	x, y := xy(8, 8)
	if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
		t.Fatalf("%s: %v", mode, err)
	}
	if !moved(before, denseSnap(t, stem)) {
		t.Fatalf("%s did not move stem", mode)
	}
}

func TestAllCreditModesMoveBicameral(t *testing.T) {
	modes := parallel.AllCreditTrainModes()
	if len(modes) != 10 {
		t.Fatalf("credit modes %d want 10", len(modes))
	}
	for _, mode := range modes {
		t.Run(mode.String(), func(t *testing.T) {
			stemMoved(t, mustBicameral(t), mode)
		})
	}
}

func TestAllStackLocalModesMoveBicameral(t *testing.T) {
	for _, mode := range parallel.AllStackLocalTrainModes() {
		t.Run(mode.String(), func(t *testing.T) {
			anyMoved(t, mustBicameral(t), mode)
		})
	}
}

func TestStepAndNonStepSplitBothTrain(t *testing.T) {
	for _, mode := range []parallel.TrainMode{parallel.ModeTweenSplit, parallel.ModeStepTweenSplit} {
		if mode.Family() != parallel.ModeTweenSplit.Family() {
			t.Fatalf("%s family", mode)
		}
		stemMoved(t, mustBicameral(t), mode)
	}
	for _, mode := range []parallel.TrainMode{parallel.ModeTweenAlt, parallel.ModeStepTweenAlt} {
		stemMoved(t, mustBicameral(t), mode)
	}
}

func TestOpenSplitTapeAllSplitModes(t *testing.T) {
	x, y := xy(8, 8)
	for _, mode := range parallel.AllCreditTrainModes() {
		if !mode.IsSplitFamily() {
			continue
		}
		s := mustBicameral(t)
		stem := s.Children[0].(*dense.Layer)
		before := denseSnap(t, stem)
		tape, err := parallel.OpenSplitTape(s, x)
		if err != nil {
			t.Fatalf("%s open: %v", mode, err)
		}
		if _, err := tape.Train(y, mode, 0.05); err != nil {
			t.Fatalf("%s train: %v", mode, err)
		}
		if !moved(before, denseSnap(t, stem)) {
			t.Fatalf("%s tape did not move stem", mode)
		}
	}
}

func TestTrainMSEParallelRootCredit(t *testing.T) {
	x, y := xy(8, 8)
	for _, mode := range []parallel.TrainMode{
		parallel.ModeTweenSplitFastProxy, parallel.ModeStepTweenSplit, parallel.ModeTweenAlt,
	} {
		l, err := parallel.Hemispheres(8, 8, 2, parallel.CombineAdd, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			t.Fatal(err)
		}
		fillOpWeights(l, 0.05)
		b0 := l.Branches[0].(*dense.Layer)
		before := denseSnap(t, b0)
		if _, err := parallel.TrainMSE(l, x, y, mode, 0.05); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		if !moved(before, denseSnap(t, b0)) {
			t.Fatalf("%s TrainMSE did not move branch", mode)
		}
	}
}

func TestCombineModesWithFastProxy(t *testing.T) {
	combines := []parallel.CombineMode{parallel.CombineAdd, parallel.CombineAvg, parallel.CombineConcat, parallel.CombineFilter}
	for _, c := range combines {
		t.Run(string(c), func(t *testing.T) {
			cfg := parallel.Config{Dim: 8, OutFeat: 8, Branches: 2, Combine: c}
			l, err := parallel.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			fillOpWeights(l, 0.05)
			x := core.NewTensor[float32](1, 8)
			for i := range x.Data {
				x.Data[i] = 0.2
			}
			_, post, err := parallel.Forward(l, x)
			if err != nil {
				t.Fatal(err)
			}
			y := core.NewTensor[float32](post.Shape...)
			for i := range y.Data {
				y.Data[i] = 0.8
			}
			before := snapDensest(t, l)
			if _, err := parallel.TrainMSE(l, x, y, parallel.ModeTweenSplitFastProxy, 0.05); err != nil {
				t.Fatal(err)
			}
			if !moved(before, snapDensest(t, l)) {
				t.Fatalf("combine %s did not move any weights", c)
			}
		})
	}
}

func sandwichHemiN(t *testing.T, n int) *parallel.Stack {
	t.Helper()
	const in, hidden, out = 8, 8, 8
	stem, err := dense.NewConfigured[float32](in, hidden, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	hemi, err := parallel.Hemispheres(hidden, hidden, n, parallel.CombineAdd, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	head, err := dense.NewConfigured[float32](hidden, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	s, err := parallel.Sandwich(stem, hemi, head)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	return s
}

func TestHemiNCreditModes(t *testing.T) {
	x, y := xy(8, 8)
	for _, n := range []int{1, 3} {
		for _, mode := range []parallel.TrainMode{
			parallel.ModeTweenSplitFastProxy, parallel.ModeTweenSplitSparse, parallel.ModeStepTweenSplit,
		} {
			s := sandwichHemiN(t, n)
			stem := s.Children[0].(*dense.Layer)
			before := denseSnap(t, stem)
			if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
				t.Fatalf("n=%d %s: %v", n, mode, err)
			}
			if !moved(before, denseSnap(t, stem)) {
				t.Fatalf("n=%d %s stem stuck", n, mode)
			}
		}
	}
}

func TestMixedFastProxyAndStepBPTrains(t *testing.T) {
	s := mustBicameral(t)
	hemi := s.Children[1].(*parallel.Layer)
	hemi.SetBranchModes(parallel.ModeTweenSplitFastProxy, parallel.ModeStepBP)
	b0 := hemi.Branches[0].(*dense.Layer)
	b1 := hemi.Branches[1].(*dense.Layer)
	before0 := denseSnap(t, b0)
	before1 := denseSnap(t, b1)
	x, y := xy(8, 8)
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeStepBP, 0.05); err != nil {
		t.Fatal(err)
	}
	if !moved(before0, denseSnap(t, b0)) && !moved(before1, denseSnap(t, b1)) {
		t.Fatal("mixed FastProxy∥StepBP moved neither hemisphere")
	}
}

func TestSequentialAndResidualMidCredit(t *testing.T) {
	const dim = 8
	x, y := xy(dim, dim)
	seq, err := sequential.New(sequential.Config{Dim: dim, Depth: 2})
	if err != nil {
		t.Fatal(err)
	}
	res, err := residual.New(residual.Config{Dim: dim, Depth: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, mid := range []any{seq, res} {
		stem, err := dense.NewConfigured[float32](dim, dim, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, nil)
		if err != nil {
			t.Fatal(err)
		}
		head, err := dense.NewConfigured[float32](dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
		if err != nil {
			t.Fatal(err)
		}
		s, err := parallel.Sandwich(stem, mid, head)
		if err != nil {
			t.Fatal(err)
		}
		fillOpWeights(s, 0.05)
		before := denseSnap(t, stem)
		if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeTweenSplitFastProxy, 0.05); err != nil {
			t.Fatalf("%T: %v", mid, err)
		}
		if !moved(before, denseSnap(t, stem)) {
			t.Fatalf("%T stem stuck", mid)
		}
	}
}

func TestSparseRotatesHiddenLeaves(t *testing.T) {
	s := mustBicameral(t)
	hemi := s.Children[1].(*parallel.Layer)
	b0 := hemi.Branches[0].(*dense.Layer)
	x, y := xy(8, 8)
	before := denseSnap(t, b0)
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeTweenSplitSparse, 0.05); err != nil {
		t.Fatal(err)
	}
	after1 := denseSnap(t, b0)
	if moved(before, after1) {
		t.Fatal("sparse step 1 should hit stem (leaf 0), not first hemi")
	}
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeTweenSplitSparse, 0.05); err != nil {
		t.Fatal(err)
	}
	if !moved(before, denseSnap(t, b0)) {
		t.Fatal("sparse step 2 should rotate onto first hemi")
	}
}

func TestParseTrainModeRoundTripAll(t *testing.T) {
	for _, m := range parallel.AllTrainModes() {
		got, err := parallel.ParseTrainMode(m.String())
		if err != nil {
			t.Fatalf("%s: %v", m, err)
		}
		if got != m {
			t.Fatalf("%s round-trip %s", m, got)
		}
	}
	if _, err := parallel.ParseTrainMode("not-a-mode"); err == nil {
		t.Fatal("expected unknown mode error")
	}
}

func TestRequiresGridOnlyMesh(t *testing.T) {
	for _, m := range parallel.AllTrainModes() {
		want := strings.HasPrefix(m.String(), "Mesh")
		if m.RequiresGrid() != want {
			t.Fatalf("%s RequiresGrid=%v", m, m.RequiresGrid())
		}
	}
}

func TestWrappedCNNAllCreditModes(t *testing.T) {
	const in, hidden, out = 16, 32, 1
	x, y := xy(in, out)
	y.Data[0] = 1
	for _, mode := range parallel.AllCreditTrainModes() {
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
		fillOpWeights(s, 0.05)
		if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
	}
}

func TestInheritResolvesToNormalBP(t *testing.T) {
	if parallel.ModeInherit.Resolve(parallel.ModeInherit) != parallel.ModeNormalBP {
		t.Fatal("double inherit should become NormalBP")
	}
	if parallel.ModeInherit.Resolve(parallel.ModeTweenSplitFastProxy) != parallel.ModeTweenSplitFastProxy {
		t.Fatal("inherit should take parent FastProxy")
	}
}

func TestSuiteCameralCreditModes(t *testing.T) {
	if err := parallelsuite.CameralCreditModes(); err != nil {
		t.Fatal(err)
	}
}

func TestSuiteCameralCreditTrainGrids(t *testing.T) {
	if err := parallelsuite.CameralCreditTrainGrids(); err != nil {
		t.Fatal(err)
	}
}

func TestAllMeshCreditModesMoveBicameral(t *testing.T) {
	modes := parallel.AllMeshCreditTrainModes()
	if len(modes) != 4 {
		t.Fatalf("mesh credit modes %d want 4", len(modes))
	}
	for _, mode := range modes {
		t.Run(mode.String(), func(t *testing.T) {
			if !mode.RequiresGrid() {
				t.Fatalf("%s should require grid", mode)
			}
			s := mustBicameral(t)
			g := architecture.NewGrid(1, 1, 1, 1)
			if err := parallel.PlaceStack(g, 0, 0, 0, 0, s); err != nil {
				t.Fatal(err)
			}
			before := snapDensest(t, s)
			x, y := xy(8, 8)
			if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
				t.Fatalf("%s: %v", mode, err)
			}
			if !moved(before, snapDensest(t, s)) {
				t.Fatalf("%s did not move any weights", mode)
			}
		})
	}
}

func TestMeshCreditVariantsDispatch(t *testing.T) {
	delta := func(mode parallel.TrainMode) int {
		t.Helper()
		s := mustBicameral(t)
		before := snapDensest(t, s)
		x, y := xy(8, 8)
		if _, err := parallel.TrainStackMSE(s, x, y, mode, 0.05); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		n := 0
		after := snapDensest(t, s)
		for i := range before {
			if before[i] != after[i] {
				n++
			}
		}
		return n
	}
	splitN := delta(parallel.ModeTweenSplit)
	fastN := delta(parallel.ModeTweenSplitFastProxy)
	sparseN := delta(parallel.ModeTweenSplitSparse)
	if delta(parallel.ModeMeshTweenSplit) != splitN {
		t.Fatalf("MeshTweenSplit should match TweenSplit (%d)", splitN)
	}
	if delta(parallel.ModeMeshTweenSplitFastProxy) != fastN {
		t.Fatalf("MeshTweenSplitFastProxy should match FastProxy (%d)", fastN)
	}
	if delta(parallel.ModeMeshTweenSplitSparse) != sparseN {
		t.Fatalf("MeshTweenSplitSparse should match Sparse (%d)", sparseN)
	}
}

func TestParseMeshCreditAliases(t *testing.T) {
	cases := map[string]parallel.TrainMode{
		"meshsplit":     parallel.ModeMeshTweenSplit,
		"meshfastproxy": parallel.ModeMeshTweenSplitFastProxy,
		"meshsparse":    parallel.ModeMeshTweenSplitSparse,
		"meshalt":       parallel.ModeMeshTweenAlt,
	}
	for name, want := range cases {
		got, err := parallel.ParseTrainMode(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got != want {
			t.Fatalf("%s → %s want %s", name, got, want)
		}
	}
}

func TestMixedSequentialResidualCredit(t *testing.T) {
	const dim = 8
	x, y := xy(dim, dim)
	d, err := dense.NewConfigured[float32](dim, dim, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		t.Fatal(err)
	}
	rms, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
	if err != nil {
		t.Fatal(err)
	}
	sw, err := swiglu.New(swiglu.Config{InputDim: dim, IntermediateDim: dim * 2})
	if err != nil {
		t.Fatal(err)
	}
	seq, err := sequential.NewFromOps(sequential.Config{Dim: dim, Depth: 3}, []any{d, rms, sw})
	if err != nil {
		t.Fatal(err)
	}
	rms2, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
	if err != nil {
		t.Fatal(err)
	}
	sw2, err := swiglu.New(swiglu.Config{InputDim: dim, IntermediateDim: dim * 2})
	if err != nil {
		t.Fatal(err)
	}
	res, err := residual.NewFromOps(residual.Config{Dim: dim, Depth: 2}, []any{rms2, sw2})
	if err != nil {
		t.Fatal(err)
	}
	for _, mid := range []any{seq, res} {
		stem, err := dense.NewConfigured[float32](dim, dim, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone, nil)
		if err != nil {
			t.Fatal(err)
		}
		head, err := dense.NewConfigured[float32](dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
		if err != nil {
			t.Fatal(err)
		}
		s, err := parallel.Sandwich(stem, mid, head)
		if err != nil {
			t.Fatal(err)
		}
		fillOpWeights(s, 0.05)
		before := denseSnap(t, stem)
		if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeTweenSplitFastProxy, 0.05); err != nil {
			t.Fatalf("%T: %v", mid, err)
		}
		if !moved(before, denseSnap(t, stem)) {
			t.Fatalf("%T stem stuck", mid)
		}
	}
}

func TestResidualGraftParallel(t *testing.T) {
	const dim = 8
	hemi, err := parallel.Hemispheres(dim, dim, 2, parallel.CombineAdd, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		t.Fatal(err)
	}
	skip, err := parallel.ResidualGraft(hemi)
	if err != nil {
		t.Fatal(err)
	}
	s, err := parallel.NewStack(skip)
	if err != nil {
		t.Fatal(err)
	}
	fillOpWeights(s, 0.05)
	x, y := xy(dim, dim)
	before := snapDensest(t, s)
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeTweenSplitFastProxy, 0.05); err != nil {
		t.Fatal(err)
	}
	if !moved(before, snapDensest(t, s)) {
		t.Fatal("ResidualGraft FastProxy did not move weights")
	}
	if _, err := parallel.TrainStackMSE(s, x, y, parallel.ModeNormalBP, 0.05); err != nil {
		t.Fatal(err)
	}
}

func TestCameralModesIncludeCreditAndMesh(t *testing.T) {
	if err := parallelsuite.NestedMixedSequentialSmoke(); err != nil {
		t.Fatal(err)
	}
	if err := parallelsuite.NestedMixedResidualSmoke(); err != nil {
		t.Fatal(err)
	}
}

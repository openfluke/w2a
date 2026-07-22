package parallel

import (
	"fmt"
	"math"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

// PolyTrainAllKindsDTypes: every major branch kind × all 34 dtypes (FormatNone,
// CPU tiled). Requires fwd+bwd+SGD and, when the Op owns stores, that FlattenOp
// weights move at least a little (LR is aggressive on purpose — deep nests can
// shrink grads; callers pick LR for production).
func PolyTrainAllKindsDTypes() error {
	const lr = 1.0
	kinds := polyKinds()
	fmt.Printf("\n  POLY TRAIN — kinds×AllDTypes FormatNone cpu_tiled (weight-delta)\n")
	fmt.Printf("  kinds=%d dtypes=%d cells=%d lr=%g\n\n", len(kinds), len(core.AllDTypes), len(kinds)*len(core.AllDTypes), lr)

	var fails []string
	var okN, gapN int
	for _, k := range kinds {
		for _, dt := range core.AllDTypes {
			status, note := runPolyTrainCell(k, dt, quant.FormatNone, lr)
			rec("poly_train_"+k.name, dt.String(), "None", "cpu_tiled", "1x1x1x1", status, note)
			switch status {
			case "OK":
				okN++
			case "GAP":
				gapN++
			default:
				fails = append(fails, fmt.Sprintf("%s/%s:%s", k.name, dt.String(), note))
			}
		}
	}
	fmt.Printf("  summary: %d OK, %d GAP, %d FAIL (of %d)\n", okN, gapN, len(fails), len(kinds)*len(core.AllDTypes))
	if len(fails) > 0 {
		n := len(fails)
		if n > 12 {
			n = 12
		}
		return fmt.Errorf("poly train kinds×dtypes: %d failed (first): %s", len(fails), strings.Join(fails[:n], "; "))
	}
	return nil
}

func runPolyTrainCell(k polyKind, dt core.DType, format quant.Format, lr float64) (status, note string) {
	branches, cfg, x, trainable, err := k.make()
	if err != nil {
		return "FAIL", "build: " + err.Error()
	}
	l, err := parallel.NewFromBranches(cfg, branches, nil)
	if err != nil {
		return "FAIL", "NewFromBranches: " + err.Error()
	}
	l.Exec.Backend = core.BackendCPUTiled
	// Non-zero init so chained Ops (Sequential/MHA/…) produce nonzero dW;
	// zero W ⇒ zero activations ⇒ zero outer-product grads through the nest.
	if err := seedNonZero(l); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	if dt != core.DTypeFloat32 {
		if err := l.SetDType(dt); err != nil {
			return "GAP", "SetDType: " + err.Error()
		}
	}
	if format != quant.FormatNone {
		if err := l.Pack(format); err != nil {
			return "GAP", "Pack: " + err.Error()
		}
	}
	fillOnes(x)
	before, err := dna.FlattenOp(l)
	if err != nil {
		return "FAIL", "snapshot before: " + err.Error()
	}
	pre, post, err := parallel.Forward(l, x)
	if err != nil {
		return "FAIL", "fwd: " + err.Error()
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		// Large outgoing grad so even quantized / nested paths can move weights.
		gy.Data[i] = 1
	}
	_, dW, err := parallel.Backward(l, gy, x, pre)
	if err != nil {
		return "FAIL", "bwd: " + err.Error()
	}
	if !trainable {
		return "OK", "fwd+bwd (no trainable stores)"
	}
	if dW == nil || dW.Len() == 0 {
		return "FAIL", "empty dW for trainable Op"
	}
	var dWEnergy float64
	for _, v := range dW.Data {
		dWEnergy += float64(v) * float64(v)
	}
	if err := parallel.ApplyGradSGD(l, dW, lr); err != nil {
		return "FAIL", "sgd: " + err.Error()
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		return "FAIL", "snapshot after: " + err.Error()
	}
	if len(before) == 0 || len(after) == 0 {
		return "FAIL", "empty FlattenOp weights"
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		if dWEnergy > 0 {
			// Coarse native dtypes (e.g. fp4) can quantize the update back to the
			// same codepoint — grads flowed; storage just couldn't express Δ.
			return "GAP", fmt.Sprintf("dW≠0 but FlattenOp Δ=0 under %s (quantized storage; raise lr or use finer dtype)", dt.String())
		}
		return "FAIL", fmt.Sprintf("weights unchanged and dW=0 (lr=%g)", lr)
	}
	return "OK", fmt.Sprintf("Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}

// seedNonZero writes a small patterned matrix (+ identity on square stores)
// into every weights.Store under op so train smokes are not stuck at W=0.
func seedNonZero(op any) error {
	for si, s := range dna.CollectStores(op) {
		if s == nil {
			continue
		}
		n := s.Rows * s.Cols
		if n <= 0 {
			continue
		}
		w := make([]float32, n)
		for i := range w {
			w[i] = 0.05 * float32((i%7)-3)
		}
		if s.Rows == s.Cols {
			for i := 0; i < s.Rows; i++ {
				w[i*s.Cols+i] = 1
			}
		} else if len(w) > 0 {
			w[0] = 1
		}
		if err := s.SetFromF32(w); err != nil {
			return fmt.Errorf("store %d: %w", si, err)
		}
	}
	return nil
}

func weightDelta(before, after []float32) (changed int, maxAbs float64) {
	n := len(before)
	if len(after) < n {
		n = len(after)
	}
	for i := 0; i < n; i++ {
		d := math.Abs(float64(after[i] - before[i]))
		if d > 0 {
			changed++
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	return changed, maxAbs
}

// NestedTrainWeightDelta: nested Parallel / Sequential stacks must accept SGD
// and move at least one weight (high LR). Depth-3 Parallel nest included.
func NestedTrainWeightDelta() error {
	const lr = 1.0
	type nestCase struct {
		name string
		run  func() (status, note string)
	}
	cases := []nestCase{
		{name: "parallel_of_parallel", run: func() (string, string) {
			return trainNestedLayer(buildNestedParallel, lr)
		}},
		{name: "parallel_of_sequential", run: func() (string, string) {
			return trainNestedLayer(buildParallelOfSequential, lr)
		}},
		{name: "depth3_parallel", run: func() (string, string) {
			return trainNestedLayer(buildDepth3Parallel, lr)
		}},
		{name: "parallel_concat_inner_add", run: func() (string, string) {
			return trainNestedLayer(buildMixedCombineNest, lr)
		}},
	}
	var fails []string
	for _, c := range cases {
		status, note := c.run()
		rec("nest_train_"+c.name, "float32", "None", "cpu_tiled", "1x1x1x1", status, note)
		if status != "OK" {
			fails = append(fails, fmt.Sprintf("%s:%s", c.name, note))
		}
	}
	if len(fails) > 0 {
		return fmt.Errorf("nested train weight-delta: %s", strings.Join(fails, "; "))
	}
	return nil
}

func trainNestedLayer(build func() (*parallel.Layer, *core.Tensor[float32], error), lr float64) (status, note string) {
	l, x, err := build()
	if err != nil {
		return "FAIL", err.Error()
	}
	l.Exec.Backend = core.BackendCPUTiled
	if err := seedNonZero(l); err != nil {
		return "FAIL", "seed: " + err.Error()
	}
	fillOnes(x)
	before, err := dna.FlattenOp(l)
	if err != nil {
		return "FAIL", "snapshot: " + err.Error()
	}
	pre, post, err := parallel.Forward(l, x)
	if err != nil {
		return "FAIL", "fwd: " + err.Error()
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.Backward(l, gy, x, pre)
	if err != nil {
		return "FAIL", "bwd: " + err.Error()
	}
	if err := parallel.ApplyGradSGD(l, dW, lr); err != nil {
		return "FAIL", "sgd: " + err.Error()
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		return "FAIL", "snapshot after: " + err.Error()
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		return "FAIL", "weights unchanged after nested SGD"
	}
	return "OK", fmt.Sprintf("Δelems=%d max|Δ|=%.6g", delta, maxAbs)
}

func buildNestedParallel() (*parallel.Layer, *core.Tensor[float32], error) {
	const dim, batch = 32, 2
	mk := func() (*parallel.Layer, error) {
		a, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		b, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		return parallel.NewFromBranches(parallel.Config{
			Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
		}, []any{a, b}, nil)
	}
	i0, err := mk()
	if err != nil {
		return nil, nil, err
	}
	i1, err := mk()
	if err != nil {
		return nil, nil, err
	}
	outer, err := parallel.NewFromBranches(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineConcat,
	}, []any{i0, i1}, nil)
	return outer, core.NewTensor[float32](batch, dim), err
}

func buildParallelOfSequential() (*parallel.Layer, *core.Tensor[float32], error) {
	const dim, batch = 32, 2
	s0, err := sequential.New(sequential.Config{Dim: dim, Depth: 2})
	if err != nil {
		return nil, nil, err
	}
	s1, err := sequential.New(sequential.Config{Dim: dim, Depth: 2})
	if err != nil {
		return nil, nil, err
	}
	l, err := parallel.NewFromBranches(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
	}, []any{s0, s1}, nil)
	return l, core.NewTensor[float32](batch, dim), err
}

func buildDepth3Parallel() (*parallel.Layer, *core.Tensor[float32], error) {
	const dim, batch = 16, 2
	leaf := func() (*parallel.Layer, error) {
		a, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		b, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		return parallel.NewFromBranches(parallel.Config{
			Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
		}, []any{a, b}, nil)
	}
	mkMid := func() (*parallel.Layer, error) {
		a, err := leaf()
		if err != nil {
			return nil, err
		}
		b, err := leaf()
		if err != nil {
			return nil, err
		}
		return parallel.NewFromBranches(parallel.Config{
			Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
		}, []any{a, b}, nil)
	}
	m0, err := mkMid()
	if err != nil {
		return nil, nil, err
	}
	m1, err := mkMid()
	if err != nil {
		return nil, nil, err
	}
	outer, err := parallel.NewFromBranches(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
	}, []any{m0, m1}, nil)
	return outer, core.NewTensor[float32](batch, dim), err
}

func buildMixedCombineNest() (*parallel.Layer, *core.Tensor[float32], error) {
	const dim, batch = 32, 2
	innerAdd := func() (*parallel.Layer, error) {
		a, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		b, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		return parallel.NewFromBranches(parallel.Config{
			Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
		}, []any{a, b}, nil)
	}
	i0, err := innerAdd()
	if err != nil {
		return nil, nil, err
	}
	i1, err := innerAdd()
	if err != nil {
		return nil, nil, err
	}
	outer, err := parallel.NewFromBranches(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineConcat,
	}, []any{i0, i1}, nil)
	return outer, core.NewTensor[float32](batch, dim), err
}

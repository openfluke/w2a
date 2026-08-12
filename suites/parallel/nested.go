package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

// NestedParallelSmoke: Parallel of Parallel (loom nesting).
func NestedParallelSmoke() error {
	const dim, batch = 32, 2
	mkInner := func() (*parallel.Layer, error) {
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
	inner0, err := mkInner()
	if err != nil {
		return err
	}
	inner1, err := mkInner()
	if err != nil {
		return err
	}
	outer, err := parallel.NewFromBranches(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineConcat,
	}, []any{inner0, inner1}, nil)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](batch, dim)
	fillOnes(x)
	if err := seedNonZero(outer); err != nil {
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	before, err := dna.FlattenOp(outer)
	if err != nil {
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	pre, post, err := parallel.Forward(outer, x)
	if err != nil {
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if post.Shape[1] != dim*2 {
		err := fmt.Errorf("nested concat out feat %d want %d", post.Shape[1], dim*2)
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.Backward(outer, gy, x, pre)
	if err != nil {
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if err := parallel.ApplyGradSGD(outer, dW, 1.0); err != nil {
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	after, err := dna.FlattenOp(outer)
	if err != nil {
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		err := fmt.Errorf("nested parallel weights unchanged")
		rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	rec("nested_parallel", "float32", "none", "cpu_tiled", "1x1x1x1", "OK",
		fmt.Sprintf("Parallel of Parallel Δelems=%d max|Δ|=%.6g", delta, maxAbs))
	return nil
}

// NestedSequentialSmoke: Parallel of Sequential Dense stacks.
func NestedSequentialSmoke() error {
	const dim, batch = 32, 2
	s0, err := sequential.New(sequential.Config{Dim: dim, Depth: 2})
	if err != nil {
		return err
	}
	s1, err := sequential.New(sequential.Config{Dim: dim, Depth: 2})
	if err != nil {
		return err
	}
	l, err := parallel.NewFromBranches(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAdd,
	}, []any{s0, s1}, nil)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](batch, dim)
	fillOnes(x)
	if err := seedNonZero(l); err != nil {
		rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	before, err := dna.FlattenOp(l)
	if err != nil {
		rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	pre, post, err := parallel.Forward(l, x)
	if err != nil {
		rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.Backward(l, gy, x, pre)
	if err != nil {
		rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if err := parallel.ApplyGradSGD(l, dW, 1.0); err != nil {
		rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		err := fmt.Errorf("nested sequential weights unchanged")
		rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	rec("nested_sequential", "float32", "none", "cpu_tiled", "1x1x1x1", "OK",
		fmt.Sprintf("Parallel of Sequential Δelems=%d max|Δ|=%.6g", delta, maxAbs))
	return nil
}

// CameralBicameralSmoke: parallel.Bicameral sandwich (Dense→Hemi∥Hemi→Dense).
func CameralBicameralSmoke() error {
	const in, hidden, out, batch = 10, 32, 1, 2
	s, err := parallel.Bicameral(in, hidden, out, core.ActivationLeakyReLU, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](batch, in)
	fillOnes(x)
	if err := seedNonZero(s); err != nil {
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	before, err := dna.FlattenOp(s)
	if err != nil {
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	pre, post, err := parallel.ForwardStack(s, x)
	if err != nil {
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if len(post.Shape) != 2 || post.Shape[0] != batch || post.Shape[1] != out {
		err := fmt.Errorf("bicameral out shape %v want [%d,%d]", post.Shape, batch, out)
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.BackwardStack(s, gy, x, pre)
	if err != nil {
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if dW == nil || dW.Len() != s.GradWSize() {
		err := fmt.Errorf("dW len %v want %d", dW, s.GradWSize())
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if err := parallel.ApplyGradSGDStack(s, dW, 1.0); err != nil {
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	after, err := dna.FlattenOp(s)
	if err != nil {
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		err := fmt.Errorf("bicameral weights unchanged")
		rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	rec("cameral_bicameral", "float32", "none", "cpu_tiled", "1x1x1x1", "OK",
		fmt.Sprintf("Bicameral Δelems=%d max|Δ|=%.6g", delta, maxAbs))
	return nil
}

// CameralNestedStackInParallel: Parallel of Stack[Dense, Hemispheres, Dense] (nested cameral).
func CameralNestedStackInParallel() error {
	const dim, batch = 16, 2
	mkBranch := func() (*parallel.Stack, error) {
		a, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		b, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
		if err != nil {
			return nil, err
		}
		hemi, err := parallel.Hemispheres(dim, dim, 2, parallel.CombineAdd, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
		if err != nil {
			return nil, err
		}
		return parallel.NewStack(a, hemi, b)
	}
	left, err := mkBranch()
	if err != nil {
		return err
	}
	right, err := mkBranch()
	if err != nil {
		return err
	}
	outer, err := parallel.HemispheresFrom(parallel.Config{
		Dim: dim, OutFeat: dim, Branches: 2, Combine: parallel.CombineAvg,
	}, []any{left, right}, nil)
	if err != nil {
		return err
	}
	root, err := parallel.Sandwich(outer)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](batch, dim)
	fillOnes(x)
	if err := seedNonZero(root); err != nil {
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	before, err := dna.FlattenOp(root)
	if err != nil {
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	pre, post, err := parallel.ForwardStack(root, x)
	if err != nil {
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if post.Shape[1] != dim {
		err := fmt.Errorf("nested cameral feat %d want %d", post.Shape[1], dim)
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.BackwardStack(root, gy, x, pre)
	if err != nil {
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if err := parallel.ApplyGradSGDStack(root, dW, 1.0); err != nil {
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	after, err := dna.FlattenOp(root)
	if err != nil {
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		err := fmt.Errorf("nested cameral weights unchanged")
		rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	rec("cameral_nested_stack", "float32", "none", "cpu_tiled", "1x1x1x1", "OK",
		fmt.Sprintf("Stack-in-Parallel cameral Δelems=%d max|Δ|=%.6g", delta, maxAbs))
	return nil
}

// CameralHemispheresN: Hemispheres(n=3, concat) inside a Sandwich stack.
func CameralHemispheresN() error {
	const dim, batch, n = 16, 2, 3
	stem, err := dense.New(dim, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return err
	}
	hemi, err := parallel.Hemispheres(dim, dim, n, parallel.CombineConcat, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	head, err := dense.New(dim*n, dim, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		return err
	}
	root, err := parallel.Sandwich(stem, hemi, head)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](batch, dim)
	fillOnes(x)
	if err := seedNonZero(root); err != nil {
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	before, err := dna.FlattenOp(root)
	if err != nil {
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	pre, post, err := parallel.ForwardStack(root, x)
	if err != nil {
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if post.Shape[1] != dim {
		err := fmt.Errorf("tri-cameral out feat %d want %d", post.Shape[1], dim)
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.BackwardStack(root, gy, x, pre)
	if err != nil {
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if err := parallel.ApplyGradSGDStack(root, dW, 1.0); err != nil {
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	after, err := dna.FlattenOp(root)
	if err != nil {
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		err := fmt.Errorf("tri-cameral weights unchanged")
		rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	rec("cameral_hemispheres_n", "float32", "none", "cpu_tiled", "1x1x1x1", "OK",
		fmt.Sprintf("Hemispheres(n=%d) sandwich Δelems=%d max|Δ|=%.6g", n, delta, maxAbs))
	return nil
}

package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/sequential"
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

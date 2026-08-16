package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

func mixedSeq(dim int) (*sequential.Layer, error) {
	d, err := dense.NewConfigured[float32](dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		return nil, err
	}
	rms, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
	if err != nil {
		return nil, err
	}
	sw, err := swiglu.New(swiglu.Config{InputDim: dim, IntermediateDim: dim * 2})
	if err != nil {
		return nil, err
	}
	return sequential.NewFromOps(sequential.Config{Dim: dim, Depth: 3}, []any{d, rms, sw})
}

// NestedMixedSequentialSmoke: Parallel of Sequential with non-Dense children.
func NestedMixedSequentialSmoke() error {
	const dim, batch = 8, 2
	s0, err := mixedSeq(dim)
	if err != nil {
		return err
	}
	s1, err := mixedSeq(dim)
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
		rec("nested_mixed_seq", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	before, err := dna.FlattenOp(l)
	if err != nil {
		return err
	}
	pre, post, err := parallel.Forward(l, x)
	if err != nil {
		rec("nested_mixed_seq", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.Backward(l, gy, x, pre)
	if err != nil {
		rec("nested_mixed_seq", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if err := parallel.ApplyGradSGD(l, dW, 1.0); err != nil {
		rec("nested_mixed_seq", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		return err
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		err := fmt.Errorf("mixed sequential nest weights unchanged")
		rec("nested_mixed_seq", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	rec("nested_mixed_seq", "float32", "none", "cpu_tiled", "1x1x1x1", "OK",
		fmt.Sprintf("Parallel of mixed Sequential Δelems=%d max|Δ|=%.6g", delta, maxAbs))
	return nil
}

// NestedMixedResidualSmoke: Residual mixed F plus Parallel ResidualGraft.
func NestedMixedResidualSmoke() error {
	const dim, batch = 8, 2
	rms, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
	if err != nil {
		return err
	}
	sw, err := swiglu.New(swiglu.Config{InputDim: dim, IntermediateDim: dim * 2})
	if err != nil {
		return err
	}
	res, err := residual.NewFromOps(residual.Config{Dim: dim, Depth: 2}, []any{rms, sw})
	if err != nil {
		return err
	}
	hemi, err := parallel.Hemispheres(dim, dim, 2, parallel.CombineAdd, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone)
	if err != nil {
		return err
	}
	graft, err := parallel.ResidualGraft(hemi)
	if err != nil {
		return err
	}
	root, err := parallel.NewStack(res, graft)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](batch, dim)
	fillOnes(x)
	if err := seedNonZero(root); err != nil {
		rec("nested_mixed_res", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	before, err := dna.FlattenOp(root)
	if err != nil {
		return err
	}
	pre, post, err := parallel.ForwardStack(root, x)
	if err != nil {
		rec("nested_mixed_res", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := parallel.BackwardStack(root, gy, x, pre)
	if err != nil {
		rec("nested_mixed_res", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	if err := parallel.ApplyGradSGDStack(root, dW, 1.0); err != nil {
		rec("nested_mixed_res", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	after, err := dna.FlattenOp(root)
	if err != nil {
		return err
	}
	delta, maxAbs := weightDelta(before, after)
	if delta == 0 {
		err := fmt.Errorf("mixed residual / ResidualGraft weights unchanged")
		rec("nested_mixed_res", "float32", "none", "cpu_tiled", "1x1x1x1", "FAIL", err.Error())
		return err
	}
	rec("nested_mixed_res", "float32", "none", "cpu_tiled", "1x1x1x1", "OK",
		fmt.Sprintf("mixed Residual+Graft Δelems=%d max|Δ|=%.6g", delta, maxAbs))
	return nil
}

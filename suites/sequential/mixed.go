package sequential

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/systems/dna"
)

func mixedNonDenseChildren() error {
	const dim, batch = 8, 2
	d, err := dense.NewConfigured[float32](dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, nil)
	if err != nil {
		return err
	}
	rms, err := rmsnorm.New(rmsnorm.Config{Dim: dim})
	if err != nil {
		return err
	}
	sw, err := swiglu.New(swiglu.Config{InputDim: dim, IntermediateDim: dim * 2})
	if err != nil {
		return err
	}
	l, err := sequential.NewFromOps(sequential.Config{Dim: dim, Depth: 3}, []any{d, rms, sw})
	if err != nil {
		return err
	}
	if err := seedOp(l); err != nil {
		return err
	}
	before, err := dna.FlattenOp(l)
	if err != nil {
		return err
	}
	x := core.NewTensor[float32](batch, dim)
	for i := range x.Data {
		x.Data[i] = 0.2
	}
	pre, post, err := sequential.Forward(l, x)
	if err != nil {
		return fmt.Errorf("fwd: %w", err)
	}
	if post.Shape[1] != dim {
		return fmt.Errorf("post feat %d want %d", post.Shape[1], dim)
	}
	gy := core.NewTensor[float32](post.Shape...)
	for i := range gy.Data {
		gy.Data[i] = 1
	}
	_, dW, err := sequential.Backward(l, gy, x, pre)
	if err != nil {
		return fmt.Errorf("bwd: %w", err)
	}
	if dW == nil || dW.Len() != l.GradWSize() {
		return fmt.Errorf("dW len %v want %d", dW, l.GradWSize())
	}
	if err := sequential.ApplyGradSGD(l, dW, 0.1); err != nil {
		return err
	}
	after, err := dna.FlattenOp(l)
	if err != nil {
		return err
	}
	if sameF32(before, after) {
		return fmt.Errorf("mixed sequential weights unchanged")
	}
	fmt.Printf("(Dense→RMS→SwiGLU dim=%d) ", dim)
	rec("mixed_ops", "float32", "none", "cpu_tiled", "-", "OK", "Dense→RMSNorm→SwiGLU")
	return nil
}

func seedOp(op any) error {
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

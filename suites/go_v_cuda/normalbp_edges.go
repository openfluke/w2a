package go_v_cuda

import (
	"bytes"
	"fmt"
	"math"
	"os/exec"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/runtime/training"
	"github.com/openfluke/welvet/weights"
)

func encodeParallelChild(cfg parallel.Config, children [][]byte, gate []byte) []byte {
	var b bytes.Buffer
	writeI32(&b, skParallel, parallelCombineCode(cfg.Combine), int32(cfg.Dim), int32(cfg.OutFeat), int32(cfg.SeqLen), int32(cfg.SparseK))
	writeF64(&b, cfg.DisagreeBeta)
	writeI32(&b, int32(len(children)))
	for _, ch := range children {
		b.Write(ch)
	}
	if gate == nil {
		writeI32(&b, 0)
	} else {
		writeI32(&b, 1)
		b.Write(gate)
	}
	return b.Bytes()
}

func packParallelAll(l *parallel.Layer) ([]float32, error) {
	var out []float32
	for _, br := range l.Branches {
		var w []float32
		var err error
		if nested, ok := br.(*parallel.Layer); ok {
			w, err = packParallelAll(nested)
		} else {
			w, err = packPolyBranchW(br)
		}
		if err != nil {
			return nil, err
		}
		out = append(out, w...)
	}
	if l.Gate != nil {
		w, err := weights.MatrixF32(l.Gate.Weights)
		if err != nil {
			return nil, err
		}
		out = append(out, w...)
	}
	return out, nil
}

func checkParallelGradients(l *parallel.Layer, encoded []byte, shape []int32, x []float32, lr float64) error {
	input := core.NewTensor[float32](ints(shape)...)
	copy(input.Data, x)
	pre, post, err := parallel.Forward(l, input)
	if err != nil {
		return err
	}
	target := core.NewTensor[float32](post.Shape...)
	copy(target.Data, seedPat(len(post.Data), 0x31745))
	loss, err := training.MSE(post, target)
	if err != nil {
		return err
	}
	gy, err := training.MSEGrad(post, target)
	if err != nil {
		return err
	}
	dx, dw, err := parallel.Backward(l, gy, input, pre)
	if err != nil {
		return err
	}
	if err = parallel.ApplyGradSGD(l, dw, lr); err != nil {
		return err
	}
	w, err := packParallelAll(l)
	if err != nil {
		return err
	}
	ref := nbpLeafOut{pre: pre.Data, post: post.Data, dX: dx.Data, dW: dw.Data, WAfter: w, loss: loss}
	var b bytes.Buffer
	b.WriteString("NBL1")
	b.Write(encoded)
	writeI32(&b, int32(len(shape)))
	writeI32(&b, shape...)
	polyWriteF32s(&b, x)
	writeI32(&b, int32(len(post.Shape)))
	for _, d := range post.Shape {
		writeI32(&b, int32(d))
	}
	polyWriteF32s(&b, target.Data)
	writeF64(&b, lr)
	tool, err := findCUDAFwd()
	if err != nil {
		return err
	}
	cmd := exec.Command(tool)
	cmd.Stdin = &b
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("cuda gradient wire: %w: %s", err, stderr.String())
	}
	got, _, err := decodeRecNBPLeaf(raw, "NBLO")
	if err != nil {
		return err
	}
	return compareNBP(ref, got, true)
}

func CasesNormalBPEdges() []Case {
	var cases []Case
	for _, mode := range []parallel.CombineMode{parallel.CombineConcat, parallel.CombineAdd, parallel.CombineAvg, parallel.CombineFilter, parallel.CombineMax, parallel.CombineSparseK, parallel.CombineDisagree} {
		mode := mode
		cases = append(cases, Case{Name: fmt.Sprintf("Parallel direct gradients combine %s rank3 ×34 including gate", mode), Run: func() error {
			for _, dt := range core.AllDTypes {
				cfg := parallel.Config{Dim: 3, OutFeat: 3, Branches: 3, SeqLen: 2, Combine: mode, SparseK: 2, DisagreeBeta: 0.375}
				w := seedPat(27, 0x7771)
				gate := seedPat(9, 0x7772)
				l, err := parallel.NewConfigured[float32](cfg, dt, quant.FormatNone, w, gate)
				if err != nil {
					return err
				}
				var children [][]byte
				for i := 0; i < 3; i++ {
					children = append(children, polyEncDense(3, 3, dt, w[i*9:(i+1)*9]))
				}
				var gateWire []byte
				if mode == parallel.CombineFilter {
					gateWire = polyEncDense(3, 3, dt, gate)
				}
				if err = checkParallelGradients(l, encodeParallelChild(cfg, children, gateWire), []int32{2, 2, 3}, seedPat(12, 0x7773), 0.017); err != nil {
					return fmt.Errorf("%s: %w", dt, err)
				}
			}
			return nil
		}})
	}
	cases = append(cases, Case{Name: "Parallel deterministic Max/SparseK ties and zero learning rate", Run: func() error {
		for _, mode := range []parallel.CombineMode{parallel.CombineMax, parallel.CombineSparseK} {
			cfg := parallel.Config{Dim: 3, OutFeat: 3, Branches: 3, Combine: mode, SparseK: 1}
			w := seedPat(9, 0x3310)
			packed := append(append(append([]float32{}, w...), w...), w...)
			l, err := parallel.NewConfigured[float32](cfg, core.DTypeFloat32, quant.FormatNone, packed, nil)
			if err != nil {
				return err
			}
			child := polyEncDense(3, 3, core.DTypeFloat32, w)
			if err = checkParallelGradients(l, encodeParallelChild(cfg, [][]byte{child, child, child}, nil), []int32{2, 3}, seedPat(6, 0x3311), 0); err != nil {
				return err
			}
		}
		return nil
	}})
	cases = append(cases, Case{Name: "Parallel unequal Concat widths and nested Filter gate", Run: func() error {
		dt := core.DTypeFloat32
		innerCfg := parallel.Config{Dim: 3, OutFeat: 3, Branches: 2, Combine: parallel.CombineFilter}
		w := seedPat(18, 0x8181)
		gate := seedPat(6, 0x8182)
		nested, err := parallel.NewConfigured[float32](innerCfg, dt, quant.FormatNone, w, gate)
		if err != nil {
			return err
		}
		innerWire := encodeParallelChild(innerCfg, [][]byte{polyEncDense(3, 3, dt, w[:9]), polyEncDense(3, 3, dt, w[9:])}, polyEncDense(3, 2, dt, gate))
		wideW := seedPat(15, 0x8183)
		wide, err := dense.NewConfigured(3, 5, core.ActivationLinear, dt, quant.FormatNone, wideW)
		if err != nil {
			return err
		}
		cfg := parallel.Config{Dim: 3, OutFeat: 0, Branches: 2, Combine: parallel.CombineConcat}
		l, err := parallel.NewFromBranches(cfg, []any{nested, wide}, nil)
		if err != nil {
			return err
		}
		return checkParallelGradients(l, encodeParallelChild(cfg, [][]byte{innerWire, polyEncDense(3, 5, dt, wideW)}, nil), []int32{2, 3}, seedPat(6, 0x8184), 0.01)
	}})
	cases = append(cases, Case{Name: "KMeans Features ×34 dtypes with explicit backward", Run: func() error {
		for _, dt := range core.AllDTypes {
			cfg := kmeans.Config{NumClusters: 3, FeatureDim: 4, Temperature: 0.7, OutputMode: kmeans.OutputFeatures, Activation: core.ActivationLinear}
			fc, err := makeKmLeafCase(cfg, 2, 3, dt, 0x6611)
			if err != nil {
				return err
			}
			a, err := runGoKmNBP(fc)
			if err != nil {
				return err
			}
			b, err := runCppKmNBP(fc)
			if err != nil {
				return err
			}
			if err = compareKmNBP(a, b); err != nil {
				return fmt.Errorf("%s: %w", dt, err)
			}
		}
		return nil
	}})
	cases = append(cases, Case{Name: "KMeans all activations and temperature zero/default/sharp", Run: func() error {
		for _, bad := range []float32{1e-10, float32(math.NaN()), float32(math.Inf(1))} {
			if kmeansInputParity([]float32{0}, []float32{bad}) {
				return fmt.Errorf("KMeans cancellation gate accepted %g", bad)
			}
		}

		for act := core.ActivationType(-1); act <= 6; act++ {
			for _, temp := range []float64{0, 0.25, 2.5} {
				for _, mode := range []kmeans.OutputMode{kmeans.OutputProbabilities, kmeans.OutputFeatures} {
					cfg := kmeans.Config{NumClusters: 3, FeatureDim: 4, Temperature: temp, OutputMode: mode, Activation: act}
					fc, err := makeKmLeafCase(cfg, 2, 2, core.DTypeFloat32, 0x6677)
					if err != nil {
						return err
					}
					a, err := runGoKmNBP(fc)
					if err != nil {
						return err
					}
					b, err := runCppKmNBP(fc)
					if err != nil {
						return err
					}
					if err = compareKmNBP(a, b); err != nil {
						return fmt.Errorf("activation=%d temp=%g mode=%s: %w", act, temp, mode, err)
					}
				}
			}
		}
		return nil
	}})
	return cases
}

func CasesParallelVolumeDTypes() []Case {
	var cases []Case
	for _, mode := range []parallel.CombineMode{parallel.CombineAdd, parallel.CombineAvg} {
		for _, top := range []struct {
			name       string
			z, y, x, l int
		}{{"111", 1, 1, 1, 1}, {"chain", 1, 1, 1, 2}, {"211", 2, 1, 1, 1}} {
			mode, top := mode, top
			cases = append(cases, Case{Name: fmt.Sprintf("Parallel volume %s combine %s ×34", top.name, mode), Run: func() error {
				const batch, dim = 2, 4
				x := seedPat(batch*dim, 0x5121)
				target := seedPat(batch*dim, 0x5122)
				for _, dt := range core.AllDTypes {
					cells := makePnVolCells(top.z*top.y*top.x*top.l, dim, dim, 3, mode, 0x5123)
					for i := range cells {
						cells[i].dtype = dt
					}
					ref, err := runGoPnVolNBP(top.z, top.y, top.x, top.l, batch, cells, x, target, 0.013)
					if err != nil {
						return err
					}
					payload, err := encodePnVolNBP(top.z, top.y, top.x, top.l, batch, cells, x, target, 0.013)
					if err != nil {
						return err
					}
					tool, err := findCUDAFwd()
					if err != nil {
						return err
					}
					cmd := exec.Command(tool)
					cmd.Stdin = bytes.NewReader(payload)
					var stderr bytes.Buffer
					cmd.Stderr = &stderr
					raw, err := cmd.Output()
					if err != nil {
						return fmt.Errorf("%s: %w: %s", dt, err, stderr.String())
					}
					got, err := decodePnVolNBP(raw)
					if err != nil {
						return err
					}
					if err = compareVolNBP(ref, got, true); err != nil {
						return fmt.Errorf("%s: %w", dt, err)
					}
				}
				return nil
			}})
		}
	}
	return cases
}

package seven

import (
	"fmt"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func eye(rows, cols int) []float32 {
	w := make([]float32, rows*cols)
	n := rows
	if cols < n {
		n = cols
	}
	for i := 0; i < n; i++ {
		w[i*cols+i] = 1
	}
	return w
}

func placeAndRun(name string, format quant.Format, build func(g *architecture.Grid, format quant.Format) (*core.Tensor[float32], *core.Tensor[float32], error)) error {
	formats := []quant.Format{quant.FormatNone}
	if format == quant.FormatQ8_0 {
		formats = []quant.Format{quant.FormatNone, quant.FormatQ8_0}
	}
	for _, f := range formats {
		for _, multi := range []bool{false, true} {
			_ = multi // used inside via rebuild
		}
		// repeat-det
		g0 := architecture.NewGrid(1, 1, 1, 1)
		g0.Exec.Backend = core.BackendCPUTiled
		g0.Exec.MultiCore = false
		x, y, err := build(g0, f)
		if err != nil {
			return fmt.Errorf("%s/%s build: %w", name, f, err)
		}
		base, _, err := captureFwdBwd(g0, x)
		if err != nil {
			return fmt.Errorf("%s/%s fwd: %w", name, f, err)
		}
		g1 := architecture.NewGrid(1, 1, 1, 1)
		g1.Exec.Backend = core.BackendCPUTiled
		g1.Exec.MultiCore = false
		if _, _, err := build(g1, f); err != nil {
			return err
		}
		p1, _, err := captureFwdBwd(g1, x)
		if err != nil {
			return err
		}
		if err := suites.RequireDet(name+"/"+f.String()+" repeat", suites.MaxAbsDiff(base, p1), suites.DetTolFwd); err != nil {
			return err
		}

		// SC↔MC
		gSC := architecture.NewGrid(1, 1, 1, 1)
		gSC.Exec.Backend = core.BackendCPUTiled
		gSC.Exec.MultiCore = false
		if _, _, err := build(gSC, f); err != nil {
			return err
		}
		gMC := architecture.NewGrid(1, 1, 1, 1)
		gMC.Exec.Backend = core.BackendCPUTiled
		gMC.Exec.MultiCore = true
		if _, _, err := build(gMC, f); err != nil {
			return err
		}
		pSC, wSC, err := captureFwdBwd(gSC, x)
		if err != nil {
			return fmt.Errorf("%s/%s SC: %w", name, f, err)
		}
		pMC, wMC, err := captureFwdBwd(gMC, x)
		if err != nil {
			return fmt.Errorf("%s/%s MC: %w", name, f, err)
		}
		if err := suites.RequireDet(name+"/"+f.String()+" fwd SC↔MC", suites.MaxAbsDiff(pSC, pMC), suites.DetTolFwd); err != nil {
			return err
		}
		if len(wSC) > 0 && len(wMC) > 0 {
			if err := suites.RequireDet(name+"/"+f.String()+" bwd SC↔MC", suites.MaxAbsDiff(wSC, wMC), suites.DetTolBwd); err != nil {
				return err
			}
		}

		// short train (skip if no learnable path fails — still require finite)
		first, last, err := trainEpochs(gSC, x, y, 4, 1e-2)
		if err != nil {
			return fmt.Errorf("%s/%s train: %w", name, f, err)
		}
		_ = first
		_ = last
		fmt.Printf("(%s/%s ok) ", name, f)
	}
	return nil
}

func peakLayersBlock() error {
	// SwiGLU
	if err := placeAndRun("swiglu", quant.FormatQ8_0, func(g *architecture.Grid, format quant.Format) (*core.Tensor[float32], *core.Tensor[float32], error) {
		cfg := swiglu.Config{InputDim: 64, IntermediateDim: 128}
		in, inter := cfg.InputDim, cfg.IntermediateDim
		l, err := swiglu.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, eye(inter, in), eye(inter, in), eye(in, inter))
		if err != nil {
			return nil, nil, err
		}
		if format != quant.FormatNone {
			if err := l.Pack(format); err != nil {
				return nil, nil, err
			}
		}
		if err := swiglu.Place(g, 0, 0, 0, 0, l); err != nil {
			return nil, nil, err
		}
		x := core.NewTensor[float32](2, in)
		y := core.NewTensor[float32](2, in)
		for i := range x.Data {
			x.Data[i] = float32((i%5)+1) * 0.1
			y.Data[i] = float32((i%3)+1) * 0.05
		}
		return x, y, nil
	}); err != nil {
		return err
	}

	// MHA DecoderCausal
	if err := placeAndRun("mha", quant.FormatQ8_0, func(g *architecture.Grid, format quant.Format) (*core.Tensor[float32], *core.Tensor[float32], error) {
		cfg := mha.DecoderCausal(64, 4, 4)
		cfg.MaxSeqLen = 16
		if err := cfg.Validate(); err != nil {
			return nil, nil, err
		}
		d, qDim, kvDim := cfg.DModel, cfg.QDim(), cfg.KVDim()
		l, err := mha.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone,
			eye(qDim, d), eye(kvDim, d), eye(kvDim, d), eye(d, qDim))
		if err != nil {
			return nil, nil, err
		}
		if format != quant.FormatNone {
			if err := l.Pack(format); err != nil {
				return nil, nil, err
			}
		}
		if err := mha.Place(g, 0, 0, 0, 0, l); err != nil {
			return nil, nil, err
		}
		x := core.NewTensor[float32](1, 4, d)
		y := core.NewTensor[float32](1, 4, d)
		for i := range x.Data {
			x.Data[i] = float32((i%5)+1) * 0.1
			y.Data[i] = float32((i%3)+1) * 0.05
		}
		return x, y, nil
	}); err != nil {
		return err
	}

	// CNN1
	if err := placeAndRun("cnn1", quant.FormatQ8_0, func(g *architecture.Grid, format quant.Format) (*core.Tensor[float32], *core.Tensor[float32], error) {
		cfg := cnn1.Config{
			InChannels: 64, Filters: 64, SeqLen: 8, Kernel: 1, Stride: 1, Padding: 0,
			Activation: core.ActivationLinear,
		}
		w := make([]float32, cfg.Filters*cfg.InChannels*cfg.Kernel)
		for i := range w {
			w[i] = float32((i%5)-2) * 0.05
		}
		l, err := cnn1.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, nil, err
		}
		if format != quant.FormatNone {
			if err := l.Pack(format); err != nil {
				return nil, nil, err
			}
		}
		if err := cnn1.Place(g, 0, 0, 0, 0, l); err != nil {
			return nil, nil, err
		}
		x := core.NewTensor[float32](2, cfg.InChannels, cfg.SeqLen)
		// output same shape for K=1 shape-preserving
		y := core.NewTensor[float32](2, cfg.Filters, cfg.SeqLen)
		for i := range x.Data {
			x.Data[i] = float32((i%5)+1) * 0.1
		}
		for i := range y.Data {
			y.Data[i] = float32((i%3)+1) * 0.05
		}
		return x, y, nil
	}); err != nil {
		return err
	}

	// RNN
	if err := placeAndRun("rnn", quant.FormatQ8_0, func(g *architecture.Grid, format quant.Format) (*core.Tensor[float32], *core.Tensor[float32], error) {
		cfg := rnn.Config{InputSize: 64, HiddenSize: 64, SeqLen: 4}
		w := make([]float32, cfg.WeightCount())
		for i := range w {
			w[i] = float32((i%5)-2) * 0.02
		}
		l, err := rnn.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, nil, err
		}
		if format != quant.FormatNone {
			if err := l.Pack(format); err != nil {
				return nil, nil, err
			}
		}
		if err := rnn.Place(g, 0, 0, 0, 0, l); err != nil {
			return nil, nil, err
		}
		x := core.NewTensor[float32](2, cfg.SeqLen, cfg.InputSize)
		y := core.NewTensor[float32](2, cfg.SeqLen, cfg.HiddenSize)
		for i := range x.Data {
			x.Data[i] = float32((i%5)+1) * 0.1
		}
		for i := range y.Data {
			y.Data[i] = float32((i%3)+1) * 0.05
		}
		return x, y, nil
	}); err != nil {
		return err
	}

	// LSTM
	if err := placeAndRun("lstm", quant.FormatQ8_0, func(g *architecture.Grid, format quant.Format) (*core.Tensor[float32], *core.Tensor[float32], error) {
		cfg := lstm.Config{InputSize: 64, HiddenSize: 64, SeqLen: 4}
		w := make([]float32, cfg.WeightCount())
		for i := range w {
			w[i] = float32((i%5)-2) * 0.02
		}
		l, err := lstm.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, nil, err
		}
		if format != quant.FormatNone {
			if err := l.Pack(format); err != nil {
				return nil, nil, err
			}
		}
		if err := lstm.Place(g, 0, 0, 0, 0, l); err != nil {
			return nil, nil, err
		}
		x := core.NewTensor[float32](2, cfg.SeqLen, cfg.InputSize)
		y := core.NewTensor[float32](2, cfg.SeqLen, cfg.HiddenSize)
		for i := range x.Data {
			x.Data[i] = float32((i%5)+1) * 0.1
		}
		for i := range y.Data {
			y.Data[i] = float32((i%3)+1) * 0.05
		}
		return x, y, nil
	}); err != nil {
		return err
	}

	// Embedding
	if err := placeAndRun("embedding", quant.FormatQ8_0, func(g *architecture.Grid, format quant.Format) (*core.Tensor[float32], *core.Tensor[float32], error) {
		cfg := embedding.Config{VocabSize: 64, EmbeddingDim: 64, SeqLen: 4}
		w := make([]float32, cfg.WeightCount())
		for i := range w {
			w[i] = float32((i%5)-2) * 0.05
		}
		l, err := embedding.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, w)
		if err != nil {
			return nil, nil, err
		}
		if format != quant.FormatNone {
			if err := l.Pack(format); err != nil {
				return nil, nil, err
			}
		}
		if err := embedding.Place(g, 0, 0, 0, 0, l); err != nil {
			return nil, nil, err
		}
		x := core.NewTensor[float32](2, cfg.SeqLen)
		for i := range x.Data {
			x.Data[i] = float32(i % cfg.VocabSize)
		}
		y := core.NewTensor[float32](2, cfg.SeqLen, cfg.EmbeddingDim)
		for i := range y.Data {
			y.Data[i] = 0.01
		}
		return x, y, nil
	}); err != nil {
		return err
	}

	return nil
}

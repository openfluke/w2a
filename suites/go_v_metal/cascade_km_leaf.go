package go_v_metal

import (
	"bytes"
	"encoding/binary"
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

type kmNBPLeaf struct {
	batch, seq int
	lr         float64
	cfg        kmeans.Config
	dtype      core.DType
	W, x, target []float32
}

func smokeKmProbsCfg() kmeans.Config {
	return kmeans.Config{
		NumClusters: 4, FeatureDim: 6, Temperature: 1,
		OutputMode: kmeans.OutputProbabilities, Activation: core.ActivationLinear,
	}
}

func kmOutLen(batch, seq int, cfg kmeans.Config) int {
	if seq <= 0 {
		seq = 1
	}
	return batch * seq * cfg.OutDim()
}

func disableKMeansTrain(l *kmeans.Layer) {
	be := core.BackendCPUTiled
	l.Exec.Backend = be
	l.Exec.MultiCore = false
	l.Core.MultiCore = false
	if l.Centers != nil {
		l.Centers.Exec.MultiCore = false
		l.Centers.Core.MultiCore = false
		l.Centers.Exec.Backend = be
	}
}

func packKMeansWeightsF32(l *kmeans.Layer) ([]float32, error) {
	if l == nil || l.Centers == nil {
		return nil, fmt.Errorf("kmeans: nil layer")
	}
	return weights.MatrixF32(l.Centers.Weights)
}

func makeKmLeafCase(cfg kmeans.Config, batch, seq int, dt core.DType, seed uint32) (kmNBPLeaf, error) {
	if err := cfg.Validate(); err != nil {
		return kmNBPLeaf{}, err
	}
	if seq <= 0 {
		seq = 1
	}
	fc := kmNBPLeaf{
		batch: batch, seq: seq, cfg: cfg, dtype: dt, lr: 1e-2,
		W: randInit(cfg.NumClusters, cfg.FeatureDim, seed),
		x: randInit(batch*seq, cfg.FeatureDim, seed^0x9e3779b9),
	}
	fc.target = make([]float32, kmOutLen(batch, seq, cfg))
	for i := range fc.target {
		fc.target[i] = float32(0.05 * float64(i%7+1))
	}
	return fc, nil
}

// writeKmChild encodes KMeans (kind 22) for STB1/TWB1/FRB1… read_child.
// Field order matches dense_fwd Kind::KMeans: k,d,seq,output_mode,act,temperature,dtype,W[k*d].
func writeKmChild(buf *bytes.Buffer, cfg kmeans.Config, seq int, dtype core.DType, W []float32) {
	_ = binary.Write(buf, binary.LittleEndian, int32(22)) // KMeans
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.NumClusters))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.FeatureDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, outputModeInt(cfg.OutputMode))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.Activation))
	_ = binary.Write(buf, binary.LittleEndian, cfg.Temperature)
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
}

func makeKmModeLeaf(dtype core.DType, seed uint32) (cfg kmeans.Config, batch, seq int, W, x, tgt []float32, lr float64, err error) {
	cfg = smokeKmProbsCfg() // OutputProbabilities + Linear cascade smoke
	batch, seq = 2, 3
	fc, err := makeKmLeafCase(cfg, batch, seq, dtype, seed)
	if err != nil {
		return cfg, 0, 0, nil, nil, nil, 0, err
	}
	return cfg, batch, seq, fc.W, fc.x, fc.target, fc.lr, nil
}

func newKmStack(cfg kmeans.Config, dtype core.DType, W []float32) (*parallel.Stack, error) {
	l, err := kmeans.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return nil, err
	}
	disableKMeansTrain(l)
	return parallel.NewStack(l)
}

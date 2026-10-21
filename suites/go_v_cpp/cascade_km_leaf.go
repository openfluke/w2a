package go_v_cpp

import (
	"bytes"
	"encoding/binary"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
)

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

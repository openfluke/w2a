package go_v_cpp

import (
	"bytes"
	"encoding/binary"
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
)

// writeMcChild encodes Metacognition (kind 20) for STB1/TWB1/FRB1… read_child.
func writeMcChild(buf *bytes.Buffer, dim, seq int, dtype core.DType, W []float32) {
	_ = binary.Write(buf, binary.LittleEndian, int32(20)) // Metacognition
	_ = binary.Write(buf, binary.LittleEndian, int32(dim))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
}

// writeGdChild encodes GDN (kind 21) for read_child.
func writeGdChild(buf *bytes.Buffer, cfg gdn.Config, seq int, W []float32) {
	_ = binary.Write(buf, binary.LittleEndian, int32(21)) // GDN
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.HiddenSize))
	_ = binary.Write(buf, binary.LittleEndian, int32(seq))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.NumKeyHeads))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.NumValueHeads))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.KeyHeadDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.ValueHeadDim))
	_ = binary.Write(buf, binary.LittleEndian, int32(cfg.ConvKernel))
	_ = binary.Write(buf, binary.LittleEndian, int32(core.DTypeFloat32))
	for _, v := range W {
		_ = binary.Write(buf, binary.LittleEndian, v)
	}
}

// writeNestedStackDenseChild encodes NestedStack (kind 19) of n Dense Dim→Dim Linear.
func writeNestedStackDenseChild(buf *bytes.Buffer, dim, nChildren int, dtype core.DType, W []float32) error {
	need := stWeightCount(dim, nChildren)
	if len(W) < need {
		return fmt.Errorf("nested stack W short need %d have %d", need, len(W))
	}
	_ = binary.Write(buf, binary.LittleEndian, int32(19)) // NestedStack
	_ = binary.Write(buf, binary.LittleEndian, int32(nChildren))
	off := 0
	for i := 0; i < nChildren; i++ {
		nw := dim * dim
		_ = binary.Write(buf, binary.LittleEndian, int32(1)) // Dense
		_ = binary.Write(buf, binary.LittleEndian, int32(dim))
		_ = binary.Write(buf, binary.LittleEndian, int32(dim))
		_ = binary.Write(buf, binary.LittleEndian, int32(core.ActivationLinear))
		_ = binary.Write(buf, binary.LittleEndian, int32(dtype))
		for _, v := range W[off : off+nw] {
			_ = binary.Write(buf, binary.LittleEndian, v)
		}
		_ = binary.Write(buf, binary.LittleEndian, uint32(0)) // bias_len
		off += nw
	}
	return nil
}

func makeMcModeLeaf(dtype core.DType, seed uint32) (cfg metacognition.Config, batch, seq int, W, x, tgt []float32, lr float64, err error) {
	cfg = smokeMcCfg()
	batch, seq = 2, cfg.SeqLen
	fc, err := makeMcLeafCase(cfg, batch, seq, dtype, seed)
	if err != nil {
		return cfg, 0, 0, nil, nil, nil, 0, err
	}
	return cfg, batch, seq, fc.W, fc.x, fc.target, fc.lr, nil
}

func makeGdModeLeaf(seed uint32) (cfg gdn.Config, batch, seq int, W, x, tgt []float32, lr float64, err error) {
	cfg = smokeGdCfg()
	batch, seq = 2, 4
	fc, err := makeGdLeafCase(cfg, batch, seq, seed)
	if err != nil {
		return cfg, 0, 0, nil, nil, nil, 0, err
	}
	return cfg, batch, seq, fc.W, fc.x, fc.target, fc.lr, nil
}

func makeStModeLeaf(dtype core.DType, seed uint32) (dim, nChildren, batch, seq int, W, x, tgt []float32, lr float64, err error) {
	dim, nChildren, seq = 4, 2, 1 // match stack_train NestedStack-of-Dense geometry (small)
	batch = 2
	fc, err := makeStLeafCase(dim, nChildren, batch, seq, dtype, seed)
	if err != nil {
		return 0, 0, 0, 0, nil, nil, nil, 0, err
	}
	return dim, nChildren, batch, seq, fc.W, fc.x, fc.target, fc.lr, nil
}

func newMcStack(cfg metacognition.Config, dtype core.DType, W []float32) (*parallel.Stack, error) {
	l, err := metacognition.NewConfigured(cfg, dtype, quant.FormatNone, W)
	if err != nil {
		return nil, err
	}
	disableMetacognitionTrain(l)
	return parallel.NewStack(l)
}

func newGdStack(cfg gdn.Config, W []float32) (*parallel.Stack, error) {
	inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma, err := splitGdnPacked(cfg, W)
	if err != nil {
		return nil, err
	}
	l, err := gdn.NewConfigured(cfg, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma)
	if err != nil {
		return nil, err
	}
	disableGDNTrain(l)
	return parallel.NewStack(l)
}

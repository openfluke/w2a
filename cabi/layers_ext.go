package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"
	"strings"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/gdn"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

func parsePlaceJSON(cfgJSON *C.char) (placeSpec, []byte, error) {
	raw := []byte(C.GoString(cfgJSON))
	var s placeSpec
	if err := json.Unmarshal(raw, &s); err != nil {
		return placeSpec{}, nil, err
	}
	if s.Dim == 0 && s.In > 0 {
		s.Dim = s.In
	}
	if s.Out == 0 && s.Dim > 0 {
		s.Out = s.Dim
	}
	return s, raw, nil
}

func dtOf(s placeSpec) core.DType {
	if s.DType == 0 {
		return core.DTypeFloat32
	}
	return core.DType(s.DType)
}

func fmtOf(s placeSpec) quant.Format {
	return quant.Format(s.Format)
}

func actOf(s placeSpec) core.ActivationType {
	if s.Act == "" {
		return core.ActivationReLU
	}
	return core.ParseActivation(s.Act)
}

func placeOnGrid(g *architecture.Grid, kind string, cfgJSON *C.char) *C.char {
	s, raw, err := parsePlaceJSON(cfgJSON)
	if err != nil {
		return errJSON(err.Error())
	}
	z, y, x, l := s.Z, s.Y, s.X, s.L
	dt, format := dtOf(s), fmtOf(s)
	kind = strings.ToLower(strings.TrimSpace(kind))

	switch kind {
	case "dense":
		in, out := s.In, s.Out
		if in <= 0 {
			in = s.Dim
		}
		if out <= 0 {
			out = in
		}
		if in <= 0 {
			return errJSON("dense needs in/out or dim")
		}
		layer, err := dense.NewConfigured[float32](in, out, actOf(s), dt, format, nil)
		if err != nil {
			return errJSON(err.Error())
		}
		if err := dense.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "mha":
		var cfg mha.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.DModel == 0 {
			cfg.DModel = s.Dim
		}
		if cfg.NumHeads == 0 {
			cfg.NumHeads = 4
		}
		layer, err := mha.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := mha.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "swiglu":
		var cfg swiglu.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InputDim == 0 {
			cfg.InputDim = s.Dim
		}
		if cfg.IntermediateDim == 0 {
			cfg.IntermediateDim = cfg.InputDim * 2
			if cfg.IntermediateDim == 0 {
				cfg.IntermediateDim = 16
			}
		}
		layer, err := swiglu.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := swiglu.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "rmsnorm":
		var cfg rmsnorm.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.Dim == 0 {
			cfg.Dim = s.Dim
		}
		layer, err := rmsnorm.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := rmsnorm.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "layernorm":
		var cfg layernorm.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.Dim == 0 {
			cfg.Dim = s.Dim
		}
		layer, err := layernorm.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := layernorm.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "embedding":
		var cfg embedding.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.EmbeddingDim == 0 {
			cfg.EmbeddingDim = s.Dim
		}
		if cfg.VocabSize == 0 {
			cfg.VocabSize = 256
		}
		if cfg.SeqLen == 0 {
			cfg.SeqLen = 8
		}
		layer, err := embedding.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := embedding.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "softmax":
		var cfg softmax.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.Dim == 0 {
			cfg.Dim = s.Dim
		}
		layer, err := softmax.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		if err := softmax.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "sequential":
		var cfg sequential.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.Dim == 0 {
			cfg.Dim = s.Dim
		}
		if cfg.Depth == 0 {
			cfg.Depth = 2
		}
		layer, err := sequential.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := sequential.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "residual":
		var cfg residual.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.Dim == 0 {
			cfg.Dim = s.Dim
		}
		if cfg.Depth == 0 {
			cfg.Depth = 1
		}
		layer, err := residual.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := residual.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "cnn1":
		var cfg cnn1.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InChannels == 0 {
			cfg.InChannels = 1
		}
		if cfg.Filters == 0 {
			cfg.Filters = 4
		}
		if cfg.SeqLen == 0 {
			cfg.SeqLen = s.Dim
			if cfg.SeqLen == 0 {
				cfg.SeqLen = 16
			}
		}
		if cfg.Kernel == 0 {
			cfg.Kernel = 3
		}
		layer, err := cnn1.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := cnn1.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "cnn2":
		var cfg cnn2.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InChannels == 0 {
			cfg.InChannels = 1
		}
		if cfg.Filters == 0 {
			cfg.Filters = 4
		}
		if cfg.Height == 0 {
			cfg.Height = 8
		}
		if cfg.Width == 0 {
			cfg.Width = 8
		}
		if cfg.Kernel == 0 {
			cfg.Kernel = 3
		}
		layer, err := cnn2.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := cnn2.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "cnn3":
		var cfg cnn3.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InChannels == 0 {
			cfg.InChannels = 1
		}
		if cfg.Filters == 0 {
			cfg.Filters = 2
		}
		if cfg.Depth == 0 {
			cfg.Depth = 4
		}
		if cfg.Height == 0 {
			cfg.Height = 4
		}
		if cfg.Width == 0 {
			cfg.Width = 4
		}
		if cfg.Kernel == 0 {
			cfg.Kernel = 3
		}
		layer, err := cnn3.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := cnn3.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "rnn":
		var cfg rnn.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InputSize == 0 {
			cfg.InputSize = s.Dim
		}
		if cfg.HiddenSize == 0 {
			cfg.HiddenSize = s.Dim
		}
		if cfg.SeqLen == 0 {
			cfg.SeqLen = 8
		}
		layer, err := rnn.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := rnn.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "lstm":
		var cfg lstm.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InputSize == 0 {
			cfg.InputSize = s.Dim
		}
		if cfg.HiddenSize == 0 {
			cfg.HiddenSize = s.Dim
		}
		if cfg.SeqLen == 0 {
			cfg.SeqLen = 8
		}
		layer, err := lstm.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := lstm.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "convt1":
		var cfg convt1.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InChannels == 0 {
			cfg.InChannels = 4
		}
		if cfg.Filters == 0 {
			cfg.Filters = 2
		}
		if cfg.SeqLen == 0 {
			cfg.SeqLen = 8
		}
		if cfg.Kernel == 0 {
			cfg.Kernel = 3
		}
		layer, err := convt1.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := convt1.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "convt2":
		var cfg convt2.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InChannels == 0 {
			cfg.InChannels = 4
		}
		if cfg.Filters == 0 {
			cfg.Filters = 2
		}
		if cfg.Height == 0 {
			cfg.Height = 4
		}
		if cfg.Width == 0 {
			cfg.Width = 4
		}
		if cfg.Kernel == 0 {
			cfg.Kernel = 3
		}
		layer, err := convt2.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := convt2.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "convt3":
		var cfg convt3.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.InChannels == 0 {
			cfg.InChannels = 2
		}
		if cfg.Filters == 0 {
			cfg.Filters = 2
		}
		if cfg.Depth == 0 {
			cfg.Depth = 4
		}
		if cfg.Height == 0 {
			cfg.Height = 4
		}
		if cfg.Width == 0 {
			cfg.Width = 4
		}
		if cfg.Kernel == 0 {
			cfg.Kernel = 3
		}
		layer, err := convt3.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := convt3.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "parallel":
		var cfg parallel.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.Dim == 0 {
			cfg.Dim = s.Dim
		}
		if cfg.OutFeat == 0 {
			cfg.OutFeat = cfg.Dim
		}
		if cfg.Branches == 0 {
			cfg.Branches = 2
		}
		if cfg.Combine == "" {
			cfg.Combine = parallel.CombineAdd
		}
		layer, err := parallel.NewConfigured[float32](cfg, dt, format, nil, nil)
		if err != nil {
			return errJSON(err.Error())
		}
		if err := parallel.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "stack":
		dim := s.Dim
		if dim == 0 {
			dim = 8
		}
		st, err := parallel.Bicameral(dim, dim, dim, actOf(s), dt, format)
		if err != nil {
			return errJSON(err.Error())
		}
		if err := parallel.PlaceStack(g, z, y, x, l, st); err != nil {
			return errJSON(err.Error())
		}
	case "kmeans":
		var cfg kmeans.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.FeatureDim == 0 {
			cfg.FeatureDim = s.Dim
		}
		if cfg.FeatureDim == 0 {
			cfg.FeatureDim = 8
		}
		if cfg.NumClusters == 0 {
			cfg.NumClusters = 4
		}
		layer, err := kmeans.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := kmeans.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "mamba":
		var cfg mamba.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.DModel == 0 {
			cfg.DModel = s.Dim
		}
		if cfg.DModel == 0 {
			cfg.DModel = 8
		}
		if cfg.DState == 0 {
			cfg.DState = 8
		}
		if cfg.SeqLen == 0 {
			cfg.SeqLen = 4
		}
		layer, err := mamba.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := mamba.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "metacognition":
		var cfg metacognition.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.Dim == 0 {
			cfg.Dim = s.Dim
		}
		if cfg.Dim == 0 {
			cfg.Dim = 8
		}
		layer, err := metacognition.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		_ = layer.SetDType(dt)
		_ = layer.Pack(format)
		if err := metacognition.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	case "gdn":
		var cfg gdn.Config
		_ = json.Unmarshal(raw, &cfg)
		if cfg.HiddenSize == 0 {
			cfg.HiddenSize = s.Dim
		}
		if cfg.HiddenSize == 0 {
			cfg.HiddenSize = 8
		}
		if cfg.NumKeyHeads == 0 {
			cfg.NumKeyHeads = 2
		}
		if cfg.NumValueHeads == 0 {
			cfg.NumValueHeads = 2
		}
		if cfg.KeyHeadDim == 0 {
			cfg.KeyHeadDim = 4
		}
		if cfg.ValueHeadDim == 0 {
			cfg.ValueHeadDim = 4
		}
		if cfg.ConvKernel == 0 {
			cfg.ConvKernel = 3
		}
		layer, err := gdn.New(cfg)
		if err != nil {
			return errJSON(err.Error())
		}
		if err := gdn.Place(g, z, y, x, l, layer); err != nil {
			return errJSON(err.Error())
		}
	default:
		return errJSON("unknown layer kind " + kind)
	}
	return okJSON()
}

func placeExport(handle C.longlong, kind string, cfgJSON *C.char) *C.char {
	g, ok := getGrid(int64(handle))
	if !ok {
		return errJSON("invalid grid handle")
	}
	return placeOnGrid(g, kind, cfgJSON)
}

//export WelvetPlace
func WelvetPlace(handle C.longlong, kind *C.char, cfgJSON *C.char) *C.char {
	return placeExport(handle, C.GoString(kind), cfgJSON)
}

//export WelvetPlaceMHA
func WelvetPlaceMHA(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "mha", cfgJSON)
}

//export WelvetPlaceSwiGLU
func WelvetPlaceSwiGLU(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "swiglu", cfgJSON)
}

//export WelvetPlaceRMSNorm
func WelvetPlaceRMSNorm(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "rmsnorm", cfgJSON)
}

//export WelvetPlaceLayerNorm
func WelvetPlaceLayerNorm(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "layernorm", cfgJSON)
}

//export WelvetPlaceEmbedding
func WelvetPlaceEmbedding(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "embedding", cfgJSON)
}

//export WelvetPlaceSoftmax
func WelvetPlaceSoftmax(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "softmax", cfgJSON)
}

//export WelvetPlaceSequential
func WelvetPlaceSequential(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "sequential", cfgJSON)
}

//export WelvetPlaceResidual
func WelvetPlaceResidual(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "residual", cfgJSON)
}

//export WelvetPlaceCNN1
func WelvetPlaceCNN1(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "cnn1", cfgJSON)
}

//export WelvetPlaceCNN2
func WelvetPlaceCNN2(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "cnn2", cfgJSON)
}

//export WelvetPlaceCNN3
func WelvetPlaceCNN3(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "cnn3", cfgJSON)
}

//export WelvetPlaceRNN
func WelvetPlaceRNN(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "rnn", cfgJSON)
}

//export WelvetPlaceLSTM
func WelvetPlaceLSTM(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "lstm", cfgJSON)
}

//export WelvetPlaceConvT1
func WelvetPlaceConvT1(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "convt1", cfgJSON)
}

//export WelvetPlaceConvT2
func WelvetPlaceConvT2(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "convt2", cfgJSON)
}

//export WelvetPlaceConvT3
func WelvetPlaceConvT3(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "convt3", cfgJSON)
}

//export WelvetPlaceParallel
func WelvetPlaceParallel(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "parallel", cfgJSON)
}

//export WelvetPlaceStack
func WelvetPlaceStack(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "stack", cfgJSON)
}

//export WelvetPlaceKMeans
func WelvetPlaceKMeans(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "kmeans", cfgJSON)
}

//export WelvetPlaceMamba
func WelvetPlaceMamba(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "mamba", cfgJSON)
}

//export WelvetPlaceMetacognition
func WelvetPlaceMetacognition(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "metacognition", cfgJSON)
}

//export WelvetPlaceGDN
func WelvetPlaceGDN(handle C.longlong, cfgJSON *C.char) *C.char {
	return placeExport(handle, "gdn", cfgJSON)
}

//export WelvetPermutationOK
func WelvetPermutationOK(kind *C.char, dtype, format, backend C.int) C.int {
	dt := core.DType(dtype)
	fmt := quant.Format(format)
	be := core.Backend(backend)
	k := strings.ToLower(C.GoString(kind))
	ok := false
	switch k {
	case "gdn":
		ok = gdn.PermutationOK(dt, fmt, be)
	default:
		ok = dense.PermutationOK(dt, fmt, be)
	}
	if ok {
		return 1
	}
	return 0
}

package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
)

//export WelvetListNamedTrainModes
func WelvetListNamedTrainModes() *C.char {
	modes := parallel.AllTrainModes()
	names := make([]string, 0, len(modes))
	for _, m := range modes {
		names = append(names, m.String())
	}
	return jsonOut(names)
}

//export WelvetListConcreteTrainModes
func WelvetListConcreteTrainModes() *C.char {
	modes := parallel.AllConcreteTrainModes()
	names := make([]string, 0, len(modes))
	for _, m := range modes {
		names = append(names, m.String())
	}
	return jsonOut(names)
}

//export WelvetCreateBicameral
func WelvetCreateBicameral(cfgJSON *C.char) C.longlong {
	var cfg struct {
		In     int `json:"in"`
		Hidden int `json:"hidden"`
		Out    int `json:"out"`
		DType  int `json:"dtype"`
		Format int `json:"format"`
	}
	if err := json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg); err != nil {
		return 0
	}
	if cfg.In < 1 {
		cfg.In = 4
	}
	if cfg.Hidden < 1 {
		cfg.Hidden = cfg.In
	}
	if cfg.Out < 1 {
		cfg.Out = cfg.In
	}
	dt := core.DTypeFloat32
	if cfg.DType != 0 {
		dt = core.DType(cfg.DType)
	}
	st, err := parallel.Bicameral(cfg.In, cfg.Hidden, cfg.Out, core.ActivationReLU, dt, quant.Format(cfg.Format))
	if err != nil {
		return 0
	}
	return C.longlong(storeStack(st))
}

//export WelvetCreateParallel
func WelvetCreateParallel(cfgJSON *C.char) C.longlong {
	var cfg parallel.Config
	_ = json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg)
	if cfg.Dim < 1 {
		cfg.Dim = 8
	}
	if cfg.OutFeat < 1 {
		cfg.OutFeat = cfg.Dim
	}
	if cfg.Branches < 1 {
		cfg.Branches = 2
	}
	if cfg.Combine == "" {
		cfg.Combine = parallel.CombineAdd
	}
	var wrap struct {
		DType  int `json:"dtype"`
		Format int `json:"format"`
	}
	_ = json.Unmarshal([]byte(C.GoString(cfgJSON)), &wrap)
	dt := core.DTypeFloat32
	if wrap.DType != 0 {
		dt = core.DType(wrap.DType)
	}
	layer, err := parallel.NewConfigured[float32](cfg, dt, quant.Format(wrap.Format), nil, nil)
	if err != nil {
		return 0
	}
	return C.longlong(storeParallel(layer))
}

//export WelvetSetChildModes
func WelvetSetChildModes(handle C.longlong, modesJSON *C.char) *C.char {
	st, ok := getStack(int64(handle))
	if !ok {
		return errJSON("invalid stack handle")
	}
	var names []string
	if err := json.Unmarshal([]byte(C.GoString(modesJSON)), &names); err != nil {
		return errJSON(err.Error())
	}
	modes := make([]parallel.TrainMode, len(names))
	for i, n := range names {
		m, err := parallel.ParseTrainMode(n)
		if err != nil {
			return errJSON(err.Error())
		}
		modes[i] = m
	}
	st.SetChildModes(modes...)
	return okJSON()
}

//export WelvetSetBranchModes
func WelvetSetBranchModes(handle C.longlong, modesJSON *C.char) *C.char {
	p, ok := getParallel(int64(handle))
	if !ok {
		return errJSON("invalid parallel handle")
	}
	var names []string
	if err := json.Unmarshal([]byte(C.GoString(modesJSON)), &names); err != nil {
		return errJSON(err.Error())
	}
	modes := make([]parallel.TrainMode, len(names))
	for i, n := range names {
		m, err := parallel.ParseTrainMode(n)
		if err != nil {
			return errJSON(err.Error())
		}
		modes[i] = m
	}
	p.SetBranchModes(modes...)
	return okJSON()
}

//export WelvetFreeStack
func WelvetFreeStack(handle C.longlong) {
	mu.Lock()
	defer mu.Unlock()
	delete(stacks, int64(handle))
}

//export WelvetTrainStackMSE
func WelvetTrainStackMSE(handle C.longlong, inPtr *C.float, inLen C.int, tgtPtr *C.float, tgtLen C.int, modeName *C.char, lr C.double) *C.char {
	st, ok := getStack(int64(handle))
	if !ok {
		return errJSON("invalid stack handle")
	}
	mode, err := parallel.ParseTrainMode(C.GoString(modeName))
	if err != nil {
		return errJSON(err.Error())
	}
	x := make([]float32, int(inLen))
	y := make([]float32, int(tgtLen))
	copy(x, readF32(inPtr, int(inLen)))
	copy(y, readF32(tgtPtr, int(tgtLen)))
	xin := core.NewTensor[float32](1, len(x))
	yin := core.NewTensor[float32](1, len(y))
	copy(xin.Data, x)
	copy(yin.Data, y)
	loss, err := parallel.TrainStackMSE(st, xin, yin, mode, float64(lr))
	if err != nil {
		return errJSON(err.Error())
	}
	return jsonOut(map[string]any{"status": "ok", "loss": loss, "mode": mode.String()})
}

//export WelvetCreateHemispheres
func WelvetCreateHemispheres(cfgJSON *C.char) C.longlong {
	var cfg struct {
		Dim     int    `json:"dim"`
		N       int    `json:"n"`
		Combine string `json:"combine"`
		DType   int    `json:"dtype"`
	}
	if err := json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg); err != nil {
		return 0
	}
	if cfg.Dim < 1 {
		cfg.Dim = 4
	}
	if cfg.N < 1 {
		cfg.N = 2
	}
	combine := parallel.CombineAdd
	switch cfg.Combine {
	case "avg", "mean", "Avg", "Mean":
		combine = parallel.CombineAvg
	case "concat", "Concat":
		combine = parallel.CombineConcat
	}
	dt := core.DTypeFloat32
	if cfg.DType != 0 {
		dt = core.DType(cfg.DType)
	}
	p, err := parallel.Hemispheres(cfg.Dim, cfg.Dim, cfg.N, combine, core.ActivationReLU, dt, quant.FormatNone)
	if err != nil {
		return 0
	}
	return C.longlong(storeParallel(p))
}

//export WelvetFreeParallel
func WelvetFreeParallel(handle C.longlong) {
	mu.Lock()
	defer mu.Unlock()
	delete(parallels, int64(handle))
}

//export WelvetTrainParallelMSE
func WelvetTrainParallelMSE(handle C.longlong, inPtr *C.float, inLen C.int, tgtPtr *C.float, tgtLen C.int, modeName *C.char, lr C.double) *C.char {
	p, ok := getParallel(int64(handle))
	if !ok {
		return errJSON("invalid parallel handle")
	}
	mode, err := parallel.ParseTrainMode(C.GoString(modeName))
	if err != nil {
		return errJSON(err.Error())
	}
	xin := core.NewTensor[float32](1, int(inLen))
	yin := core.NewTensor[float32](1, int(tgtLen))
	copy(xin.Data, readF32(inPtr, int(inLen)))
	copy(yin.Data, readF32(tgtPtr, int(tgtLen)))
	loss, err := parallel.TrainMSE(p, xin, yin, mode, float64(lr))
	if err != nil {
		return errJSON(err.Error())
	}
	return jsonOut(map[string]any{"status": "ok", "loss": loss, "mode": mode.String()})
}

//export WelvetSetCamSync
func WelvetSetCamSync(handle C.longlong, cfgJSON *C.char) *C.char {
	p, ok := getParallel(int64(handle))
	if !ok {
		return errJSON("invalid parallel handle")
	}
	var cfg parallel.CamSyncConfig
	if err := json.Unmarshal([]byte(C.GoString(cfgJSON)), &cfg); err != nil {
		return errJSON(err.Error())
	}
	p.SetCamSync(cfg)
	return okJSON()
}

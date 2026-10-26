package go_v_metal

import (
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

func proxyVariant(mode parallel.TrainMode) int {
	if mode >= parallel.ModeTweenSplitHeadProxy && mode <= parallel.ModeTweenSplitSparse {
		return int(mode-parallel.ModeTweenSplitHeadProxy) + 1
	}
	return 0
}
func stepProxyVariant(mode parallel.TrainMode) int {
	if mode >= parallel.ModeStepTweenSplitHeadProxy && mode <= parallel.ModeStepTweenSplitSparse {
		return int(mode-parallel.ModeStepTweenSplitHeadProxy) + 1
	}
	return 0
}
func proxyEpochs(mode parallel.TrainMode) int {
	switch mode {
	case parallel.ModeTweenSplitLinearCache:
		return 3
	case parallel.ModeTweenSplitHeadProxyAsync, parallel.ModeTweenSplitSparse:
		return 4
	}
	if proxyVariant(mode) > 0 {
		return 2
	}
	return 1
}
func proxyFamilyFixture(pc polyCase, dt core.DType) (kpModeFixture, error) {
	brs, x, _, shape, tshape, err := pc.make(dt)
	if err != nil {
		return kpModeFixture{}, err
	}
	if len(tshape) > 3 {
		ops := []any{brs[0].op, brs[1].op}
		for _, op := range ops {
			if _, err = meshMeta(op); err != nil {
				return kpModeFixture{}, err
			}
		}
		return kpModeFixture{ops: ops, wires: [][]byte{brs[0].bytes, brs[1].bytes}, xShape: shape, x: x}, nil
	}
	head, _, _, _, _, err := makeRMSPair(dt, 2, int(tshape[len(tshape)-1]))
	if err != nil {
		return kpModeFixture{}, err
	}
	ops := []any{brs[0].op, head[0].op}
	for _, op := range ops {
		if _, err = meshMeta(op); err != nil {
			return kpModeFixture{}, err
		}
	}
	return kpModeFixture{allowGDNStrict: pc.name == "gdn", ops: ops, wires: [][]byte{brs[0].bytes, head[0].bytes}, xShape: shape, x: x}, nil
}
func CasesProxyTrain(mode parallel.TrainMode) []Case {
	var cases []Case
	for _, pc := range polyTrainKinds() {
		pc := pc
		label := mode.String() + " " + pc.name + " ×34"
		if pc.name == "gdn" {
			label = mode.String() + " gdn Float32"
		}
		cases = append(cases, Case{Name: label, Run: func() error {
			dtypes := core.AllDTypes
			if pc.name == "gdn" {
				dtypes = []core.DType{core.DTypeFloat32}
			}
			for _, dt := range dtypes {
				if err := kpRunMode(mode, dt, func(dt core.DType) (kpModeFixture, error) { return proxyFamilyFixture(pc, dt) }); err != nil {
					return fmt.Errorf("%s: %w", dt, err)
				}
			}
			return nil
		}})
	}
	if mode == parallel.ModeTweenSplitLinearCache {
		for _, pc := range polyTrainKinds() {
			pc := pc
			cases = append(cases, Case{Name: "cache refresh epoch 20 " + pc.name + " Float32", Run: func() error {
				return kpRunMode(mode, core.DTypeFloat32, func(dt core.DType) (kpModeFixture, error) {
					f, e := proxyFamilyFixture(pc, dt)
					f.epochs = 22
					return f, e
				})
			}})
		}
	}
	return append(cases, CasesKMeansParallelModes(mode)...)
}

// The five remaining Line modes use ordinary forward shapes and a frozen
// popped gap; their shared volume matrix is the corresponding non-Step mode.
func CasesStepProxyTrain(mode parallel.TrainMode) []Case {
	cases := CasesProxyTrain(mode)
	for _, ticks := range []int{1, 2, 3, 4} {
		ticks := ticks
		cases = append(cases, Case{Name: fmt.Sprintf("pipeline pop %d ticks Parallel Filter Float32", ticks), Run: func() error {
			return kpRunMode(mode, core.DTypeFloat32, func(dt core.DType) (kpModeFixture, error) {
				f, e := kpParallelFixture(dt, parallel.CombineFilter)
				f.epochs = ticks
				return f, e
			})
		}})
	}
	if mode == parallel.ModeStepTweenSplitLinearCache {
		for _, combine := range []parallel.CombineMode{parallel.CombineConcat, parallel.CombineAdd, parallel.CombineAvg, parallel.CombineFilter, parallel.CombineMax, parallel.CombineSparseK, parallel.CombineDisagree} {
			combine := combine
			cases = append(cases, Case{Name: "Line cache refresh 26 ticks Parallel " + string(combine) + " Float32", Run: func() error {
				return kpRunMode(mode, core.DTypeFloat32, func(dt core.DType) (kpModeFixture, error) {
					f, e := kpParallelFixture(dt, combine)
					f.epochs = 26
					return f, e
				})
			}})
		}
	}
	return cases
}

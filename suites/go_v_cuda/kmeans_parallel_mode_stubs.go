package go_v_cuda

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/runtime/forward"
)

// Minimal helpers shared by kmeans_parallel_modes.go across TrainModes.
// Full Tween/proxy volume implementations land with those mode suites;
// ModeStepBP only needs the NormalBP volume path + these compile stubs.

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

type volApplyFwd func(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error)

type meshTensor struct {
	shape []int32
	data  []float32
}

type meshClockOut struct {
	loss      float64
	post, gin meshTensor
	grads     []meshTensor
	weights   [][]float32
	history   [][][]meshTensor // tick / cell / input, pre, post (MeshBP clock)
}

func compareProxyVolume(a, b meshClockOut) error {
	return fmt.Errorf("compareProxyVolume: proxy volume credit not implemented on go_v_cuda yet")
}

func applyModeTweenSplitVolume(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error) {
	return 0, fmt.Errorf("applyModeTweenSplitVolume: not implemented on go_v_cuda yet")
}

func applyModeTweenVolume(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error) {
	return 0, fmt.Errorf("applyModeTweenVolume: not implemented on go_v_cuda yet")
}

func runGoKmVolApply(depth, rows, cols, lpc, batch int, cells []kmVolCell, x, target []float32, lr float64, apply volApplyFwd, clearGin bool) (nbpVolOut, error) {
	_ = depth
	_ = rows
	_ = cols
	_ = lpc
	_ = batch
	_ = cells
	_ = x
	_ = target
	_ = lr
	_ = apply
	_ = clearGin
	return nbpVolOut{}, fmt.Errorf("runGoKmVolApply: not implemented on go_v_cuda yet")
}

func runGoPnVolApply(depth, rows, cols, lpc, batch int, cells []pnVolCell, x, target []float32, lr float64, apply volApplyFwd, clearGin bool) (nbpVolOut, error) {
	_ = depth
	_ = rows
	_ = cols
	_ = lpc
	_ = batch
	_ = cells
	_ = x
	_ = target
	_ = lr
	_ = apply
	_ = clearGin
	return nbpVolOut{}, fmt.Errorf("runGoPnVolApply: not implemented on go_v_cuda yet")
}

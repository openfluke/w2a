package go_v_rocm

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/runtime/forward"
)

// Temporary stubs so ModeStepBP can compile before Tween/Proxy volume helpers land.
// ModeStepBP fixtures do not call these paths.

func applyModeTweenVolume(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error) {
	_ = g
	_ = fwd
	_ = target
	_ = lr
	return 0, fmt.Errorf("applyModeTweenVolume: ModeTween volume helper not ported yet")
}

func applyModeTweenSplitVolume(g *architecture.Grid, fwd *forward.Result[float32], target *core.Tensor[float32], lr float64) (float64, error) {
	_ = g
	_ = fwd
	_ = target
	_ = lr
	return 0, fmt.Errorf("applyModeTweenSplitVolume: ModeTweenSplit volume helper not ported yet")
}

func compareProxyVolume(a, b meshClockOut) error {
	_ = a
	_ = b
	return fmt.Errorf("compareProxyVolume: proxy volume helper not ported yet")
}

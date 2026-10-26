package go_v_metal

import (
	"fmt"
	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/runtime/forward"
	"github.com/openfluke/welvet/runtime/step"
	"strings"
)

func meshGoLocalCredit(g *architecture.Grid, st *step.State[float32], gap *core.Tensor[float32], lr float64, mode parallel.TrainMode) error {
	if mode == parallel.ModeMeshTween {
		return meshGoTween(g, st, gap, lr)
	}
	if lr == 0 {
		return nil
	}
	last := len(st.HistoryIn) - 1
	coords := g.HopOrder()
	for i := len(g.Cells) - 1; i >= 0; i-- {
		c := &g.Cells[i]
		input, pre, post := st.HistoryIn[last][i], st.HistoryPre[last][i], st.LayerData[i]
		if c.Layer.IsDisabled || input == nil || pre == nil || post == nil {
			continue
		}
		fwd := &forward.Result[float32]{Grid: g, Output: post, Steps: []forward.Step[float32]{{Coord: coords[i], Input: input, Pre: pre, Post: post}}}
		leaves, err := proxyVolumeLeaves(g, fwd)
		if err != nil {
			return err
		}
		variant := 0
		if mode == parallel.ModeMeshTweenSplitFastProxy {
			variant = 3
		}
		if mode == parallel.ModeMeshTweenSplitSparse {
			variant = 6
		}
		if err = proxyApplyLeaves(leaves, gap, lr, variant, nil); err != nil {
			return err
		}
		if mode == parallel.ModeMeshTweenAlt {
			projected := projectGapF32(gap, post.Shape...)
			if err = applyTweenLocalOp(c.Op, projected, input, pre, lr*0.5); err != nil {
				return err
			}
		}
		next := core.NewTensor[float32](input.Shape...)
		if gap.Len() == next.Len() {
			copy(next.Data, gap.Data)
		}
		gap = next
	}
	return nil
}
func CasesMeshSplitMode(mode parallel.TrainMode) []Case {
	cases := meshTrainCases(mode)
	var parent []Case
	switch mode {
	case parallel.ModeMeshTweenSplit:
		parent = CasesTweenSplitTrain()
	case parallel.ModeMeshTweenAlt:
		parent = CasesTweenAltTrain()
	case parallel.ModeMeshTweenSplitFastProxy:
		parent = CasesProxyVolumes(parallel.ModeTweenSplitFastProxy)
	case parallel.ModeMeshTweenSplitSparse:
		parent = CasesProxyVolumes(parallel.ModeTweenSplitSparse)
	}
	for _, c := range parent {
		if strings.Contains(c.Name, "volume") {
			c.Name = "shared " + c.Name
			cases = append(cases, c)
		}
	}
	if len(cases) < 103 {
		panic(fmt.Sprintf("empty mesh mode %s", mode))
	}
	return cases
}

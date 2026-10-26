package go_v_metal

import (
	"fmt"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/parallel"
)

func CasesLineAltEdge() []Case {
	var out []Case
	for _, ticks := range []int{2, 3, 4, 5} {
		ticks := ticks
		out = append(out, Case{Name: fmt.Sprintf("KMeans Alt %d ticks", ticks), Run: func() error {
			return kpRunMode(parallel.ModeStepTweenAlt, core.DTypeFloat64, func(dt core.DType) (kpModeFixture, error) {
				f, e := kpKMeansFixture(dt, kmeans.OutputProbabilities)
				f.epochs = ticks
				return f, e
			})
		}})
	}
	for _, entry := range []struct {
		name    string
		dt      core.DType
		combine parallel.CombineMode
	}{{"add", core.DTypeFloat64, parallel.CombineAdd}, {"concat", core.DTypeUint32, parallel.CombineConcat}, {"disagree", core.DTypeUint32, parallel.CombineDisagree}} {
		entry := entry
		for _, ticks := range []int{2, 3, 4, 5} {
			ticks := ticks
			out = append(out, Case{Name: fmt.Sprintf("Parallel %s %d ticks", entry.name, ticks), Run: func() error {
				return kpRunMode(parallel.ModeStepTweenAlt, entry.dt, func(dt core.DType) (kpModeFixture, error) {
					f, e := kpParallelFixture(dt, entry.combine)
					f.epochs = ticks
					return f, e
				})
			}})
		}
	}
	return out
}

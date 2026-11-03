package go_v_rocm_test

import (
	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVRocmMeshSplitModes(t *testing.T) {
	for _, mode := range []parallel.TrainMode{
		parallel.ModeMeshTweenSplit,
		parallel.ModeMeshTweenAlt,
		parallel.ModeMeshTweenSplitFastProxy,
		parallel.ModeMeshTweenSplitSparse,
	} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvr.CasesMeshSplitMode(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

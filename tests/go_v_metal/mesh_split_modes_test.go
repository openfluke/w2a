package go_v_metal_test

import (
	gvm "github.com/openfluke/w2a/suites/go_v_metal"
	"github.com/openfluke/welvet/layers/parallel"
	"testing"
)

func TestGoVMetalMeshSplitModes(t *testing.T) {
	for _, mode := range []parallel.TrainMode{parallel.ModeMeshTweenSplit, parallel.ModeMeshTweenAlt, parallel.ModeMeshTweenSplitFastProxy, parallel.ModeMeshTweenSplitSparse} {
		t.Run(mode.String(), func(t *testing.T) {
			for _, c := range gvm.CasesMeshSplitMode(mode) {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

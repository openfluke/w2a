package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

func TestGoVMetalMeshBPTrain(t *testing.T) {
	for _, c := range gvm.CasesMeshBPTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

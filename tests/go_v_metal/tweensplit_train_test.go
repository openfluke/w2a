package go_v_metal_test

import (
	gvm "github.com/openfluke/w2a/suites/go_v_metal"
	"testing"
)

func TestGoVMetalTweenSplitTrain(t *testing.T) {
	for _, c := range gvm.CasesTweenSplitTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

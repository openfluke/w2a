package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

func TestGoVMetalStepBPTrain(t *testing.T) {
	if err := gvm.RunStepBPTrain(); err != nil {
		t.Fatal(err)
	}
}

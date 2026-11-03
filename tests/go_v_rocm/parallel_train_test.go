package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

func TestGoVRocmParallelTrain(t *testing.T) {
	for _, c := range gvr.CasesParallelNBPTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGoVRocmParallelPolyTrain(t *testing.T) {
	for _, c := range gvr.CasesParallelPolyTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGoVRocmNormalBPEdges(t *testing.T) {
	for _, c := range gvr.CasesNormalBPEdges() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

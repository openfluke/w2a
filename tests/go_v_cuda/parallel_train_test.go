package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

func TestGoVCudaParallelTrain(t *testing.T) {
	for _, c := range gvc.CasesParallelNBPTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGoVCudaParallelPolyTrain(t *testing.T) {
	for _, c := range gvc.CasesParallelPolyTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGoVCudaNormalBPEdges(t *testing.T) {
	for _, c := range gvc.CasesNormalBPEdges() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

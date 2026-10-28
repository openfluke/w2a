package go_v_cuda_test

import (
	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
	"testing"
)

func TestGoVCudaMeshTweenChainTrain(t *testing.T) {
	for _, c := range gvc.CasesMeshTweenChainTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

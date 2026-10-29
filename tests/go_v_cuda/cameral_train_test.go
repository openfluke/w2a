package go_v_cuda_test

import (
	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
	"testing"
)

func TestGoVCudaCameralTrain(t *testing.T) {
	for _, c := range gvc.CasesCameralTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
func TestGoVCudaCamSync(t *testing.T) {
	for _, c := range gvc.CasesCamSync() {
		t.Run(c.Name, func(t *testing.T) {
			if e := c.Run(); e != nil {
				t.Fatal(e)
			}
		})
	}
}

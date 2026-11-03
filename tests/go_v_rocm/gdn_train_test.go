package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

func TestGoVRocmGDNTrain(t *testing.T) {
	for _, c := range gvr.CasesGDNTrain() {
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

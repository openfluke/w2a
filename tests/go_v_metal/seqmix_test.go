package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalSeqmix runs Go↔Metal seqmix Kind contract smoke (no tensors).
func TestGoVMetalSeqmix(t *testing.T) {
	if err := gvm.RunSeqmix(); err != nil {
		t.Fatal(err)
	}
}

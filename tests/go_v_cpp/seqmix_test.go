package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppSeqmix runs Go↔C++ seqmix Kind contract smoke (no tensors).
func TestGoVCppSeqmix(t *testing.T) {
	if err := gvc.RunSeqmix(); err != nil {
		t.Fatal(err)
	}
}

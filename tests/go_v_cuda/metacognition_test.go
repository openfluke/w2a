package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaMetacognition — Go↔CUDA Metacognition forward parity.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaMetacognition$'
func TestGoVCudaMetacognition(t *testing.T) {
	if err := gvc.RunMetacognition(); err != nil {
		t.Fatal(err)
	}
}

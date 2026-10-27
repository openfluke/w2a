package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// TestGoVCudaSeqmix — Go↔CUDA Seqmix forward parity.
// Run: go test ./tests/go_v_cuda/ -v -run '^TestGoVCudaSeqmix$'
func TestGoVCudaSeqmix(t *testing.T) {
	if err := gvc.RunSeqmix(); err != nil {
		t.Fatal(err)
	}
}

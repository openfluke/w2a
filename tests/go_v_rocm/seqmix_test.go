package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmSeqmix — Go↔ROCm Seqmix forward parity.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmSeqmix$'
func TestGoVRocmSeqmix(t *testing.T) {
	if err := gvr.RunSeqmix(); err != nil {
		t.Fatal(err)
	}
}

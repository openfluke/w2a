package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmParallel — Go↔ROCm Parallel forward parity.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmParallel$'
func TestGoVRocmParallel(t *testing.T) {
	if err := gvr.RunParallel(); err != nil {
		t.Fatal(err)
	}
}

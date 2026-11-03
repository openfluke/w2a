package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmKMeans — Go↔ROCm KMeans forward parity.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmKMeans$'
func TestGoVRocmKMeans(t *testing.T) {
	if err := gvr.RunKMeans(); err != nil {
		t.Fatal(err)
	}
}

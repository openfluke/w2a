package go_v_rocm_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_rocm"
)

// TestGoVRocmMetacognition — Go↔ROCm Metacognition forward parity.
// Run: go test ./tests/go_v_rocm/ -v -run '^TestGoVRocmMetacognition$'
func TestGoVRocmMetacognition(t *testing.T) {
	if err := gvr.RunMetacognition(); err != nil {
		t.Fatal(err)
	}
}

package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalMetacognition — Go↔Metal Metacognition Observed+rules (exact Float32 + ×34).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalMetacognition$'
func TestGoVMetalMetacognition(t *testing.T) {
	if err := gvm.RunMetacognition(); err != nil {
		t.Fatal(err)
	}
}

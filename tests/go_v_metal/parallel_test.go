package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalParallel — Go↔Metal Parallel Dense NewConfigured (7 combines + Add×34).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalParallel$'
func TestGoVMetalParallel(t *testing.T) {
	if err := gvm.RunParallel(); err != nil {
		t.Fatal(err)
	}
}

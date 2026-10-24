package go_v_metal_test

import (
	"testing"

	gvm "github.com/openfluke/w2a/suites/go_v_metal"
)

// TestGoVMetalEmbedding — Go↔Metal Embedding gather (exact_bit ×34 + timings).
// Run: go test ./tests/go_v_metal/ -v -run '^TestGoVMetalEmbedding$'
func TestGoVMetalEmbedding(t *testing.T) {
	if err := gvm.RunEmbedding(); err != nil {
		t.Fatal(err)
	}
}

package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppEmbedding runs only Go↔C++ Embedding cases (use: go test -run '^TestGoVCppEmbedding$').
func TestGoVCppEmbedding(t *testing.T) {
	if err := gvc.RunEmbedding(); err != nil {
		t.Fatal(err)
	}
}

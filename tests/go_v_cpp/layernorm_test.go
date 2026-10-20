package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppLayerNorm runs only Go↔C++ LayerNorm cases (use: go test -run '^TestGoVCppLayerNorm$').
func TestGoVCppLayerNorm(t *testing.T) {
	if err := gvc.RunLayerNorm(); err != nil {
		t.Fatal(err)
	}
}

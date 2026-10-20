package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppRNN runs only Go↔C++ RNN cases (use: go test -run '^TestGoVCppRNN$').
func TestGoVCppRNN(t *testing.T) {
	if err := gvc.RunRNN(); err != nil {
		t.Fatal(err)
	}
}

package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppLSTM runs only Go↔C++ LSTM cases (use: go test -run '^TestGoVCppLSTM$').
func TestGoVCppLSTM(t *testing.T) {
	if err := gvc.RunLSTM(); err != nil {
		t.Fatal(err)
	}
}

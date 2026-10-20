package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

// TestGoVCppCNN runs only Go↔C++ CNN1/2/3 cases (use: go test -run '^TestGoVCppCNN$').
func TestGoVCppCNN(t *testing.T) {
	if err := gvc.RunCNN(); err != nil {
		t.Fatal(err)
	}
}

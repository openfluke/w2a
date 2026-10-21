package go_v_cpp_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cpp"
)

func TestGoVCppTweenSplitHeadProxyTrain(t *testing.T) {
	if err := gvc.RunTweenSplitHeadProxyTrain(); err != nil {
		t.Fatal(err)
	}
}

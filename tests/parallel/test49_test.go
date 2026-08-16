package parallel_test

import (
	"testing"

	parallelsuite "github.com/openfluke/w2a/suites/parallel"
	"github.com/openfluke/welvet/layers/parallel"
)

func Test49AllTrainModesCubes(t *testing.T) {
	modes := parallel.AllNamedTrainModes()
	if len(modes) < 23 {
		t.Fatalf("named train modes %d want ≥23 (test41+credit+mesh)", len(modes))
	}
	for _, m := range modes {
		if m == parallel.ModeInherit {
			t.Fatal("test49 must not include Inherit")
		}
	}
	if err := parallelsuite.Test49TrainGrids(); err != nil {
		t.Fatal(err)
	}
}

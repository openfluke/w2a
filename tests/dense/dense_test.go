package dense_test

import (
	"testing"

	denssuite "github.com/openfluke/w2a/suites/dense"
)

func TestDenseSuite(t *testing.T) {
	restore, err := denssuite.BeginLogging()
	if err != nil {
		t.Fatalf("dense logs: %v", err)
	}
	defer restore()

	for i, c := range denssuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			if err := denssuite.BeginCaseLog(i, c.Name); err != nil {
				t.Fatalf("case log: %v", err)
			}
			defer denssuite.EndCaseLog()

			if err := c.Run(); err != nil {
				t.Fatalf("[%d] %v", i+1, err)
			}
		})
	}
}

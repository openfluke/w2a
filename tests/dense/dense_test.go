package dense_test

import (
	"testing"

	denssuite "github.com/openfluke/w2a/suites/dense"
)

func TestDenseSuite(t *testing.T) {
	for i, c := range denssuite.Cases() {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Run(); err != nil {
				t.Fatalf("[%d] %v", i+1, err)
			}
		})
	}
}

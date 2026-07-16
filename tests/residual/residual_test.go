package residual_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	residualsuite "github.com/openfluke/w2a/suites/residual"
)

func TestResidualSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range residualsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("residual", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("residual", c.Name, "PASS", "")
		})
	}
}

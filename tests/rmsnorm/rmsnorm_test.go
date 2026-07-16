package rmsnorm_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	rmsnsuite "github.com/openfluke/w2a/suites/rmsnorm"
)

func TestRMSNormSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range rmsnsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("rmsnorm", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("rmsnorm", c.Name, "PASS", "")
		})
	}
}

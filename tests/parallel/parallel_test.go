package parallel_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	parallelsuite "github.com/openfluke/w2a/suites/parallel"
)

func TestParallelSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range parallelsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			if err := c.Run(); err != nil {
				suites.EndCase("parallel", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("parallel", c.Name, "PASS", "")
		})
	}
}

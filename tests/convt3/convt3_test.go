package convt3_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	convt3suite "github.com/openfluke/w2a/suites/convt3"
)

func TestConvT3Suite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range convt3suite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			if err := c.Run(); err != nil {
				suites.EndCase("convt3", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("convt3", c.Name, "PASS", "")
		})
	}
}

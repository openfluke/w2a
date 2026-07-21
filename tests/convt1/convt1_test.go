package convt1_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	convt1suite "github.com/openfluke/w2a/suites/convt1"
)

func TestConvT1Suite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range convt1suite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			if err := c.Run(); err != nil {
				suites.EndCase("convt1", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("convt1", c.Name, "PASS", "")
		})
	}
}

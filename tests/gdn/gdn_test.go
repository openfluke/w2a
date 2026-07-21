package gdn_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	gdnsuite "github.com/openfluke/w2a/suites/gdn"
)

func TestGDNSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range gdnsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			if err := c.Run(); err != nil {
				suites.EndCase("gdn", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("gdn", c.Name, "PASS", "")
		})
	}
}

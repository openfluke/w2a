package cnn3_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	cnn3suite "github.com/openfluke/w2a/suites/cnn3"
)

func TestCNN3Suite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range cnn3suite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("cnn3", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("cnn3", c.Name, "PASS", "")
		})
	}
}

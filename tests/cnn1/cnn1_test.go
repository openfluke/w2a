package cnn1_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	cnn1suite "github.com/openfluke/w2a/suites/cnn1"
)

func TestCNN1Suite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range cnn1suite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("cnn1", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("cnn1", c.Name, "PASS", "")
		})
	}
}

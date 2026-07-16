package rnn_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	rnnsuite "github.com/openfluke/w2a/suites/rnn"
)

func TestRNNSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range rnnsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("rnn", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("rnn", c.Name, "PASS", "")
		})
	}
}

package softmax_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	softmaxsuite "github.com/openfluke/w2a/suites/softmax"
)

func TestSoftmaxSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range softmaxsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("softmax", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("softmax", c.Name, "PASS", "")
		})
	}
}

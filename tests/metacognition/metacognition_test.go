package metacognition_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	metacognitionsuite "github.com/openfluke/w2a/suites/metacognition"
)

func TestMetacognitionSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range metacognitionsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			if err := c.Run(); err != nil {
				suites.EndCase("metacognition", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("metacognition", c.Name, "PASS", "")
		})
	}
}

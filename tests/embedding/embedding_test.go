package embedding_test

import (
	"testing"

	"github.com/openfluke/w2a/suites"
	embeddingsuite "github.com/openfluke/w2a/suites/embedding"
)

func TestEmbeddingSuite(t *testing.T) {
	restore, err := suites.BeginLog()
	if err != nil {
		t.Fatalf("suite log: %v", err)
	}
	defer func() {
		suites.PrintReport()
		restore()
	}()

	for i, c := range embeddingsuite.Cases() {
		c := c
		i := i
		t.Run(c.Name, func(t *testing.T) {
			suites.BeginCase()
			err := c.Run()
			if err != nil {
				suites.EndCase("embedding", c.Name, "FAIL", err.Error())
				t.Fatalf("[%d] %v", i+1, err)
			}
			suites.EndCase("embedding", c.Name, "PASS", "")
		})
	}
}

package go_v_metal_test

import (
	gvm "github.com/openfluke/w2a/suites/go_v_metal"
	"testing"
)

// Exactly the Dense NormalBP through Stack NormalBP checklist range. Every dtype,
// leaf, chain and volumetric case runs; StepBP/Tween and other modes are separate.
func TestGoVMetalNormalBP(t *testing.T) {
	t.Log("Metal parity: exact tensor bits where specified; loss uses the finite float2 band; FP32 subnormal flushing is allowed. Softmax and GDN retain their declared spectrum floors.")
	groups := []struct {
		name  string
		cases []gvm.Case
	}{
		{"Dense", gvm.CasesDenseTrain()},
		{"RMSNorm", gvm.CasesRMSNormTrain()},
		{"LayerNorm", gvm.CasesLayerNormTrain()},
		{"SwiGLU", gvm.CasesSwiGLUTrain()},
		{"Softmax", gvm.CasesSoftmaxTrain()},
		{"Embedding", gvm.CasesEmbeddingTrain()},
		{"MHA", gvm.CasesMHATrain()},
		{"CNN", gvm.CasesCNNTrain()},
		{"ConvT", gvm.CasesConvTTrain()},
		{"RNN", gvm.CasesRNNTrain()},
		{"LSTM", gvm.CasesLSTMTrain()},
		{"Sequential", gvm.CasesSequentialTrain()},
		{"Residual", gvm.CasesResidualTrain()},
		{"Mamba", gvm.CasesMambaTrain()},
		{"Metacognition", gvm.CasesMetacognitionTrain()},
		{"GDN", gvm.CasesGDNTrain()},
		{"Stack", gvm.CasesStackTrain()},
	}
	for _, group := range groups {
		t.Run(group.name, func(t *testing.T) {
			for _, c := range group.cases {
				t.Run(c.Name, func(t *testing.T) {
					if err := c.Run(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

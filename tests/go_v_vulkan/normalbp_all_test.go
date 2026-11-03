package go_v_vulkan_test

import (
	"testing"

	gvr "github.com/openfluke/w2a/suites/go_v_vulkan"
)

// Complete NormalBP checklist, including KMeans and Parallel.
func TestGoVVulkanNormalBP(t *testing.T) {
	t.Log("Vulkan parity: exact tensor bits where specified; loss uses the finite float2 band; FP32 subnormal flushing is allowed. Softmax and GDN retain their declared spectrum floors.")
	groups := []struct {
		name  string
		cases []gvr.Case
	}{
		{"Dense", gvr.CasesDenseTrain()},
		{"RMSNorm", gvr.CasesRMSNormTrain()},
		{"LayerNorm", gvr.CasesLayerNormTrain()},
		{"SwiGLU", gvr.CasesSwiGLUTrain()},
		{"Softmax", gvr.CasesSoftmaxTrain()},
		{"Embedding", gvr.CasesEmbeddingTrain()},
		{"MHA", gvr.CasesMHATrain()},
		{"CNN", gvr.CasesCNNTrain()},
		{"ConvT", gvr.CasesConvTTrain()},
		{"RNN", gvr.CasesRNNTrain()},
		{"LSTM", gvr.CasesLSTMTrain()},
		{"Sequential", gvr.CasesSequentialTrain()},
		{"Residual", gvr.CasesResidualTrain()},
		{"Mamba", gvr.CasesMambaTrain()},
		{"Metacognition", gvr.CasesMetacognitionTrain()},
		{"GDN", gvr.CasesGDNTrain()},
		{"Stack", gvr.CasesStackTrain()},
		{"KMeans", gvr.CasesKMeansTrain()},
		{"Parallel", gvr.CasesParallelNBPTrain()},
		{"ParallelPoly", gvr.CasesParallelPolyTrain()},
		{"NormalBPEdges", gvr.CasesNormalBPEdges()},
		{"ParallelVolumes", gvr.CasesParallelVolumeDTypes()},
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

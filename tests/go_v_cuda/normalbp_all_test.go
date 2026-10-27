package go_v_cuda_test

import (
	"testing"

	gvc "github.com/openfluke/w2a/suites/go_v_cuda"
)

// Complete NormalBP checklist, including KMeans and Parallel.
func TestGoVCudaNormalBP(t *testing.T) {
	t.Log("CUDA parity: exact tensor bits where specified; loss uses the finite float2 band; FP32 subnormal flushing is allowed. Softmax and GDN retain their declared spectrum floors.")
	groups := []struct {
		name  string
		cases []gvc.Case
	}{
		{"Dense", gvc.CasesDenseTrain()},
		{"RMSNorm", gvc.CasesRMSNormTrain()},
		{"LayerNorm", gvc.CasesLayerNormTrain()},
		{"SwiGLU", gvc.CasesSwiGLUTrain()},
		{"Softmax", gvc.CasesSoftmaxTrain()},
		{"Embedding", gvc.CasesEmbeddingTrain()},
		{"MHA", gvc.CasesMHATrain()},
		{"CNN", gvc.CasesCNNTrain()},
		{"ConvT", gvc.CasesConvTTrain()},
		{"RNN", gvc.CasesRNNTrain()},
		{"LSTM", gvc.CasesLSTMTrain()},
		{"Sequential", gvc.CasesSequentialTrain()},
		{"Residual", gvc.CasesResidualTrain()},
		{"Mamba", gvc.CasesMambaTrain()},
		{"Metacognition", gvc.CasesMetacognitionTrain()},
		{"GDN", gvc.CasesGDNTrain()},
		{"Stack", gvc.CasesStackTrain()},
		{"KMeans", gvc.CasesKMeansTrain()},
		{"Parallel", gvc.CasesParallelNBPTrain()},
		{"ParallelPoly", gvc.CasesParallelPolyTrain()},
		{"NormalBPEdges", gvc.CasesNormalBPEdges()},
		{"ParallelVolumes", gvc.CasesParallelVolumeDTypes()},
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

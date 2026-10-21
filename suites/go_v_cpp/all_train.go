package go_v_cpp

import (
	"fmt"
	"runtime"
	"strings"
	"time"
)

// TrainSection is one checklist-aligned train suite for OS/CPU sign-off.
type TrainSection struct {
	Name string
	Run  func() error
}

// AllTrainSections lists every go_v_cpp train runner (NormalBP → cameral → Parallel).
func AllTrainSections() []TrainSection {
	return []TrainSection{
		{"DenseTrain", RunDenseTrain},
		{"RMSNormTrain", RunRMSNormTrain},
		{"LayerNormTrain", RunLayerNormTrain},
		{"SwiGLUTrain", RunSwiGLUTrain},
		{"SoftmaxTrain", RunSoftmaxTrain},
		{"EmbeddingTrain", RunEmbeddingTrain},
		{"MHATrain", RunMHATrain},
		{"CNNTrain", RunCNNTrain},
		{"ConvTTrain", RunConvTTrain},
		{"RNNTrain", RunRNNTrain},
		{"LSTMTrain", RunLSTMTrain},
		{"SequentialTrain", RunSequentialTrain},
		{"ResidualTrain", RunResidualTrain},
		{"MambaTrain", RunMambaTrain},
		{"MetacognitionTrain", RunMetacognitionTrain},
		{"GDNTrain", RunGDNTrain},
		{"KMeansTrain", RunKMeansTrain},
		{"StackTrain", RunStackTrain},
		{"StepBPTrain", RunStepBPTrain},
		{"TweenTrain", RunTweenTrain},
		{"TweenChainTrain", RunTweenChainTrain},
		{"StepTweenTrain", RunStepTweenTrain},
		{"StepTweenChainTrain", RunStepTweenChainTrain},
		{"MeshTweenTrain", RunMeshTweenTrain},
		{"MeshTweenChainTrain", RunMeshTweenChainTrain},
		{"TweenSplitTrain", RunTweenSplitTrain},
		{"StepTweenSplitTrain", RunStepTweenSplitTrain},
		{"TweenAltTrain", RunTweenAltTrain},
		{"StepTweenAltTrain", RunStepTweenAltTrain},
		{"TweenSplitHeadProxyTrain", RunTweenSplitHeadProxyTrain},
		{"TweenSplitLinearTrain", RunTweenSplitLinearTrain},
		{"TweenSplitFastProxyTrain", RunTweenSplitFastProxyTrain},
		{"TweenSplitLinearCacheTrain", RunTweenSplitLinearCacheTrain},
		{"TweenSplitHeadProxyAsyncTrain", RunTweenSplitHeadProxyAsyncTrain},
		{"TweenSplitSparseTrain", RunTweenSplitSparseTrain},
		{"MeshTweenSplitTrain", RunMeshTweenSplitTrain},
		{"MeshTweenAltTrain", RunMeshTweenAltTrain},
		{"MeshTweenSplitFastProxyTrain", RunMeshTweenSplitFastProxyTrain},
		{"MeshTweenSplitSparseTrain", RunMeshTweenSplitSparseTrain},
		{"StepTweenSplitHeadProxyTrain", RunStepTweenSplitHeadProxyTrain},
		{"StepTweenSplitLinearTrain", RunStepTweenSplitLinearTrain},
		{"StepTweenSplitFastProxyTrain", RunStepTweenSplitFastProxyTrain},
		{"StepTweenSplitLinearCacheTrain", RunStepTweenSplitLinearCacheTrain},
		{"StepTweenSplitHeadProxyAsyncTrain", RunStepTweenSplitHeadProxyAsyncTrain},
		{"StepTweenSplitSparseTrain", RunStepTweenSplitSparseTrain},
		{"FreezeTrain", RunFreezeTrain},
		{"ShadowTrain", RunShadowTrain},
		{"AdversarialTrain", RunAdversarialTrain},
		{"MemoryTrain", RunMemoryTrain},
		{"ParallelNBPTrain", RunParallelNBPTrain},
		{"ParallelPolyTrain", RunParallelPolyTrain},
		{"ParallelTweenTrain", RunParallelTweenTrain},
	}
}

type sectionResult struct {
	Name    string
	Pass    bool
	Err     string
	Elapsed time.Duration
}

// RunAllTrain executes every train section, prints a timing table + totals.
// Returns error if any section failed (still prints the full report).
func RunAllTrain() error {
	secs := AllTrainSections()
	results := make([]sectionResult, 0, len(secs))
	var passN, failN int
	var total time.Duration

	fmt.Printf("\n=== go_v_cpp ALL TRAIN — %s/%s — %d sections ===\n", runtime.GOOS, runtime.GOARCH, len(secs))
	fmt.Printf("%-36s %8s %10s  %s\n", "SECTION", "STATUS", "TIME", "NOTE")
	fmt.Printf("%s\n", strings.Repeat("-", 80))

	for _, s := range secs {
		start := time.Now()
		err := s.Run()
		elapsed := time.Since(start)
		total += elapsed
		r := sectionResult{Name: s.Name, Pass: err == nil, Elapsed: elapsed}
		if err != nil {
			r.Err = err.Error()
			failN++
			fmt.Printf("%-36s %8s %10s  %s\n", s.Name, "FAIL", elapsed.Round(time.Millisecond), truncate(r.Err, 60))
		} else {
			passN++
			fmt.Printf("%-36s %8s %10s\n", s.Name, "PASS", elapsed.Round(time.Millisecond))
		}
		results = append(results, r)
	}

	fmt.Printf("%s\n", strings.Repeat("-", 80))
	fmt.Printf("SECTIONS: %d  PASS: %d  FAIL: %d  TOTAL: %s\n", len(secs), passN, failN, total.Round(time.Millisecond))
	fmt.Printf("PLATFORM: %s/%s\n", runtime.GOOS, runtime.GOARCH)

	if failN > 0 {
		var fails []string
		for _, r := range results {
			if !r.Pass {
				fails = append(fails, r.Name)
			}
		}
		return fmt.Errorf("all_train: %d/%d sections failed: %s", failN, len(secs), strings.Join(fails, ", "))
	}
	return nil
}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

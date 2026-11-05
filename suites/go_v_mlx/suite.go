// Package go_v_mlx times Go CPU vs Apple MLX on welvet.mlx forward leaves.
//
// Protocol (per layer): transfer → prep → exec-only → e2e + go↔mlx accuracy.
package go_v_mlx

import (
	"fmt"
	"strings"

	"github.com/openfluke/w2a/suites"
	"github.com/openfluke/welvet/nativemlx"
)

type Case struct {
	Name string
	Run  func() error
}

func Cases() []Case {
	return []Case{
		{Name: "Caps — README MLX layers × float32", Run: caseCapsInventory},
		{Name: "Dense caps — all compute dtypes (+ float64 rejected)", Run: caseDenseCaps},
		{Name: "Dense f32 B=32 in=512 out=512 — transfer/prep/exec/e2e + accuracy", Run: caseLayerDense},
		{Name: "Dense ALL dtypes — Session prepare/exec vs Go", Run: caseDenseAllDtypes},
		{Name: "Dense Go layer — SyncToMLX + BackendMLX (f32/f16/bf16)", Run: caseDenseGoLayer},
		{Name: "Softmax f32 64×4096 — transfer/prep/exec/e2e + accuracy", Run: caseLayerSoftmax},
		{Name: "Softmax ALL float dtypes — Session × f32/f16/bf16", Run: caseSoftmaxAllDtypes},
		{Name: "RMSNorm f32 [8,128,512] — transfer/prep/exec/e2e + accuracy", Run: caseLayerRMSNorm},
		{Name: "RMSNorm ALL float dtypes — Session × f32/f16/bf16", Run: caseRMSNormAllDtypes},
		{Name: "LayerNorm f32 [8,128,512] — transfer/prep/exec/e2e + accuracy", Run: caseLayerLayerNorm},
		{Name: "LayerNorm ALL float dtypes — Session × f32/f16/bf16", Run: caseLayerNormAllDtypes},
		{Name: "CNN2 ALL dtypes — Session × floats+ints", Run: caseCNN2AllDtypes},
		{Name: "CNN3 ALL dtypes — Session × floats+ints", Run: caseCNN3AllDtypes},
		{Name: "ConvT2 ALL dtypes — Session × floats+ints", Run: caseConvT2AllDtypes},
		{Name: "ConvT3 ALL dtypes — Session × floats+ints", Run: caseConvT3AllDtypes},
		{Name: "LSTM ALL float dtypes — Session × f32/f16/bf16", Run: caseLSTMAllDtypes},
		{Name: "Embedding ALL dtypes — Session × floats+ints", Run: caseEmbeddingAllDtypes},
		{Name: "Residual f32 n=65536 — transfer/prep/exec/e2e + accuracy", Run: caseLayerResidual},
		{Name: "Residual ALL dtypes — Session × floats+ints+c64", Run: caseResidualAllDtypes},
		{Name: "SwiGLU ALL float dtypes — Session × f32/f16/bf16", Run: caseSwiGLUAllDtypes},
		{Name: "MHA ALL float dtypes — Session × f32/f16/bf16", Run: caseMHAAllDtypes},
		{Name: "Sequential Dense tower dim=64 depth=3 — go/mlx f32", Run: caseSequentialF32},
	}
}

func RunAll() error {
	if !nativemlx.Available() {
		fmt.Println("  SKIP go_v_mlx: libwelvet_mlx not available (need cgo + darwin/arm64 + dylib + Metal)")
		return nil
	}
	fmt.Printf("  nativemlx=%s  layers=%v\n", nativemlx.Version(), nativemlx.SupportedLayers())
	fmt.Println("  measure: transfer → create → exec-only + e2e + go↔mlx exact%/max|Δ|/RMSE")
	var fails []string
	for i, c := range Cases() {
		nativemlx.CloseDefaultSession()
		suites.BeginCase()
		fmt.Printf("  [%d] %s …\n", i+1, c.Name)
		if err := c.Run(); err != nil {
			suites.EndCase("go_v_mlx", c.Name, "FAIL", err.Error())
			fmt.Printf("      FAIL: %v\n", err)
			fails = append(fails, fmt.Sprintf("%d:%s", i+1, c.Name))
			continue
		}
		suites.EndCase("go_v_mlx", c.Name, "PASS", "")
		fmt.Println("      PASS")
	}
	nativemlx.CloseDefaultSession()
	if len(fails) > 0 {
		return fmt.Errorf("go_v_mlx: %d failed: %s", len(fails), strings.Join(fails, ", "))
	}
	return nil
}

func RunOne(n int) error {
	if !nativemlx.Available() {
		return fmt.Errorf("go_v_mlx: libwelvet_mlx not available")
	}
	cs := Cases()
	if n < 1 || n > len(cs) {
		return fmt.Errorf("go_v_mlx: case %d out of range 1..%d", n, len(cs))
	}
	suites.BeginCase()
	fmt.Printf("  [%d] %s …\n", n, cs[n-1].Name)
	if err := cs[n-1].Run(); err != nil {
		suites.EndCase("go_v_mlx", cs[n-1].Name, "FAIL", err.Error())
		fmt.Printf("      FAIL: %v\n", err)
		return err
	}
	suites.EndCase("go_v_mlx", cs[n-1].Name, "PASS", "")
	fmt.Println("      PASS")
	return nil
}

func openSess() (*nativemlx.Session, error) {
	return nativemlx.DefaultSession()
}

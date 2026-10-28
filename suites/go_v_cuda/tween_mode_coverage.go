package go_v_cuda

import (
	"fmt"
	"strings"

	"github.com/openfluke/welvet/core"
)

func withTypedVolumes(base, parent []Case, prefix string) []Case {
	for _, c := range parent {
		if strings.Contains(c.Name, "volume 211 × all 34 dtypes") {
			base = append(base, Case{Name: prefix + "volume alias / " + c.Name, Run: c.Run})
		}
	}
	return base
}

// denseMeshCreditDTypes — Dense mesh-clock ×34. chain=false → MeshTween (MTW1);
// chain=true lands with MeshTweenChain (MTC1) once that wire is signed.
func denseMeshCreditDTypes(chain bool, depth int) error {
	if chain {
		return fmt.Errorf("denseMeshCreditDTypes(chain=true): MeshTweenChain wire not signed yet")
	}
	return runTweenVolumeSpectrum(func(dtype core.DType) error {
		cells := make([]volCell, depth)
		for i := range cells {
			cells[i] = volCell{enabled: true, in: 4, out: 4, act: core.ActivationLinear, dtype: dtype, W: randInit(4, 4, 0x4d433030+uint32(i))}
		}
		x := randInit(1, 4, 0x4d433058)
		target := []float32{0.1, 0.2, -0.3, 0.4}
		goO, err := runGoDenseMeshTween(depth, 1, 1, 1, 1, depth+2, cells, x, target, 0.01)
		if err != nil {
			return err
		}
		payload, err := encodeDenseMTW1(depth, 1, 1, 1, 1, depth+2, cells, x, target, 0.01)
		if err != nil {
			return err
		}
		got, _, err := runCUDAMTW1(payload, 0)
		if err != nil {
			return err
		}
		if err := compareMeshTween(goO, got, true); err != nil {
			return fmt.Errorf("mesh Tween %s: %w", dtype, err)
		}
		bench, _, err := runCUDAMTW1(payload, 2)
		if err != nil {
			return err
		}
		return compareMeshTween(got, bench, true)
	})
}

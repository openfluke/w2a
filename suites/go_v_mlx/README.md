# go_v_mlx — Go vs Apple MLX GPU (forward-only)

Sibling of `go_v_qnn`. Validates **welvet.mlx** leaf forwards — **no training**.

## Protocol

1. Build identical weights / activations in Go  
2. **Transfer** — host materialize (timed)  
3. **Prep** — DensePrepare / array cast (`LastPhase` create)  
4. **Exec** — `mx::eval` window (`LastPhase` execute) — headline  
5. **E2E** = transfer + create + finalize + exec  
6. **Accuracy** — go↔mlx exact% / max|Δ| / RMSE where wired  

Sign-off checklist: [`welvet.mlx/docs/FORWARD_CHECKLIST.md`](../../../welvet.mlx/docs/FORWARD_CHECKLIST.md).

## Dense (complete)

| Case | What |
|------|------|
| Dense caps | all compute dtypes + float64 rejected |
| Dense f32 large | B=32 in=out=512 prepare/exec + Go accuracy |
| Dense ALL dtypes | Session prepare/exec for every Dense caps dtype |
| Dense Go layer | `SyncToMLX` → `PrepareMLX` → `BackendMLX` Forward (f32/f16/bf16) |

```bash
go test ./tests/go_v_mlx -run TestGoVMLXDense -v -count=1
```

## Gate

Skips unless `nativemlx.Available()` (cgo + darwin/arm64 + `libwelvet_mlx.dylib` + Metal).

```bash
../welvet.mlx/scripts/macos_build_w2a.sh
cd w2a
go test ./tests/go_v_mlx -v -count=1 -timeout 60m
```

# Dense dtype conversion heatmap (volumetric)

Train a **2×2×2** FP32 Dense grid → snapshot `.entity` → `weights.Convert` every cell to each of `core.AllDTypes` → forward-eval vs the FP32 reference → ASCII heatmap.

```bash
cd apps/w2a
go run ./examples/dense_dtype_heatmap
```

Pipeline:

1. Place Dense `8→8` ReLU on every volumetric cell (one remote hop for spice)
2. `training.StepMesh` in **FP32** for a few epochs on a toy batch
3. `serialization.SerializeEntity`
4. Reload FP32 `.entity` as the **reference** (fair baseline)
5. For each dtype: `DeserializeEntity` → `weights.Convert(..., FormatNone)` on every Dense cell → compare outputs to FP32 refs
6. Print table + ASCII heatmap (` .:+*#@`) of `mse_vs_fp32` and `max|d|`
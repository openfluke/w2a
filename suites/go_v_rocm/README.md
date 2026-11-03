# go_v_rocm — Go ↔ welvet.rocm determinism

Leaf FormatNone + NormalBP Mode 1 + StepBP + ModeTween suites (mirror of `go_v_cuda`).

```bash
cd ../welvet.rocm
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --normalbp-all
./scripts/linux_build_w2a.sh --stepbp-train
./scripts/linux_build_w2a.sh --tween-train
```

Env: `WELVET_ROCM_DENSE_FWD` → `welvet.rocm/build/welvet_dense_fwd`.

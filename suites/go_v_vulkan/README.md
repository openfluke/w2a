# go_v_vulkan — Go ↔ welvet.vulkan determinism

Leaf FormatNone + NormalBP Mode 1 + StepBP + ModeTween suites (mirror of `go_v_cuda` / `go_v_rocm`).

Wire-subprocess tests only (no `nativecuda` / Go routing FFI in this package).

```bash
cd ../welvet.vulkan
./scripts/linux_build_w2a.sh --dense-fwd
./scripts/linux_build_w2a.sh --normalbp-all
./scripts/linux_build_w2a.sh --training-all
```

Env: `WELVET_VULKAN_DENSE_FWD` → `welvet.vulkan/build/welvet_dense_fwd` (default when sibling repo is built).

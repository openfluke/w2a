# Welvet for Python

**Python package `1.1.2`** · powered by the **[Welvet](https://github.com/openfluke/welvet) Go AI engine `v1.7.3`**

ctypes bindings over a prebuilt C ABI (`welvet.so` / `welvet.dylib`). Same volumetric
grids, layers, dtypes/quants, training, and cameral modes as Go — plus optional
Linux/macOS accelerators (CUDA, ROCm, Vulkan, Metal, …) selected at runtime.

> Replaces the Loom-era PyPI line (`welvet==0.80.0`). Prefer `create_grid` + `place_*`
> over legacy Loom network helpers.

---

## Install

```bash
pip install welvet==1.1.2
```

Wheels ship with the matching native CABI:

| Wheel | Includes |
|-------|----------|
| Linux x86_64 (manylinux) | `welvet.so` + cuda / rocm / vulkan / cpp siblings |
| macOS arm64 | `welvet.dylib` + Metal sibling |

```python
import welvet as w

print(w.__version__)       # 1.1.2  (Python / CABI contract)
print(w.engine_version())  # 1.1.2  (embedded CABI string)
# Engine behind the ABI: Welvet Go v1.7.3
```

---

## Quick start

```python
import welvet as w

w.assert_engine_version()

# See what this machine can run
for b in w.list_backends():
    mark = "✓" if b["available"] else "·"
    print(f"  {mark} {b['id']}")

g = w.create_grid(backend="cpu_tiled")  # or cuda / metal / vulkan / …
w.place_dense(g, {"in": 8, "out": 8, "act": "relu", "dtype": w.DType.FLOAT32})

x = [0.1] * 8
y = [0.0] * 7 + [1.0]
print("out", w.forward(g, x)[:4])
print("loss", w.train_sgd(g, x, y, lr=0.05))

blob = w.serialize_entity(g)   # .entity checkpoint bytes
w.free_grid(g)

g2 = w.deserialize_entity(blob)
w.free_grid(g2)
```

### GPU / accelerator backend

```python
import welvet as w

# Pick any available backend from list_backends()
g = w.create_grid(backend="cuda")       # Linux + NVIDIA
# g = w.create_grid(backend="metal")    # macOS Apple silicon
# g = w.create_grid(backend="vulkan")   # Linux Vulkan
# g = w.create_grid(backend="rocm")     # Linux AMD ROCm
# g = w.create_grid(backend="native_cpu")

w.place_dense(g, {"in": 16, "out": 16, "act": "relu", "dtype": 1})
out = w.forward(g, [0.05] * 16)
w.free_grid(g)
```

Routed backends (`cuda` / `metal` / `rocm` / `vulkan`) enable Go `EnableRouting`
inside the CABI. For `qnn` / `mlx`, call `w.prepare_grid(g)` after `place_*`
(SyncTo* before forward).

### Attention layer

```python
import welvet as w

g = w.create_grid()
w.place(g, "mha", {
    "dim": 8,
    "DModel": 8,
    "NumHeads": 2,
    "SeqLen": 4,
    "dtype": w.DType.FLOAT32,
})
# batch=1, seq=4, dim=8
out = w.forward(g, [0.1] * 32, shape=[1, 4, 8])
print(len(out))  # 32
w.free_grid(g)
```

### Cameral / bicameral train

```python
import welvet as w

stack = w.create_bicameral(in_=4, hidden=8, out=4, dtype=1)
w.set_child_modes(stack, ["NormalBP", "NormalBP"])
meta = w.train_stack_mse(stack, [0.1, 0.2, 0.3, 0.4], [0, 0, 0, 1], mode="NormalBP", lr=0.05)
print(meta)
w.free_stack(stack)
```

---

## Backends

| `backend=` | Typical host | Notes |
|------------|--------------|--------|
| `cpu_tiled` | everywhere | Default |
| `simd` | CPU | Plan 9 / SIMD paths |
| `webgpu` | desktop | Needs a WebGPU device |
| `native_cpu` | Linux / macOS | welvet.cpp via linked `libwelvet` |
| `cuda` | Linux amd64 | Needs NVIDIA driver + `libwelvet_cuda` |
| `rocm` | Linux amd64 | Needs ROCm + `libwelvet_rocm` |
| `vulkan` | Linux amd64 | Needs Vulkan + `libwelvet_vulkan` |
| `metal` | macOS arm64 | Needs `libwelvet_metal` |
| `mlx` | macOS arm64 | Forward-oriented; `prepare_grid` |
| `qnn` | Windows/Android arm64 | Hexagon HTP; `prepare_grid` |

```python
w.list_backends()   # rich probe: id, available, routing, note
w.enable_backend("cuda")
w.set_grid_backend(g, "vulkan")
```

---

## Layers (`place` / `place_dense`)

`dense` · `mha` · `swiglu` · `rmsnorm` · `layernorm` · `embedding` · `softmax` ·
`sequential` · `residual` · `cnn1`/`cnn2`/`cnn3` · `rnn` · `lstm` ·
`convt1`/`convt2`/`convt3` · `parallel` · `stack` · `kmeans` · `mamba` ·
`metacognition` · `gdn`

```python
w.place(g, "swiglu", {"dim": 32, "dtype": 1})
w.place_dense(g, {"in": 32, "out": 10, "act": "relu", "dtype": 1})
```

Use `w.DType.FLOAT32` (or integer ids from `w.list_dtypes()`).

---

## Training helpers

| Call | Role |
|------|------|
| `train_sgd` / `train_tween` | Grid SGD / tween step |
| `train_stack_mse` / `train_parallel_mse` | Cameral MSE |
| `serialize_entity` / `deserialize_entity` | `.entity` checkpoints |
| `run_suite` / `run_all_suites` | Go w2a suites via CABI |

```bash
python -m welvet.cabi_verify          # smoke: version, dense, MHA, backends
python -m welvet.w2a run --quick      # suite smoke set
python -m welvet.w2a catalog
```

---

## Versions

| Component | Version |
|-----------|---------|
| PyPI / Python package | **1.1.2** |
| Embedded CABI (`engine_version()`) | **1.1.2** |
| Welvet Go engine (source of this build) | **v1.7.3** |

CABI **1.1.2** adds runtime backend selection (`runtime/accel`: cuda · metal ·
rocm · vulkan · native_cpu · qnn · mlx) on top of the Welvet **v1.7.3** engine.

---

## Links

- Engine: [github.com/openfluke/welvet](https://github.com/openfluke/welvet)
- Harness / CABI / releases: [github.com/openfluke/w2a](https://github.com/openfluke/w2a)
- CABI binaries: [w2a releases · cabi-v1.1.2](https://github.com/openfluke/w2a/releases/tag/cabi-v1.1.2)
- Feature book: [openfluke.github.io/welvet](https://openfluke.github.io/welvet/)

---

## Dev / maintainers

```bash
# Fetch prebuilts into the checkout (gitignored)
cd w2a
./scripts/fetch_cabi.sh --all cabi-v1.1.2

# Or build Linux CABI from Go
./scripts/pack_cabi_linux.sh --build

pip install -e python/
python -m welvet.cabi_verify

# Wheels → PyPI
./scripts/build_pypi_wheels.sh
twine upload python/dist/*.whl
```

License: Apache-2.0

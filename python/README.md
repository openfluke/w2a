# Welvet Python (C-ABI) 1.1.2

ctypes bindings over the Welvet shared library built from `apps/w2a/cabi`.

**1.1.2** — `create_grid(backend=…)` selects `cpu_tiled` / `simd` / `webgpu` /
`native_cpu` / `cuda` / `metal` / `rocm` / `vulkan` / `qnn` / `mlx`. Routed
GPU backends call `EnableRouting` inside the CABI; sibling `libwelvet_*.so`
next to `welvet.so` are pinned via `WELVET_*_LIBRARY`. Use `list_backends()` /
`probe_backends()` to see what this host can attempt. For QNN/MLX call
`prepare_grid(g)` after place (SyncTo*).

## Setup

Prebuilt CABI ships on **GitHub Releases** (not in git). PyPI wheels bundle the
matching platform libs (`linux_amd64`, `macos_arm64`, …).

```bash
pip install welvet==1.1.2

# Dev from checkout:
cd w2a
./scripts/fetch_cabi.sh cabi-v1.1.2   # or: python -m welvet.fetch_cabi cabi-v1.1.2
cd python && pip install -e .
python -m welvet.cabi_verify
```

Build/publish wheels (maintainers):

```bash
cd w2a
./scripts/fetch_cabi.sh cabi-v1.1.2          # linux + run on mac for macos, or curl both
./scripts/build_pypi_wheels.sh               # → python/dist/*.whl
twine upload python/dist/*.whl
```

Build from source instead:

```bash
cd w2a/cabi/internal/build
./build_linux.sh --test
./copy_to_python.sh
# or:  cd w2a && ./scripts/pack_cabi_linux.sh --build
```


## API

```python
import welvet as w

print(w.list_backends())  # [{id, available, …}, …]

g = w.create_grid(backend="cuda")  # or cpu_tiled / vulkan / …
w.place_dense(g, {"in": 8, "out": 8, "act": "relu", "dtype": 1})
out = w.forward(g, [0.1] * 8)
w.train_sgd(g, [0.1] * 8, [0.0] * 8, lr=0.05)
blob = w.serialize_entity(g)
w.free_grid(g)

print(w.run_all_suites(quick=True))
```

## Suite CLI

```bash
python -m welvet.w2a run --quick   # Go suites via CABI (smoke set)
python -m welvet.w2a run --mid     # denser Go subset + Python dense matrix
python -m welvet.w2a run --mega    # all Go suites + Python mega matrix
python -m welvet.w2a run --all     # all Go suites only
python -m welvet.w2a catalog
```

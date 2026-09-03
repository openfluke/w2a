# Welvet C-ABI (apps/w2a/cabi)

Go `c-shared` / `c-archive` bridge for Welvet **1.1.1**. Mirrors Loom’s polyglot builder layout; exports mirror WASM `place*` + suites.

## Build

```bash
cd apps/w2a/cabi/internal/build

./build_linux.sh --test          # native linux
./build_linux.sh all             # amd64 + arm64
./build_unix.sh darwin arm64     # macOS
./build_unix.sh windows amd64    # needs mingw
./build_android.sh               # needs ANDROID_NDK_HOME
./build_ios.sh                   # macOS only
./build_all.sh                   # master try — soft-SKIP missing toolchains
```

Outputs: `dist/<os>_<arch>/welvet.so|dylib|dll|a` + `welvet.h` (+ `cabi_verify` when possible).

## Python mirror

```bash
./copy_to_python.sh
# or: apps/w2a/python/copy_from_cabi.sh
```

## Exports (Welvet-native / WASM parity)

- Grid: create/free, `WelvetPlace` + all `WelvetPlace*` (dense…gdn), `Forward`/`ForwardEx`, `TrainSGD`/`TrainSGDEx`/`TrainTween`/`TrainMesh`, convertDense, entity, DNA, `PermutationOK`
- Cameral: bicameral / hemispheres / parallel, setChild/BranchModes, TrainStackMSE / TrainParallelMSE
- Lists + `WelvetRunSuite` / `WelvetRunAllSuites`

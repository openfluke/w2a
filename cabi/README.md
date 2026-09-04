# Welvet C-ABI (apps/w2a/cabi)

Go `c-shared` / `c-archive` bridge for Welvet **1.1.1**. Mirrors Loom’s polyglot builder layout; exports mirror WASM `place*` + suites.

## Build

```bash
cd w2a/cabi

./build_macos.sh                 # native macOS dylib (+ cabi_verify)
./build_macos.sh universal       # arm64 + amd64 + lipo → macos_universal

./build_windows_from_mac.sh      # Windows amd64 DLL via mingw (brew install mingw-w64)
./build_android.sh               # Android arm64 + x86_64 (.so) via NDK
./build_ios.sh                   # iOS arm64 static archive (.a) via Xcode

# Or drive the builder directly:
cd internal/build
./build_linux.sh --test          # native linux
./build_linux.sh all             # amd64 + arm64
./build_unix.sh darwin arm64     # macOS
./build_unix.sh windows amd64    # needs mingw
./build_android.sh               # needs ANDROID_NDK_HOME / ANDROID_HOME
./build_ios.sh                   # macOS + Xcode only
./build_all.sh                   # master try — soft-SKIP missing toolchains
```

Outputs: `internal/build/dist/<os>_<arch>/welvet.so|dylib|dll|a` + `welvet.h` (+ `cabi_verify` when possible).

## Python mirror

```bash
./copy_to_python.sh
# or: apps/w2a/python/copy_from_cabi.sh
```

## Exports (Welvet-native / WASM parity)

- Grid: create/free, `WelvetPlace` + all `WelvetPlace*` (dense…gdn), `Forward`/`ForwardEx`, `TrainSGD`/`TrainSGDEx`/`TrainTween`/`TrainMesh`, convertDense, entity, DNA, `PermutationOK`
- Cameral: bicameral / hemispheres / parallel, setChild/BranchModes, TrainStackMSE / TrainParallelMSE
- Lists + `WelvetRunSuite` / `WelvetRunAllSuites`

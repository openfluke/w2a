"""Locate and load the Welvet C-ABI shared library."""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

_LIB = None


def _platform_dir() -> str:
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64"
    else:
        arch = machine
    if sysname == "linux":
        return f"linux_{arch}"
    if sysname == "darwin":
        return f"macos_{arch}"
    if sysname == "windows":
        return f"windows_{arch}"
    raise RuntimeError(f"unsupported platform {sysname}/{machine}")


def _lib_name() -> str:
    sysname = platform.system().lower()
    if sysname == "windows":
        return "welvet.dll"
    if sysname == "darwin":
        return "welvet.dylib"
    return "welvet.so"


def lib_path() -> Path:
    here = Path(__file__).resolve().parent
    p = here / _platform_dir() / _lib_name()
    if p.is_file():
        return p
    # fallback: any sibling platform dir (dev)
    for cand in here.glob(f"*/{_lib_name()}"):
        return cand
    raise FileNotFoundError(
        f"Welvet C-ABI not found at {p}. Build: "
        "cd apps/w2a/cabi/internal/build && ./build_linux.sh && ./copy_to_python.sh"
    )


def _pin_native_env(lib_dir: Path) -> None:
    """Point WELVET_*_LIBRARY at sibling accel .so/.dll/.dylib next to welvet.so."""
    sysname = platform.system().lower()
    pairs = [
        ("WELVET_CUDA_LIBRARY", "libwelvet_cuda.so" if sysname != "windows" else "libwelvet_cuda.dll"),
        ("WELVET_ROCM_LIBRARY", "libwelvet_rocm.so"),
        ("WELVET_VULKAN_LIBRARY", "libwelvet_vulkan.so"),
        ("WELVET_METAL_LIBRARY", "libwelvet_metal.dylib"),
        ("WELVET_MLX_LIBRARY", "libwelvet_mlx.dylib"),
    ]
    for env, name in pairs:
        cand = lib_dir / name
        if cand.is_file():
            os.environ.setdefault(env, str(cand))
    # nativecpp / mlx are cgo-linked with @rpath; keep sibling dir on the loader path.
    siblings = []
    for name in (
        "libwelvet.so",
        "libwelvet.dylib",
        "libwelvet.dll",
        "libwelvet_mlx.dylib",
        "libwelvet_metal.dylib",
    ):
        if (lib_dir / name).is_file():
            siblings.append(name)
    if siblings:
        sep = ";" if sysname == "windows" else ":"
        if sysname == "windows":
            key = "PATH"
        elif sysname == "darwin":
            key = "DYLD_LIBRARY_PATH"
        else:
            key = "LD_LIBRARY_PATH"
        cur = os.environ.get(key, "")
        root = str(lib_dir)
        if root not in cur.split(sep):
            os.environ[key] = root + (sep + cur if cur else "")


def get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    path = lib_path()
    _pin_native_env(path.parent)
    if platform.system().lower() == "linux":
        _LIB = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    else:
        _LIB = ctypes.CDLL(str(path))
    _configure(_LIB)
    return _LIB


def _configure(lib: ctypes.CDLL) -> None:
    lib.FreeWelvetString.argtypes = [ctypes.c_void_p]
    lib.FreeWelvetString.restype = None

    lib.WelvetEngineVersion.argtypes = []
    lib.WelvetEngineVersion.restype = ctypes.c_void_p

    lib.WelvetCreateGrid.argtypes = [ctypes.c_char_p]
    lib.WelvetCreateGrid.restype = ctypes.c_longlong
    lib.WelvetFreeGrid.argtypes = [ctypes.c_longlong]
    lib.WelvetFreeGrid.restype = None

    lib.WelvetPlaceDense.argtypes = [ctypes.c_longlong, ctypes.c_char_p]
    lib.WelvetPlaceDense.restype = ctypes.c_void_p
    lib.WelvetPlace.argtypes = [ctypes.c_longlong, ctypes.c_char_p, ctypes.c_char_p]
    lib.WelvetPlace.restype = ctypes.c_void_p

    for _name in (
        "WelvetPlaceMHA", "WelvetPlaceSwiGLU", "WelvetPlaceRMSNorm", "WelvetPlaceLayerNorm",
        "WelvetPlaceEmbedding", "WelvetPlaceSoftmax", "WelvetPlaceSequential", "WelvetPlaceResidual",
        "WelvetPlaceCNN1", "WelvetPlaceCNN2", "WelvetPlaceCNN3", "WelvetPlaceRNN", "WelvetPlaceLSTM",
        "WelvetPlaceConvT1", "WelvetPlaceConvT2", "WelvetPlaceConvT3", "WelvetPlaceParallel",
        "WelvetPlaceStack", "WelvetPlaceKMeans", "WelvetPlaceMamba", "WelvetPlaceMetacognition",
        "WelvetPlaceGDN",
    ):
        fn = getattr(lib, _name)
        fn.argtypes = [ctypes.c_longlong, ctypes.c_char_p]
        fn.restype = ctypes.c_void_p

    lib.WelvetPermutationOK.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.WelvetPermutationOK.restype = ctypes.c_int

    lib.WelvetForward.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.WelvetForward.restype = ctypes.c_void_p
    lib.WelvetForwardEx.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.WelvetForwardEx.restype = ctypes.c_void_p

    lib.WelvetTrainSGD.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_double,
    ]
    lib.WelvetTrainSGD.restype = ctypes.c_void_p
    lib.WelvetTrainSGDEx.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_double,
    ]
    lib.WelvetTrainSGDEx.restype = ctypes.c_void_p
    lib.WelvetTrainTween.argtypes = lib.WelvetTrainSGDEx.argtypes
    lib.WelvetTrainTween.restype = ctypes.c_void_p
    lib.WelvetTrainMesh.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
    ]
    lib.WelvetTrainMesh.restype = ctypes.c_void_p

    lib.WelvetConvertDense.argtypes = [
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.WelvetConvertDense.restype = ctypes.c_void_p

    lib.WelvetEntityByteLen.argtypes = [ctypes.c_longlong]
    lib.WelvetEntityByteLen.restype = ctypes.c_int
    lib.WelvetSerializeEntity.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.WelvetSerializeEntity.restype = ctypes.c_void_p
    lib.WelvetDeserializeEntity.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
    lib.WelvetDeserializeEntity.restype = ctypes.c_longlong

    lib.WelvetListDTypes.argtypes = []
    lib.WelvetListDTypes.restype = ctypes.c_void_p
    lib.WelvetListFormats.argtypes = []
    lib.WelvetListFormats.restype = ctypes.c_void_p
    lib.WelvetListLayerTypes.argtypes = []
    lib.WelvetListLayerTypes.restype = ctypes.c_void_p
    lib.WelvetListNamedTrainModes.argtypes = []
    lib.WelvetListNamedTrainModes.restype = ctypes.c_void_p
    lib.WelvetListSuiteCatalog.argtypes = []
    lib.WelvetListSuiteCatalog.restype = ctypes.c_void_p
    lib.WelvetListBackends.argtypes = []
    lib.WelvetListBackends.restype = ctypes.c_void_p
    lib.WelvetProbeBackends.argtypes = []
    lib.WelvetProbeBackends.restype = ctypes.c_void_p
    lib.WelvetLastError.argtypes = []
    lib.WelvetLastError.restype = ctypes.c_void_p
    lib.WelvetEnableBackend.argtypes = [ctypes.c_char_p]
    lib.WelvetEnableBackend.restype = ctypes.c_void_p
    lib.WelvetSetGridBackend.argtypes = [ctypes.c_longlong, ctypes.c_char_p]
    lib.WelvetSetGridBackend.restype = ctypes.c_void_p
    lib.WelvetPrepareGrid.argtypes = [ctypes.c_longlong]
    lib.WelvetPrepareGrid.restype = ctypes.c_void_p

    lib.WelvetCreateBicameral.argtypes = [ctypes.c_char_p]
    lib.WelvetCreateBicameral.restype = ctypes.c_longlong
    lib.WelvetCreateHemispheres.argtypes = [ctypes.c_char_p]
    lib.WelvetCreateHemispheres.restype = ctypes.c_longlong
    lib.WelvetCreateParallel.argtypes = [ctypes.c_char_p]
    lib.WelvetCreateParallel.restype = ctypes.c_longlong
    lib.WelvetTrainStackMSE.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_double,
    ]
    lib.WelvetTrainStackMSE.restype = ctypes.c_void_p
    lib.WelvetTrainParallelMSE.argtypes = lib.WelvetTrainStackMSE.argtypes
    lib.WelvetTrainParallelMSE.restype = ctypes.c_void_p
    lib.WelvetSetChildModes.argtypes = [ctypes.c_longlong, ctypes.c_char_p]
    lib.WelvetSetChildModes.restype = ctypes.c_void_p
    lib.WelvetSetBranchModes.argtypes = [ctypes.c_longlong, ctypes.c_char_p]
    lib.WelvetSetBranchModes.restype = ctypes.c_void_p
    lib.WelvetFreeStack.argtypes = [ctypes.c_longlong]
    lib.WelvetFreeStack.restype = None
    lib.WelvetFreeParallel.argtypes = [ctypes.c_longlong]
    lib.WelvetFreeParallel.restype = None

    lib.WelvetRunSuite.argtypes = [ctypes.c_char_p]
    lib.WelvetRunSuite.restype = ctypes.c_void_p
    lib.WelvetRunAllSuites.argtypes = [ctypes.c_char_p]
    lib.WelvetRunAllSuites.restype = ctypes.c_void_p


def c_str(ptr) -> str:
    if not ptr:
        return ""
    lib = get_lib()
    raw = ctypes.cast(ptr, ctypes.c_char_p).value
    s = raw.decode("utf-8") if raw else ""
    lib.FreeWelvetString(ptr)
    return s

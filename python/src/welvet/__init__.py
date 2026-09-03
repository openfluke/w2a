"""Welvet 1.1.1 — Python C-ABI bindings."""
from __future__ import annotations

import ctypes
import json
from array import array
from typing import Any, Sequence

from ._lib import c_str, get_lib

__version__ = "1.1.1"
ENGINE_VERSION = "1.1.1"

PLACE_KINDS = (
    "dense", "mha", "swiglu", "rmsnorm", "layernorm", "embedding", "softmax",
    "sequential", "residual", "cnn1", "cnn2", "cnn3", "rnn", "lstm",
    "convt1", "convt2", "convt3", "parallel", "stack", "kmeans", "mamba",
    "metacognition", "gdn",
)


def engine_version() -> str:
    return c_str(get_lib().WelvetEngineVersion())


def assert_engine_version() -> None:
    v = engine_version()
    if v != ENGINE_VERSION:
        raise RuntimeError(f"CABI version {v} != package {ENGINE_VERSION}")


def _f32(seq: Sequence[float]) -> array:
    return array("f", [float(x) for x in seq])


def _encode(spec: dict[str, Any] | str) -> bytes:
    return spec.encode() if isinstance(spec, str) else json.dumps(spec).encode()


def create_grid(
    depth: int = 1,
    rows: int = 1,
    cols: int = 1,
    layers_per_cell: int = 1,
    backend: str = "cpu_tiled",
) -> int:
    cfg = json.dumps(
        {
            "depth": depth,
            "rows": rows,
            "cols": cols,
            "layers_per_cell": layers_per_cell,
            "backend": backend,
        }
    ).encode()
    h = int(get_lib().WelvetCreateGrid(cfg))
    if h == 0:
        raise RuntimeError("WelvetCreateGrid failed")
    return h


def free_grid(handle: int) -> None:
    get_lib().WelvetFreeGrid(handle)


def place(handle: int, kind: str, spec: dict[str, Any] | str) -> dict:
    """Place any layer kind (dense, mha, cnn1, …) — WASM place* parity."""
    meta = json.loads(c_str(get_lib().WelvetPlace(handle, kind.encode(), _encode(spec))))
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return meta


def place_dense(handle: int, spec: dict[str, Any] | str) -> dict:
    return place(handle, "dense", spec)


def permutation_ok(kind: str, dtype: int, format: int, backend: int) -> bool:
    return bool(get_lib().WelvetPermutationOK(kind.encode(), dtype, format, backend))


def forward(
    handle: int,
    x: Sequence[float],
    shape: list[int] | None = None,
    out_cap: int | None = None,
) -> list[float]:
    lib = get_lib()
    xin = _f32(x)
    n = len(xin)
    cap = out_cap or max(n * 16, 256)
    in_buf = (ctypes.c_float * n)(*xin)
    out_buf = (ctypes.c_float * cap)()
    shape_b = json.dumps(shape).encode() if shape else None
    if shape_b is not None:
        meta = json.loads(c_str(lib.WelvetForwardEx(handle, in_buf, n, shape_b, out_buf, cap)))
    else:
        meta = json.loads(c_str(lib.WelvetForward(handle, in_buf, n, out_buf, cap)))
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return list(out_buf)[: int(meta.get("len", 0))]


def train_sgd(
    handle: int,
    x: Sequence[float],
    y: Sequence[float],
    lr: float = 0.05,
    shape: list[int] | None = None,
) -> dict:
    lib = get_lib()
    xin, yin = _f32(x), _f32(y)
    in_buf = (ctypes.c_float * len(xin))(*xin)
    tgt_buf = (ctypes.c_float * len(yin))(*yin)
    shape_b = json.dumps(shape).encode() if shape else None
    if shape_b is not None:
        meta = json.loads(
            c_str(lib.WelvetTrainSGDEx(handle, in_buf, len(xin), shape_b, tgt_buf, len(yin), float(lr)))
        )
    else:
        meta = json.loads(
            c_str(lib.WelvetTrainSGD(handle, in_buf, len(xin), tgt_buf, len(yin), float(lr)))
        )
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return meta


def train_tween(
    handle: int,
    x: Sequence[float],
    y: Sequence[float],
    lr: float = 0.05,
    shape: list[int] | None = None,
) -> dict:
    lib = get_lib()
    xin, yin = _f32(x), _f32(y)
    in_buf = (ctypes.c_float * len(xin))(*xin)
    tgt_buf = (ctypes.c_float * len(yin))(*yin)
    shape_b = json.dumps(shape).encode() if shape else b""
    meta = json.loads(
        c_str(lib.WelvetTrainTween(handle, in_buf, len(xin), shape_b, tgt_buf, len(yin), float(lr)))
    )
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return meta


def convert_dense(handle: int, dtype: int, format: int = 0, z=0, y=0, x=0, l=0) -> dict:
    meta = json.loads(c_str(get_lib().WelvetConvertDense(handle, dtype, format, z, y, x, l)))
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return meta


def serialize_entity(handle: int) -> bytes:
    lib = get_lib()
    n = int(lib.WelvetEntityByteLen(handle))
    if n <= 0:
        raise RuntimeError("WelvetEntityByteLen failed")
    buf = (ctypes.c_ubyte * n)()
    out_len = ctypes.c_int(0)
    meta = json.loads(c_str(lib.WelvetSerializeEntity(handle, buf, n, ctypes.byref(out_len))))
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return bytes(buf[: int(out_len.value)])


def deserialize_entity(data: bytes) -> int:
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    h = int(get_lib().WelvetDeserializeEntity(buf, len(data)))
    if h == 0:
        raise RuntimeError("WelvetDeserializeEntity failed")
    return h


def list_dtypes() -> list[dict]:
    return json.loads(c_str(get_lib().WelvetListDTypes()))


def list_formats() -> list[dict]:
    return json.loads(c_str(get_lib().WelvetListFormats()))


def list_layer_types() -> list[str]:
    return json.loads(c_str(get_lib().WelvetListLayerTypes()))


def list_named_train_modes() -> list[str]:
    return json.loads(c_str(get_lib().WelvetListNamedTrainModes()))


def list_suite_catalog() -> list[str]:
    return json.loads(c_str(get_lib().WelvetListSuiteCatalog()))


def create_bicameral(
    in_: int = 4, hidden: int = 4, out: int = 4, dtype: int = 1, format: int = 0
) -> int:
    cfg = json.dumps({"in": in_, "hidden": hidden, "out": out, "dtype": dtype, "format": format}).encode()
    h = int(get_lib().WelvetCreateBicameral(cfg))
    if h == 0:
        raise RuntimeError("WelvetCreateBicameral failed")
    return h


def create_hemispheres(dim: int = 4, n: int = 2, combine: str = "add", dtype: int = 1) -> int:
    cfg = json.dumps({"dim": dim, "n": n, "combine": combine, "dtype": dtype}).encode()
    h = int(get_lib().WelvetCreateHemispheres(cfg))
    if h == 0:
        raise RuntimeError("WelvetCreateHemispheres failed")
    return h


def create_parallel(
    dim: int = 8, out_feat: int | None = None, branches: int = 2, combine: str = "add",
    dtype: int = 1, format: int = 0,
) -> int:
    cfg = json.dumps({
        "Dim": dim, "OutFeat": out_feat or dim, "Branches": branches, "Combine": combine,
        "dtype": dtype, "format": format,
    }).encode()
    h = int(get_lib().WelvetCreateParallel(cfg))
    if h == 0:
        raise RuntimeError("WelvetCreateParallel failed")
    return h


def set_child_modes(handle: int, modes: list[str]) -> dict:
    return json.loads(c_str(get_lib().WelvetSetChildModes(handle, json.dumps(modes).encode())))


def set_branch_modes(handle: int, modes: list[str]) -> dict:
    return json.loads(c_str(get_lib().WelvetSetBranchModes(handle, json.dumps(modes).encode())))


def train_stack_mse(
    handle: int, x: Sequence[float], y: Sequence[float], mode: str, lr: float = 0.05
) -> dict:
    lib = get_lib()
    xin, yin = _f32(x), _f32(y)
    in_buf = (ctypes.c_float * len(xin))(*xin)
    tgt_buf = (ctypes.c_float * len(yin))(*yin)
    meta = json.loads(
        c_str(lib.WelvetTrainStackMSE(handle, in_buf, len(xin), tgt_buf, len(yin), mode.encode(), float(lr)))
    )
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return meta


def train_parallel_mse(
    handle: int, x: Sequence[float], y: Sequence[float], mode: str, lr: float = 0.05
) -> dict:
    lib = get_lib()
    xin, yin = _f32(x), _f32(y)
    in_buf = (ctypes.c_float * len(xin))(*xin)
    tgt_buf = (ctypes.c_float * len(yin))(*yin)
    meta = json.loads(
        c_str(lib.WelvetTrainParallelMSE(handle, in_buf, len(xin), tgt_buf, len(yin), mode.encode(), float(lr)))
    )
    if meta.get("error"):
        raise RuntimeError(meta["error"])
    return meta


def free_stack(handle: int) -> None:
    get_lib().WelvetFreeStack(handle)


def free_parallel(handle: int) -> None:
    get_lib().WelvetFreeParallel(handle)


def run_suite(name: str) -> dict:
    return json.loads(c_str(get_lib().WelvetRunSuite(name.encode())))


def run_all_suites(
    *, quick: bool = False, only: list[str] | None = None, skip: list[str] | None = None
) -> dict:
    flags = {"quick": quick, "only": only or [], "skip": skip or []}
    return json.loads(c_str(get_lib().WelvetRunAllSuites(json.dumps(flags).encode())))


class DType:
    FLOAT64 = 0
    FLOAT32 = 1
    FLOAT16 = 2
    BFLOAT16 = 3
    INT8 = 9
    FP4 = 16

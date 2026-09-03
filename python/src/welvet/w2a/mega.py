"""Python mega matrix: layers×dtype×format×backend×ops + cameral (TS mega parity)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import welvet as w

Status = Literal["OK", "FAIL", "SKIP"]

BACKENDS = [
    {"id": 0, "name": "cpu_tiled", "json": "cpu_tiled"},
    {"id": 1, "name": "simd", "json": "simd"},
    {"id": 2, "name": "webgpu", "json": "webgpu"},
]

# Mirrors typescript/src/suites/catalog.ts LAYER_DEFS
LAYER_DEFS: list[dict[str, Any]] = [
    {"id": "dense", "spec": {"in": 8, "out": 8, "act": "relu"}, "inLen": 8, "outLen": 8},
    {"id": "mha", "spec": {"dim": 8, "DModel": 8, "NumHeads": 2, "SeqLen": 4}, "inLen": 32, "outLen": 32, "shape": [1, 4, 8]},
    {"id": "swiglu", "spec": {"dim": 8, "InputDim": 8, "IntermediateDim": 16}, "inLen": 8, "outLen": 8},
    {"id": "rmsnorm", "spec": {"dim": 8}, "inLen": 8, "outLen": 8},
    {"id": "layernorm", "spec": {"dim": 8}, "inLen": 8, "outLen": 8},
    {"id": "embedding", "spec": {"VocabSize": 32, "EmbeddingDim": 8, "SeqLen": 4}, "inLen": 4, "outLen": 32, "shape": [1, 4]},
    {"id": "softmax", "spec": {"dim": 8}, "inLen": 8, "outLen": 8, "skipTrain": True},
    {"id": "sequential", "spec": {"dim": 8, "Depth": 2}, "inLen": 8, "outLen": 8},
    {"id": "residual", "spec": {"dim": 8, "Depth": 1}, "inLen": 8, "outLen": 8},
    {"id": "cnn1", "spec": {"InChannels": 1, "Filters": 4, "SeqLen": 16, "Kernel": 3}, "inLen": 16, "outLen": 56, "shape": [1, 1, 16]},
    {"id": "cnn2", "spec": {"InChannels": 1, "Filters": 4, "Height": 8, "Width": 8, "Kernel": 3}, "inLen": 64, "outLen": 144, "shape": [1, 1, 8, 8]},
    {"id": "cnn3", "spec": {"InChannels": 1, "Filters": 2, "Depth": 4, "Height": 4, "Width": 4, "Kernel": 3}, "inLen": 64, "outLen": 16, "shape": [1, 1, 4, 4, 4]},
    {"id": "rnn", "spec": {"InputSize": 8, "HiddenSize": 8, "SeqLen": 4}, "inLen": 32, "outLen": 32, "shape": [1, 4, 8]},
    {"id": "lstm", "spec": {"InputSize": 8, "HiddenSize": 8, "SeqLen": 4}, "inLen": 32, "outLen": 32, "shape": [1, 4, 8]},
    {"id": "convt1", "spec": {"InChannels": 4, "Filters": 2, "SeqLen": 8, "Kernel": 3}, "inLen": 32, "outLen": 20, "shape": [1, 4, 8]},
    {"id": "convt2", "spec": {"InChannels": 4, "Filters": 2, "Height": 4, "Width": 4, "Kernel": 3}, "inLen": 64, "outLen": 72, "shape": [1, 4, 4, 4]},
    {"id": "convt3", "spec": {"InChannels": 2, "Filters": 2, "Depth": 4, "Height": 4, "Width": 4, "Kernel": 3}, "inLen": 128, "outLen": 250, "shape": [1, 2, 4, 4, 4]},
    {"id": "parallel", "spec": {"dim": 8, "OutFeat": 8, "Branches": 2}, "inLen": 8, "outLen": 8},
    {"id": "stack", "spec": {"dim": 8, "act": "relu"}, "inLen": 8, "outLen": 8},
    {"id": "kmeans", "spec": {"FeatureDim": 8, "NumClusters": 4}, "inLen": 8, "outLen": 4},
    {"id": "mamba", "spec": {"DModel": 8, "DState": 8, "SeqLen": 4}, "inLen": 32, "outLen": 32, "shape": [1, 4, 8]},
    {"id": "metacognition", "spec": {"Dim": 8}, "inLen": 8, "outLen": 8},
    {"id": "gdn", "spec": {"HiddenSize": 8, "NumKeyHeads": 2, "NumValueHeads": 2, "KeyHeadDim": 4, "ValueHeadDim": 4, "ConvKernel": 3}, "inLen": 8, "outLen": 8, "shape": [1, 1, 8]},
]


@dataclass
class Report:
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    fails: list[str] = field(default_factory=list)
    progress_every: int = 500
    _n: int = 0

    def record(self, suite: str, name: str, status: Status, note: str = "") -> None:
        if status == "OK":
            self.ok += 1
        elif status == "FAIL":
            self.failed += 1
            self.fails.append(f"{suite}/{name}: {note}")
            print(f"  FAIL {suite}/{name}: {note}")
        else:
            self.skipped += 1
        self._n += 1
        if self.progress_every and self._n % self.progress_every == 0:
            print(f"  … {self._n} cells (ok={self.ok} fail={self.failed} skip={self.skipped})")

    def summary(self) -> str:
        lines = [f"Python matrix: ok={self.ok} fail={self.failed} skip={self.skipped} total={self._n}"]
        if self.fails[:15]:
            lines.append("First fails:")
            lines.extend(f"  - {f}" for f in self.fails[:15])
        return "\n".join(lines)


def _fill(n: int, v: float = 0.1) -> list[float]:
    return [v * ((i % 7) + 1) for i in range(n)]


def _classify(msg: str) -> Status:
    m = (msg or "").lower()
    if any(k in m for k in ("unsupported", "not available", "webgpu", "native-only", "permission", "permutation")):
        return "SKIP"
    return "FAIL"


def _place_body(layer: dict, dtype: int, format: int) -> dict:
    spec = dict(layer["spec"])
    in_len, out_len = layer["inLen"], layer["outLen"]
    if format == 20 and layer["id"] == "dense":
        spec["in"] = 64
        spec["out"] = 64
        in_len = out_len = 64
    body = {"z": 0, "y": 0, "x": 0, "l": 0, "dtype": dtype, "format": format, "act": "relu", **spec}
    return body, in_len, out_len


def run_layer_cell(
    report: Report,
    layer: dict,
    dt: dict,
    fmt: dict,
    be: dict,
    *,
    split: bool = True,
) -> None:
    base = f"{layer['id']}/dt={dt['name']}/fmt={fmt['name']}/{be['name']}"
    if not w.permutation_ok(layer["id"], dt["id"], fmt["id"], be["id"]):
        for op in ("place", "forward", "train", "entity") if split else ("all",):
            report.record("mega_layers", f"{base}/{op}" if split else base, "SKIP", "PermutationOK=false")
        return

    g = None
    try:
        g = w.create_grid(backend=be["json"])
        body, in_len, out_len = _place_body(layer, dt["id"], fmt["id"])
        try:
            w.place(g, layer["id"], body)
            report.record("mega_layers", f"{base}/place", "OK")
        except Exception as e:  # noqa: BLE001
            st = _classify(str(e))
            report.record("mega_layers", f"{base}/place", st, str(e))
            for op in ("forward", "train", "entity"):
                report.record("mega_layers", f"{base}/{op}", "SKIP", "place failed")
            return

        shape = None if fmt["id"] == 20 else layer.get("shape")
        inp = _fill(in_len)
        train_out = out_len
        try:
            out = w.forward(g, inp, shape=shape)
            report.record("mega_layers", f"{base}/forward", "OK")
            if out:
                train_out = len(out)
        except Exception as e:  # noqa: BLE001
            report.record("mega_layers", f"{base}/forward", _classify(str(e)), str(e))

        if layer.get("skipTrain"):
            report.record("mega_layers", f"{base}/train", "SKIP", "skipTrain")
        else:
            try:
                w.train_sgd(g, inp, _fill(train_out, 0.0), lr=0.01, shape=shape)
                report.record("mega_layers", f"{base}/train", "OK")
            except Exception as e:  # noqa: BLE001
                report.record("mega_layers", f"{base}/train", _classify(str(e)), str(e))

        try:
            blob = w.serialize_entity(g)
            if len(blob) < 4:
                raise RuntimeError("entity too short")
            h2 = w.deserialize_entity(blob)
            w.free_grid(h2)
            report.record("mega_layers", f"{base}/entity", "OK")
        except Exception as e:  # noqa: BLE001
            report.record("mega_layers", f"{base}/entity", _classify(str(e)), str(e))
    finally:
        if g is not None:
            w.free_grid(g)


def run_layers_matrix(
    report: Report,
    *,
    layers: list[dict],
    dtypes: list[dict],
    formats: list[dict],
    backends: list[dict],
) -> None:
    n = len(layers) * len(dtypes) * len(formats) * len(backends) * 4
    print(f"\n## layers×dtype×format×backend×ops → ~{n} cells")
    for layer in layers:
        for be in backends:
            for dt in dtypes:
                for fmt in formats:
                    run_layer_cell(report, layer, dt, fmt, be)


def run_cameral_matrix(
    report: Report,
    *,
    modes: list[str] | None = None,
    dtypes: list[dict] | None = None,
    formats: list[dict] | None = None,
    backends: list[dict] | None = None,
) -> None:
    modes = modes or w.list_named_train_modes()
    dtypes = dtypes or [{"id": 1, "name": "float32"}]
    formats = formats or [{"id": 0, "name": "none"}]
    backends = backends or [BACKENDS[0]]
    arches = ("bicameral", "hemi2", "hemi3", "parallel")
    print(f"\n## cameral arches×modes×dtype/format×backend")

    for arch in arches:
        for mode in modes:
            for dt in dtypes:
                for be in backends:
                    name = f"{arch}/{mode}/dt={dt['name']}/none/{be['name']}"
                    _cameral_cell(report, arch, mode, dt["id"], 0, be, name)
            for fmt in formats:
                for be in backends:
                    name = f"{arch}/{mode}/dt=float32/fmt={fmt['name']}/{be['name']}"
                    _cameral_cell(report, arch, mode, 1, fmt["id"], be, name)


def _cameral_cell(report: Report, arch: str, mode: str, dtype: int, format: int, be: dict, name: str) -> None:
    if be["name"] == "webgpu":
        report.record("mega_cameral", name, "SKIP", "webgpu native-only in Python matrix")
        return
    dim = 64 if format == 20 else 4
    h = None
    kind = "stack"
    try:
        if arch == "bicameral":
            h = w.create_bicameral(in_=dim, hidden=dim, out=dim, dtype=dtype, format=format)
            w.set_child_modes(h, [mode, mode])
            tgt = _fill(dim, 0.0)
            tgt[0] = 1.0
            w.train_stack_mse(h, _fill(dim), tgt, mode, lr=0.05)
            kind = "stack"
        elif arch.startswith("hemi"):
            n = 3 if arch == "hemi3" else 2
            h = w.create_hemispheres(dim=dim, n=n, combine="add", dtype=dtype)
            w.set_branch_modes(h, [mode] * n)
            w.train_parallel_mse(h, _fill(dim), _fill(dim, 0.0), mode, lr=0.05)
            kind = "parallel"
        else:
            h = w.create_parallel(dim=dim, branches=2, dtype=dtype, format=format)
            w.set_branch_modes(h, [mode, mode])
            w.train_parallel_mse(h, _fill(dim), _fill(dim, 0.0), mode, lr=0.05)
            kind = "parallel"
        report.record("mega_cameral", name, "OK")
    except Exception as e:  # noqa: BLE001
        report.record("mega_cameral", name, _classify(str(e)), str(e))
    finally:
        if h is not None:
            if kind == "stack":
                w.free_stack(h)
            else:
                w.free_parallel(h)


def run_quick_matrix() -> Report:
    report = Report(progress_every=50)
    layers = [LAYER_DEFS[0], LAYER_DEFS[1], LAYER_DEFS[2]]  # dense, mha, swiglu
    dtypes = [d for d in w.list_dtypes() if d.get("name") in ("float32", "float16", "int8")]
    formats = [f for f in w.list_formats() if f.get("id") == 0] or [{"id": 0, "name": "none"}]
    run_layers_matrix(report, layers=layers, dtypes=dtypes or [{"id": 1, "name": "float32"}], formats=formats, backends=[BACKENDS[0]])
    run_cameral_matrix(report, modes=(w.list_named_train_modes() or ["SGD"])[:3], backends=[BACKENDS[0]])
    return report


def run_mid_matrix() -> Report:
    report = Report(progress_every=200)
    dtypes = w.list_dtypes()
    formats = [f for f in w.list_formats() if f.get("id") in (0, 1, 2, 3, 20)] or w.list_formats()[:5]
    run_layers_matrix(report, layers=LAYER_DEFS, dtypes=dtypes, formats=formats, backends=BACKENDS[:2])
    run_cameral_matrix(
        report,
        modes=w.list_named_train_modes()[:8],
        dtypes=[{"id": 1, "name": "float32"}],
        formats=[{"id": 0, "name": "none"}],
        backends=BACKENDS[:1],
    )
    return report


def run_mega_matrix() -> Report:
    report = Report(progress_every=1000)
    run_layers_matrix(
        report,
        layers=LAYER_DEFS,
        dtypes=w.list_dtypes(),
        formats=w.list_formats(),
        backends=BACKENDS,
    )
    run_cameral_matrix(
        report,
        modes=w.list_named_train_modes(),
        dtypes=[{"id": 1, "name": "float32"}],
        formats=w.list_formats(),
        backends=BACKENDS,
    )
    return report

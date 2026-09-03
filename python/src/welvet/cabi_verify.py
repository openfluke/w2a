"""Verify Welvet C-ABI symbols + basic dense smoke."""
from __future__ import annotations

import sys

from welvet import (
    assert_engine_version,
    create_grid,
    engine_version,
    forward,
    free_grid,
    list_dtypes,
    list_named_train_modes,
    list_suite_catalog,
    place_dense,
    serialize_entity,
    train_sgd,
)
from welvet._lib import get_lib, lib_path


REQUIRED = [
    "FreeWelvetString",
    "WelvetEngineVersion",
    "WelvetCreateGrid",
    "WelvetFreeGrid",
    "WelvetPlace",
    "WelvetPlaceDense",
    "WelvetPlaceMHA",
    "WelvetPlaceSwiGLU",
    "WelvetPlaceCNN1",
    "WelvetPlaceGDN",
    "WelvetPermutationOK",
    "WelvetForward",
    "WelvetForwardEx",
    "WelvetTrainSGD",
    "WelvetTrainSGDEx",
    "WelvetTrainTween",
    "WelvetTrainMesh",
    "WelvetConvertDense",
    "WelvetSerializeEntity",
    "WelvetDeserializeEntity",
    "WelvetListDTypes",
    "WelvetListFormats",
    "WelvetListLayerTypes",
    "WelvetListNamedTrainModes",
    "WelvetCreateBicameral",
    "WelvetCreateHemispheres",
    "WelvetCreateParallel",
    "WelvetTrainStackMSE",
    "WelvetTrainParallelMSE",
    "WelvetSetChildModes",
    "WelvetSetBranchModes",
    "WelvetListSuiteCatalog",
    "WelvetRunSuite",
    "WelvetRunAllSuites",
]


def main() -> int:
    print("=== Welvet Python C-ABI Verification ===")
    path = lib_path()
    print(f"[*] Library: {path}")
    lib = get_lib()
    missing = [s for s in REQUIRED if not hasattr(lib, s)]
    print(f"[+] Required symbols: {len(REQUIRED) - len(missing)} / {len(REQUIRED)}")
    if missing:
        print("[!] Missing:", ", ".join(missing))
        return 1

    ver = engine_version()
    print(f"[+] Engine version: {ver}")
    assert_engine_version()

    g = create_grid()
    try:
        place_dense(g, {"in": 4, "out": 4, "act": "relu", "dtype": 1})
        out = forward(g, [0.1, 0.2, 0.3, 0.4])
        print(f"[+] Forward len={len(out)} sample={out[:2]}")
        meta = train_sgd(g, [0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 1.0], lr=0.05)
        print(f"[+] TrainSGD loss={meta.get('loss')}")
        ent = serialize_entity(g)
        print(f"[+] Entity bytes={len(ent)}")
    finally:
        free_grid(g)

    g2 = create_grid()
    try:
        from welvet import place

        place(g2, "mha", {"dim": 8, "DModel": 8, "NumHeads": 2, "SeqLen": 4, "dtype": 1})
        mout = forward(g2, [0.1] * 32, shape=[1, 4, 8])
        print(f"[+] placeMHA forward len={len(mout)}")
    finally:
        free_grid(g2)

    print(f"[+] DTypes: {len(list_dtypes())}")
    print(f"[+] TrainModes: {len(list_named_train_modes())}")
    print(f"[+] Suite catalog: {len(list_suite_catalog())}")
    print("[+] cabi_verify OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

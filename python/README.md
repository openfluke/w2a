# Welvet Python (C-ABI) 1.1.1

ctypes bindings over the Welvet shared library built from `apps/w2a/cabi`.

## Setup

```bash
cd apps/w2a/cabi/internal/build
./build_linux.sh --test
./copy_to_python.sh

cd ../../python
pip install -e .   # or: PYTHONPATH=src
python -m welvet.cabi_verify
python -m welvet.w2a run --quick
```


## API

```python
import welvet as w

g = w.create_grid()
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

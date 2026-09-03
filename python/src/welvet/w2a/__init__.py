"""w2a suite runners: Go suites via CABI + Python mega matrix."""
from __future__ import annotations

from .cli import main
from .mega import run_mega_matrix, run_mid_matrix, run_quick_matrix

__all__ = ["main", "run_mega_matrix", "run_mid_matrix", "run_quick_matrix"]

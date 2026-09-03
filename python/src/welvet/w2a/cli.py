"""CLI: python -m welvet.w2a run --quick|--mid|--mega|--all"""
from __future__ import annotations

import argparse
import json
import sys

import welvet as w

from .mega import run_mega_matrix, run_mid_matrix, run_quick_matrix


def _print_go(result: dict) -> int:
    print(
        f"Go suites: ok={result.get('ok')} passed={result.get('passed')} "
        f"failed={result.get('failed')} skipped={result.get('skipped')} "
        f"elapsed_ms={result.get('elapsed_ms')}"
    )
    for row in result.get("suites") or []:
        status = "SKIP" if row.get("skipped") else ("OK" if row.get("ok") else "FAIL")
        extra = f"  {row.get('error', '')}" if not row.get("ok") and not row.get("skipped") else ""
        print(f"  [{status}] {row.get('suite')}{extra}")
    return 0 if result.get("ok") else 1


def cmd_catalog(_: argparse.Namespace) -> int:
    print(json.dumps(w.list_suite_catalog(), indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    w.assert_engine_version()
    rc = 0

    if args.quick:
        go = w.run_all_suites(quick=True)
        rc |= _print_go(go)
        report = run_quick_matrix()
        print(report.summary())
        rc |= 0 if report.failed == 0 else 1
        return rc

    if args.mid:
        go = w.run_all_suites(
            only=[
                "seed", "serialization", "helpers", "memory", "fountain",
                "weights", "step", "tween", "dna",
            ]
        )
        rc |= _print_go(go)
        report = run_mid_matrix()
        print(report.summary())
        rc |= 0 if report.failed == 0 else 1
        return rc

    if args.mega:
        go = w.run_all_suites(skip=["donate", "hardware"])
        rc |= _print_go(go)
        report = run_mega_matrix()
        print(report.summary())
        rc |= 0 if report.failed == 0 else 1
        return rc

    # --all: Go suites only
    go = w.run_all_suites(skip=args.skip or [])
    return _print_go(go)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m welvet.w2a", description="Welvet w2a suites")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("catalog", help="List Go suite names via CABI")
    c.set_defaults(func=cmd_catalog)

    r = sub.add_parser("run", help="Run suites")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="Go smoke + Python dense quick matrix")
    g.add_argument("--mid", action="store_true", help="Go mid subset + denser Python matrix")
    g.add_argument("--mega", action="store_true", help="All Go suites + full Python mega matrix")
    g.add_argument("--all", action="store_true", help="All Go suites via WelvetRunAllSuites")
    r.add_argument("--skip", nargs="*", default=[], help="Suite names to skip (--all)")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

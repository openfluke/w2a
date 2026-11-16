"""Download prebuilt CABI libs from openfluke/w2a GitHub Releases into this package."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from . import ENGINE_VERSION

REPO = "openfluke/w2a"


def platform_dir() -> str:
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


def asset_name(plat: str, version: str) -> str:
    return f"welvet-cabi-{plat}-{version}.tar.gz"


def download(url: str, dest: Path) -> None:
    print(f"→ GET {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def fetch(tag: str = "latest", version: str | None = None) -> Path:
    ver = version or ENGINE_VERSION
    plat = platform_dir()
    name = asset_name(plat, ver)
    dest_dir = Path(__file__).resolve().parent / plat
    dest_dir.mkdir(parents=True, exist_ok=True)

    if tag == "latest":
        url = f"https://github.com/{REPO}/releases/latest/download/{name}"
    else:
        url = f"https://github.com/{REPO}/releases/download/{tag}/{name}"

    with tempfile.TemporaryDirectory(prefix="welvet-cabi-") as tmp:
        archive = Path(tmp) / name
        try:
            download(url, archive)
        except Exception as e:
            raise RuntimeError(
                f"failed to download {name} from {url}: {e}\n"
                "Build locally: cd w2a && ./scripts/pack_cabi_linux.sh --build"
            ) from e
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)

    for cand in (dest_dir / "welvet.so", dest_dir / "welvet.dylib", dest_dir / "welvet.dll"):
        if cand.is_file():
            print(f"→ installed {cand} ({cand.stat().st_size / (1024 * 1024):.1f} MiB)")
            return dest_dir
    print(f"→ extracted into {dest_dir}")
    return dest_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag", nargs="?", default="latest", help="GitHub release tag (default: latest)")
    p.add_argument("--version", default=None, help=f"CABI ENGINE_VERSION in asset name (default: {ENGINE_VERSION})")
    args = p.parse_args(argv)
    fetch(tag=args.tag, version=args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())

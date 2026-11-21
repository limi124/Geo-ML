import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parents[1]
os.environ["JUPYTER_PATH"] = str(PROJECT / ".jupyter")


def run(path: Path, timeout: int = 2400, save: bool = True) -> None:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="geoml",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
    )
    t0 = time.time()
    print(f"[run] {path.name} kernel=geoml", flush=True)
    client.execute()
    print(f"[ok] {path.name} {time.time() - t0:.1f}s", flush=True)
    if save:
        nbformat.write(nb, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()
    for name in args.notebooks:
        run(PROJECT / "notebooks" / name, timeout=args.timeout)

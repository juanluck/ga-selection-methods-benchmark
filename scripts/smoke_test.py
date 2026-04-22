from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    outdir = ROOT / "results" / "smoke_test"
    if outdir.exists():
        shutil.rmtree(outdir)

    run(
        [
            sys.executable,
            "run_cec2017_bound_benchmarks.py",
            "--runs",
            "1",
            "--population-sizes",
            "100",
            "--dimension",
            "10",
            "--evals-per-dim",
            "100",
            "--algorithms",
            "ST",
            "BMT",
            "CS",
            "--jobs",
            "1",
            "--outdir",
            str(outdir),
        ]
    )

    print("\nSmoke test OK. Results written to:", outdir)


if __name__ == "__main__":
    main()

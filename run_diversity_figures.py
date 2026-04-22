from __future__ import annotations

import argparse
from pathlib import Path

from src.bmt_repro.diversity import generate_diversity_and_footprints


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate paper-like diversity and footprint figures for BMT vs ST.')
    parser.add_argument('--outdir', type=Path, default=Path('results/diversity_figures'))
    parser.add_argument('--generations', type=int, default=100)
    args = parser.parse_args()
    generate_diversity_and_footprints(args.outdir, generations=args.generations)
    print(args.outdir)


if __name__ == '__main__':
    main()

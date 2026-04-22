from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.bmt_repro.benchmarks import get_standard_benchmarks
from src.bmt_repro.ga import make_paper_config, run_suite
from src.bmt_repro.stats import summarize_median_std


def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep BMT bipolarity from 0.05 to 1.0 in steps of 0.05.')
    parser.add_argument('--outdir', type=Path, default=Path('results/bipolarity_sweep'))
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--runs', type=int, default=25)
    parser.add_argument('--population-size', type=int, default=100)
    args = parser.parse_args()

    cfg = make_paper_config()
    cfg.generations = args.generations
    cfg.runs = args.runs
    cfg.population_sizes = (args.population_size,)
    cfg.algorithms = ('BMT',)
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for step in range(1, 21):
        q = round(0.05 * step, 2)
        cfg.bipolarity = q
        raw = run_suite(get_standard_benchmarks(), cfg)
        raw['bipolarity'] = q
        all_rows.append(raw)
    raw_all = pd.concat(all_rows, ignore_index=True)
    summary = (
        raw_all.groupby(['bipolarity', 'problem'])['best_value']
        .agg(median='median', std='std', mean='mean')
        .reset_index()
    )
    raw_all.to_csv(args.outdir / 'raw_results.csv', index=False)
    summary.to_csv(args.outdir / 'summary.csv', index=False)
    print(args.outdir / 'summary.csv')


if __name__ == '__main__':
    main()

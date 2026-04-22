from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.bmt_repro.engineering import get_engineering_problems
from src.bmt_repro.ga import make_paper_config, run_ga


def main() -> None:
    parser = argparse.ArgumentParser(description='Run GA-BMT on the three engineering problems from the paper.')
    parser.add_argument('--outdir', type=Path, default=Path('results/engineering'))
    parser.add_argument('--population-size', type=int, default=100)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--runs', type=int, default=25)
    args = parser.parse_args()

    cfg = make_paper_config()
    cfg.generations = args.generations
    cfg.runs = args.runs
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in get_engineering_problems():
        for run in range(args.runs):
            seed = cfg.seed0 + 1000 * run + len(problem.name)
            result = run_ga(
                problem,
                'BMT',
                args.population_size,
                args.generations,
                seed,
                tournament_size=cfg.tournament_size,
                crossover_probability=cfg.crossover_probability,
                mutation_probability=cfg.mutation_probability,
                arithmetic_lambda=cfg.arithmetic_lambda,
                bipolarity=cfg.bipolarity,
                fgts_ftour=cfg.fgts_ftour,
                rts_window=cfg.rts_window,
                association_size=cfg.association_size,
            )
            row = {
                'problem': problem.name,
                'run': run,
                'best_value': result['best_value'],
            }
            for i, value in enumerate(result['best_solution'], start=1):
                row[f'x{i}'] = float(value)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / 'ga_bmt_engineering_runs.csv', index=False)
    summary = df.groupby('problem')['best_value'].agg(['min', 'median', 'mean', 'std']).reset_index()
    summary.to_csv(args.outdir / 'ga_bmt_engineering_summary.csv', index=False)
    print(args.outdir / 'ga_bmt_engineering_summary.csv')


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.bmt_repro.runtime import clamp_jobs, default_jobs, limit_library_threads
limit_library_threads(1)

from src.bmt_repro.benchmarks import get_standard_benchmarks
from src.bmt_repro.experiment_workers import BipolarityTask, run_bipolarity_task, worker_initializer
from src.bmt_repro.ga import make_paper_config
from src.bmt_repro.parallel import process_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep BMT bipolarity from 0.05 to 1.0 in steps of 0.05.")
    parser.add_argument("--outdir", type=Path, default=Path("results/bipolarity_sweep"))
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--seed0", type=int, default=12345)
    parser.add_argument("--jobs", type=int, default=default_jobs(), help="Number of worker processes (recommended: 8, max suggested: 12).")
    args = parser.parse_args()

    cfg = make_paper_config()
    cfg.generations = args.generations
    cfg.runs = args.runs
    cfg.population_sizes = (args.population_size,)
    cfg.algorithms = ("BMT",)
    cfg.seed0 = args.seed0
    jobs = clamp_jobs(args.jobs, default=default_jobs(), max_jobs=12)

    tasks: list[BipolarityTask] = []
    for step in range(1, 21):
        q = round(0.05 * step, 2)
        for problem in get_standard_benchmarks():
            for run in range(cfg.runs):
                tasks.append(
                    BipolarityTask(
                        problem_name=problem.name,
                        bipolarity=q,
                        population_size=args.population_size,
                        run=run,
                        generations=args.generations,
                        seed0=cfg.seed0,
                        fgts_ftour=cfg.fgts_ftour,
                        rts_window=cfg.rts_window,
                        association_size=cfg.association_size,
                    )
                )

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = process_map(
        run_bipolarity_task,
        tasks,
        jobs=jobs,
        initializer=worker_initializer if jobs > 1 else None,
        progress_label="bipolarity sweep tasks completed",
    )

    raw_all = pd.DataFrame(rows)
    summary = (
        raw_all.groupby(["bipolarity", "problem"])["best_value"]
        .agg(median="median", std="std", mean="mean")
        .reset_index()
    )
    raw_all.to_csv(args.outdir / "raw_results.csv", index=False)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    print(args.outdir / "summary.csv")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.bmt_repro.runtime import clamp_jobs, default_jobs, limit_library_threads
limit_library_threads(1)

from src.bmt_repro.engineering import get_engineering_problems
from src.bmt_repro.experiment_workers import EngineeringTask, run_engineering_task, worker_initializer
from src.bmt_repro.ga import make_paper_config
from src.bmt_repro.parallel import process_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA-BMT on the three engineering problems from the paper.")
    parser.add_argument("--outdir", type=Path, default=Path("results/engineering"))
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--seed0", type=int, default=12345)
    parser.add_argument("--jobs", type=int, default=default_jobs(), help="Number of worker processes (recommended: 8, max suggested: 12).")
    args = parser.parse_args()

    cfg = make_paper_config()
    cfg.generations = args.generations
    cfg.runs = args.runs
    cfg.seed0 = args.seed0
    jobs = clamp_jobs(args.jobs, default=default_jobs(), max_jobs=12)

    tasks = [
        EngineeringTask(
            problem_name=problem.name,
            population_size=args.population_size,
            run=run,
            generations=args.generations,
            seed0=cfg.seed0,
        )
        for problem in get_engineering_problems()
        for run in range(args.runs)
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = process_map(
        run_engineering_task,
        tasks,
        jobs=jobs,
        initializer=worker_initializer if jobs > 1 else None,
        progress_label="engineering tasks completed",
    )
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "ga_bmt_engineering_runs.csv", index=False)
    summary = df.groupby("problem")["best_value"].agg(["min", "median", "mean", "std"]).reset_index()
    summary.to_csv(args.outdir / "ga_bmt_engineering_summary.csv", index=False)
    print(args.outdir / "ga_bmt_engineering_summary.csv")


if __name__ == "__main__":
    main()

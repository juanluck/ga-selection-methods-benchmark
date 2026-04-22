from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.bmt_repro.runtime import clamp_jobs, default_jobs, limit_library_threads
limit_library_threads(1)

from src.bmt_repro.benchmarks import get_standard_benchmarks
from src.bmt_repro.experiment_workers import BasicTask, run_basic_task, worker_initializer
from src.bmt_repro.ga import make_paper_config
from src.bmt_repro.parallel import process_map
from src.bmt_repro.stats import friedman_by_population, summarize_median_std, wilcoxon_by_population
from src.bmt_repro.utils import ALGORITHM_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 21 classical benchmark functions with the paper-style GA."
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/basic_benchmarks"))
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--population-sizes", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(ALGORITHM_NAMES))
    parser.add_argument("--fgts-ftour", type=float, default=4.5)
    parser.add_argument("--rts-window", type=int, default=4)
    parser.add_argument("--association-size", type=int, default=4)
    parser.add_argument("--seed0", type=int, default=12345)
    parser.add_argument("--jobs", type=int, default=default_jobs(), help="Number of worker processes (recommended: 8, max suggested: 12).")
    args = parser.parse_args()

    cfg = make_paper_config()
    cfg.generations = args.generations
    cfg.runs = args.runs
    cfg.population_sizes = tuple(args.population_sizes)
    cfg.algorithms = tuple(args.algorithms)
    cfg.fgts_ftour = args.fgts_ftour
    cfg.rts_window = args.rts_window
    cfg.association_size = args.association_size
    cfg.seed0 = args.seed0

    jobs = clamp_jobs(args.jobs, default=default_jobs(), max_jobs=12)

    tasks: list[BasicTask] = []
    for pop_size in cfg.population_sizes:
        for problem in get_standard_benchmarks():
            for run in range(cfg.runs):
                tasks.append(
                    BasicTask(
                        problem_name=problem.name,
                        population_size=pop_size,
                        run=run,
                        generations=cfg.generations,
                        algorithms=cfg.algorithms,
                        seed0=cfg.seed0,
                        fgts_ftour=cfg.fgts_ftour,
                        rts_window=cfg.rts_window,
                        association_size=cfg.association_size,
                    )
                )

    args.outdir.mkdir(parents=True, exist_ok=True)
    batches = process_map(
        run_basic_task,
        tasks,
        jobs=jobs,
        initializer=worker_initializer if jobs > 1 else None,
        progress_label="basic benchmark tasks completed",
    )
    rows = [row for batch in batches for row in batch]
    raw = pd.DataFrame(rows)
    summary = summarize_median_std(raw)
    friedman = friedman_by_population(summary)
    wilcoxon = wilcoxon_by_population(summary)

    raw.to_csv(args.outdir / "raw_results.csv", index=False)
    summary.to_csv(args.outdir / "summary_median_std.csv", index=False)
    friedman.to_csv(args.outdir / "friedman_by_population.csv", index=False)
    wilcoxon.to_csv(args.outdir / "wilcoxon_by_population.csv", index=False)

    print("Wrote:")
    for name in [
        "raw_results.csv",
        "summary_median_std.csv",
        "friedman_by_population.csv",
        "wilcoxon_by_population.csv",
    ]:
        print(args.outdir / name)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.bmt_repro.runtime import clamp_jobs, default_jobs, limit_library_threads
limit_library_threads(1)

from src.bmt_repro.cec2017_suite import get_cec2017_bound_problems
from src.bmt_repro.experiment_workers import CECTask, run_cec_task, worker_initializer
from src.bmt_repro.ga import make_paper_config
from src.bmt_repro.parallel import process_map
from src.bmt_repro.stats import summarize_median_std, friedman_by_population, wilcoxon_by_population
from src.bmt_repro.utils import ALGORITHM_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CEC 2017 bound-constrained suite with the same GA backbone used by the paper. "
            "This is the recommended Python path for the benchmark block often referred to as the 2018 "
            "bound-constrained competition suite."
        )
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/cec2017_bound"))
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--population-sizes", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--evals-per-dim", type=int, default=10000)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_NAMES, default=list(ALGORITHM_NAMES))
    parser.add_argument("--fgts-ftour", type=float, default=4.5)
    parser.add_argument("--rts-window", type=int, default=4)
    parser.add_argument("--association-size", type=int, default=4)
    parser.add_argument("--include-f2", action="store_true")
    parser.add_argument("--seed0", type=int, default=12345)
    parser.add_argument("--jobs", type=int, default=default_jobs(), help="Number of worker processes (recommended: 8, max suggested: 12).")
    args = parser.parse_args()

    cfg = make_paper_config()
    cfg.runs = args.runs
    cfg.population_sizes = tuple(args.population_sizes)
    cfg.algorithms = tuple(args.algorithms)
    cfg.fgts_ftour = args.fgts_ftour
    cfg.rts_window = args.rts_window
    cfg.association_size = args.association_size
    cfg.seed0 = args.seed0

    problems = get_cec2017_bound_problems(
        dimension=args.dimension,
        exclude_deleted_f2=not args.include_f2,
    )
    jobs = clamp_jobs(args.jobs, default=default_jobs(), max_jobs=12)

    tasks: list[CECTask] = []
    for pop_size in cfg.population_sizes:
        if args.generations is None:
            budget = args.evals_per_dim * args.dimension
            generations = max(1, budget // pop_size - 1)
        else:
            generations = args.generations
            budget = pop_size * (generations + 1)

        for problem in problems:
            for run in range(cfg.runs):
                tasks.append(
                    CECTask(
                        problem_name=problem.name,
                        dimension=args.dimension,
                        exclude_deleted_f2=not args.include_f2,
                        population_size=pop_size,
                        run=run,
                        generations=generations,
                        evaluation_budget=budget,
                        algorithms=cfg.algorithms,
                        seed0=cfg.seed0,
                        fgts_ftour=cfg.fgts_ftour,
                        rts_window=cfg.rts_window,
                        association_size=cfg.association_size,
                    )
                )

    args.outdir.mkdir(parents=True, exist_ok=True)
    batches = process_map(
        run_cec_task,
        tasks,
        jobs=jobs,
        initializer=worker_initializer if jobs > 1 else None,
        progress_label="CEC tasks completed",
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

    metadata = pd.DataFrame(
        [
            {
                "suite": "CEC2017 bound-constrained (also reused by the 2018 bound-constrained competition)",
                "dimension": args.dimension,
                "runs": args.runs,
                "population_sizes": ",".join(str(value) for value in args.population_sizes),
                "algorithms": ",".join(args.algorithms),
                "generations_mode": "fixed" if args.generations is not None else "derived_from_evals_per_dim",
                "generations": args.generations if args.generations is not None else "",
                "evals_per_dim": args.evals_per_dim if args.generations is None else "",
                "tournament_size": cfg.tournament_size,
                "pc": cfg.crossover_probability,
                "pm": cfg.mutation_probability,
                "lambda": cfg.arithmetic_lambda,
                "bipolarity": cfg.bipolarity,
                "fgts_ftour": cfg.fgts_ftour,
                "rts_window": cfg.rts_window,
                "association_size": cfg.association_size,
                "seed0": cfg.seed0,
                "jobs": jobs,
            }
        ]
    )
    metadata.to_csv(args.outdir / "run_metadata.csv", index=False)

    print("Wrote:")
    for name in [
        "raw_results.csv",
        "summary_median_std.csv",
        "friedman_by_population.csv",
        "wilcoxon_by_population.csv",
        "run_metadata.csv",
    ]:
        print(args.outdir / name)


if __name__ == "__main__":
    main()

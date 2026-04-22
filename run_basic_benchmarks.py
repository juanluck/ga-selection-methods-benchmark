from __future__ import annotations

import argparse
from pathlib import Path

from src.bmt_repro.benchmarks import get_standard_benchmarks
from src.bmt_repro.ga import make_paper_config, run_suite
from src.bmt_repro.stats import summarize_median_std, friedman_by_population, wilcoxon_by_population
from src.bmt_repro.utils import ALGORITHM_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta los 21 benchmarks clásicos con la configuración tipo-paper."
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

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = run_suite(get_standard_benchmarks(), cfg)
    summary = summarize_median_std(raw)
    friedman = friedman_by_population(summary)
    wilcoxon = wilcoxon_by_population(summary)

    raw.to_csv(args.outdir / "raw_results.csv", index=False)
    summary.to_csv(args.outdir / "summary_median_std.csv", index=False)
    friedman.to_csv(args.outdir / "friedman_by_population.csv", index=False)
    wilcoxon.to_csv(args.outdir / "wilcoxon_by_population.csv", index=False)

    print("Wrote:")
    for name in ["raw_results.csv", "summary_median_std.csv", "friedman_by_population.csv", "wilcoxon_by_population.csv"]:
        print(args.outdir / name)


if __name__ == "__main__":
    main()

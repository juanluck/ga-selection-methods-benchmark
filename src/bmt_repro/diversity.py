from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .benchmarks import get_standard_benchmarks
from .ga import PaperConfig, run_ga


def generate_diversity_and_footprints(outdir: Path, pop_sizes=(50, 100, 200), generations: int = 100) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    benchmarks = {p.name: p for p in get_standard_benchmarks()}
    de_jong = benchmarks['F6_DeJong']
    axis = benchmarks['F2_AxisParallelHyperEllipsoid']
    cfg = PaperConfig(generations=generations, runs=1)

    for pop_size in pop_sizes:
        bmt = run_ga(de_jong, 'BMT', pop_size, generations, seed=123 + pop_size,
                     tournament_size=cfg.tournament_size, crossover_probability=cfg.crossover_probability,
                     mutation_probability=cfg.mutation_probability, arithmetic_lambda=cfg.arithmetic_lambda,
                     bipolarity=cfg.bipolarity, fgts_ftour=cfg.fgts_ftour, rts_window=cfg.rts_window,
                     association_size=cfg.association_size, collect_diagnostics=True)
        st = run_ga(de_jong, 'ST', pop_size, generations, seed=123 + pop_size,
                    tournament_size=cfg.tournament_size, crossover_probability=cfg.crossover_probability,
                    mutation_probability=cfg.mutation_probability, arithmetic_lambda=cfg.arithmetic_lambda,
                    bipolarity=cfg.bipolarity, fgts_ftour=cfg.fgts_ftour, rts_window=cfg.rts_window,
                    association_size=cfg.association_size, collect_diagnostics=True)

        fig = plt.figure()
        plt.semilogy(range(1, generations + 1), bmt['phenotype_diversity'], label='BMT')
        plt.semilogy(range(1, generations + 1), st['phenotype_diversity'], label='ST')
        plt.xlabel('Iteration')
        plt.ylabel('Phenotype diversity')
        plt.legend()
        plt.tight_layout()
        fig.savefig(outdir / f'phenotype_pop{pop_size}.png', dpi=150)
        plt.close(fig)

        fig = plt.figure()
        plt.semilogy(range(1, generations + 1), bmt['genotype_diversity'], label='BMT')
        plt.semilogy(range(1, generations + 1), st['genotype_diversity'], label='ST')
        plt.xlabel('Iteration')
        plt.ylabel('Genotype diversity (normalized mean pairwise distance)')
        plt.legend()
        plt.tight_layout()
        fig.savefig(outdir / f'genotype_pop{pop_size}.png', dpi=150)
        plt.close(fig)

        fig = plt.figure()
        plt.semilogy(range(1, generations + 1), bmt['best_history'], label='BMT')
        plt.semilogy(range(1, generations + 1), st['best_history'], label='ST')
        plt.xlabel('Iteration')
        plt.ylabel('Best value')
        plt.legend()
        plt.tight_layout()
        fig.savefig(outdir / f'dejong_best_pop{pop_size}.png', dpi=150)
        plt.close(fig)

        bmt_axis = run_ga(axis, 'BMT', pop_size, generations, seed=321 + pop_size,
                          tournament_size=cfg.tournament_size, crossover_probability=cfg.crossover_probability,
                          mutation_probability=cfg.mutation_probability, arithmetic_lambda=cfg.arithmetic_lambda,
                          bipolarity=cfg.bipolarity, fgts_ftour=cfg.fgts_ftour, rts_window=cfg.rts_window,
                          association_size=cfg.association_size, collect_diagnostics=True)
        st_axis = run_ga(axis, 'ST', pop_size, generations, seed=321 + pop_size,
                         tournament_size=cfg.tournament_size, crossover_probability=cfg.crossover_probability,
                         mutation_probability=cfg.mutation_probability, arithmetic_lambda=cfg.arithmetic_lambda,
                         bipolarity=cfg.bipolarity, fgts_ftour=cfg.fgts_ftour, rts_window=cfg.rts_window,
                         association_size=cfg.association_size, collect_diagnostics=True)

        for label, data in [('BMT', bmt_axis), ('ST', st_axis)]:
            points = np.vstack(data['footprints'])
            xy = points[:, :2]
            fig = plt.figure()
            plt.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6)
            plt.xlabel('Range of the first parameter')
            plt.ylabel('Range of the second parameter')
            plt.title(f'{label} discovered points (pop={pop_size})')
            plt.tight_layout()
            fig.savefig(outdir / f'footprints_{label.lower()}_pop{pop_size}.png', dpi=150)
            plt.close(fig)

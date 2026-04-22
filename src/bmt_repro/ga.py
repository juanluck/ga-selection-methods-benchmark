from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any
import numpy as np
import pandas as pd

from .selection import make_selector
from .engineering import ConstrainedProblem
from .utils import stable_problem_offset


@dataclass
class PaperConfig:
    population_sizes: tuple[int, ...] = (50, 100, 200)
    tournament_size: int = 4
    crossover_probability: float = 0.7
    mutation_probability: float = 0.05
    arithmetic_lambda: float = 0.6
    bipolarity: float = 0.25
    fgts_ftour: float = 4.5
    rts_window: int = 4
    association_size: int = 4
    generations: int = 100
    runs: int = 25
    seed0: int = 12345
    algorithms: tuple[str, ...] = ("BMT", "CS", "FGTS", "RTS", "ST", "UTS")


def make_paper_config() -> PaperConfig:
    return PaperConfig()


def arithmetic_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
    lam: float,
) -> np.ndarray:
    rand = rng.random(size=parent_a.shape)
    return (parent_a + parent_b) / 2.0 + lam * np.abs(parent_b - parent_a) * (2.0 * rand - 1.0)


def random_mutation(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    mutation_probability: float,
) -> np.ndarray:
    y = x.copy()
    mask = rng.random(size=y.shape) < mutation_probability
    if np.any(mask):
        y[mask] = rng.uniform(lower[mask], upper[mask])
    return y


def evaluate_population(problem, population: np.ndarray) -> np.ndarray:
    if isinstance(problem, ConstrainedProblem):
        return np.asarray([problem.penalized(individual) for individual in population], dtype=float)
    return np.asarray([problem.evaluate(individual) for individual in population], dtype=float)


def run_ga(
    problem,
    algorithm: str,
    pop_size: int,
    generations: int,
    seed: int,
    *,
    tournament_size: int,
    crossover_probability: float,
    mutation_probability: float,
    arithmetic_lambda: float,
    bipolarity: float,
    fgts_ftour: float,
    rts_window: int,
    association_size: int,
    collect_diagnostics: bool = False,
):
    rng = np.random.default_rng(seed)
    population = rng.uniform(problem.lower, problem.upper, size=(pop_size, problem.dimension))
    population = np.asarray([problem.repair(individual) for individual in population], dtype=float)
    selector = make_selector(
        algorithm,
        rng,
        tournament_size=tournament_size,
        bipolarity=bipolarity,
        fgts_ftour=fgts_ftour,
        rts_window=rts_window,
        association_size=association_size,
        minimize=problem.minimize,
    )

    best_history: list[float] = []
    phenotype_diversity: list[float] = []
    genotype_diversity: list[float] = []
    footprints: list[np.ndarray] = []
    baseline_pairwise: float | None = None

    for _ in range(generations):
        fitness = evaluate_population(problem, population)
        selector.start_generation(population, fitness)

        if collect_diagnostics:
            best_history.append(float(np.min(fitness) if problem.minimize else np.max(fitness)))
            _, counts = np.unique(np.round(fitness, 12), return_counts=True)
            phenotype_diversity.append(float(np.sum(counts == 1) / pop_size))

            if problem.dimension == 2:
                diff = population[:, None, :] - population[None, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                tri = dist[np.triu_indices(pop_size, k=1)]
                mean_pairwise = float(np.mean(tri)) if tri.size else 0.0
                if baseline_pairwise is None:
                    baseline_pairwise = max(mean_pairwise, 1e-12)
                genotype_diversity.append(mean_pairwise / baseline_pairwise)
                footprints.append(population.copy())
            else:
                genotype_diversity.append(np.nan)

        children = []
        while len(children) < pop_size:
            if algorithm.upper() == "BMT":
                parent_a_idx, parent_b_idx = selector.select_pair()
            else:
                parent_a_idx, parent_b_idx = selector.select_one(), selector.select_one()

            parent_a, parent_b = population[parent_a_idx], population[parent_b_idx]
            if rng.random() < crossover_probability:
                child = arithmetic_crossover(parent_a, parent_b, rng, arithmetic_lambda)
            else:
                child = parent_a.copy()
            child = random_mutation(child, problem.lower, problem.upper, rng, mutation_probability)
            child = problem.repair(child)
            children.append(child)

        population = np.asarray(children, dtype=float)

    final_fitness = evaluate_population(problem, population)
    best_idx = int(np.argmin(final_fitness) if problem.minimize else np.argmax(final_fitness))
    best_value = float(final_fitness[best_idx])
    best_solution = population[best_idx].copy()

    result: dict[str, Any] = {
        "best_value": best_value,
        "best_solution": best_solution,
    }
    if collect_diagnostics:
        result.update(
            {
                "best_history": np.asarray(best_history, dtype=float),
                "phenotype_diversity": np.asarray(phenotype_diversity, dtype=float),
                "genotype_diversity": np.asarray(genotype_diversity, dtype=float),
                "footprints": footprints,
            }
        )
    return result


def make_run_seed(config: PaperConfig, problem_name: str, pop_size: int, run: int) -> int:
    return config.seed0 + 1000 * pop_size + 100 * stable_problem_offset(problem_name, 1000) + run


def run_suite(problems: Sequence, config: PaperConfig) -> pd.DataFrame:
    rows = []
    for pop_size in config.population_sizes:
        for problem in problems:
            for run in range(config.runs):
                seed = make_run_seed(config, problem.name, pop_size, run)
                for algorithm in config.algorithms:
                    result = run_ga(
                        problem,
                        algorithm,
                        pop_size,
                        config.generations,
                        seed,
                        tournament_size=config.tournament_size,
                        crossover_probability=config.crossover_probability,
                        mutation_probability=config.mutation_probability,
                        arithmetic_lambda=config.arithmetic_lambda,
                        bipolarity=config.bipolarity,
                        fgts_ftour=config.fgts_ftour,
                        rts_window=config.rts_window,
                        association_size=config.association_size,
                        collect_diagnostics=False,
                    )
                    rows.append(
                        {
                            "problem": problem.name,
                            "population_size": pop_size,
                            "run": run,
                            "algorithm": algorithm,
                            "best_value": result["best_value"],
                        }
                    )
    return pd.DataFrame(rows)

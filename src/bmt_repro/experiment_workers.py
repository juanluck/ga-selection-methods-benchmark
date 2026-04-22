from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .benchmarks import get_standard_benchmarks
from .cec2017_suite import get_cec2017_bound_problem_map
from .engineering import get_engineering_problems
from .ga import make_paper_config, make_run_seed, run_ga
from .runtime import limit_library_threads


@dataclass(frozen=True)
class BasicTask:
    problem_name: str
    population_size: int
    run: int
    generations: int
    algorithms: tuple[str, ...]
    seed0: int
    fgts_ftour: float
    rts_window: int
    association_size: int


@dataclass(frozen=True)
class CECTask:
    problem_name: str
    dimension: int
    exclude_deleted_f2: bool
    population_size: int
    run: int
    generations: int
    evaluation_budget: int
    algorithms: tuple[str, ...]
    seed0: int
    fgts_ftour: float
    rts_window: int
    association_size: int


@dataclass(frozen=True)
class EngineeringTask:
    problem_name: str
    population_size: int
    run: int
    generations: int
    seed0: int


@dataclass(frozen=True)
class BipolarityTask:
    problem_name: str
    bipolarity: float
    population_size: int
    run: int
    generations: int
    seed0: int
    fgts_ftour: float
    rts_window: int
    association_size: int


def worker_initializer() -> None:
    limit_library_threads(1, force=True)


@lru_cache(maxsize=1)
def _standard_problem_map():
    return {problem.name: problem for problem in get_standard_benchmarks()}


@lru_cache(maxsize=1)
def _engineering_problem_map():
    return {problem.name: problem for problem in get_engineering_problems()}


@lru_cache(maxsize=None)
def _cec_problem_map(dimension: int, exclude_deleted_f2: bool):
    return get_cec2017_bound_problem_map(dimension=dimension, exclude_deleted_f2=exclude_deleted_f2)


def run_basic_task(task: BasicTask) -> list[dict]:
    config = make_paper_config()
    config.seed0 = task.seed0
    config.fgts_ftour = task.fgts_ftour
    config.rts_window = task.rts_window
    config.association_size = task.association_size

    problem = _standard_problem_map()[task.problem_name]
    seed = make_run_seed(config, task.problem_name, task.population_size, task.run)

    rows = []
    for algorithm in task.algorithms:
        result = run_ga(
            problem,
            algorithm,
            task.population_size,
            task.generations,
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
                "population_size": task.population_size,
                "run": task.run,
                "algorithm": algorithm,
                "best_value": result["best_value"],
            }
        )
    return rows


def run_cec_task(task: CECTask) -> list[dict]:
    config = make_paper_config()
    config.seed0 = task.seed0
    config.fgts_ftour = task.fgts_ftour
    config.rts_window = task.rts_window
    config.association_size = task.association_size

    problem = _cec_problem_map(task.dimension, task.exclude_deleted_f2)[task.problem_name]
    seed = make_run_seed(config, task.problem_name, task.population_size, task.run)

    rows = []
    for algorithm in task.algorithms:
        result = run_ga(
            problem,
            algorithm,
            task.population_size,
            task.generations,
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
                "suite": "CEC2017_bound_constrained",
                "dimension": task.dimension,
                "population_size": task.population_size,
                "generations": task.generations,
                "evaluation_budget": task.evaluation_budget,
                "problem": problem.name,
                "run": task.run,
                "algorithm": algorithm,
                "best_value": result["best_value"],
                "error_to_optimum": result["best_value"] - problem.optimum,
            }
        )
    return rows


def run_engineering_task(task: EngineeringTask) -> dict:
    config = make_paper_config()
    config.seed0 = task.seed0
    problem = _engineering_problem_map()[task.problem_name]
    seed = config.seed0 + 1000 * task.run + len(problem.name)

    result = run_ga(
        problem,
        "BMT",
        task.population_size,
        task.generations,
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
    row = {
        "problem": problem.name,
        "run": task.run,
        "best_value": result["best_value"],
    }
    for index, value in enumerate(result["best_solution"], start=1):
        row[f"x{index}"] = float(value)
    return row


def run_bipolarity_task(task: BipolarityTask) -> dict:
    config = make_paper_config()
    config.seed0 = task.seed0
    config.bipolarity = task.bipolarity
    config.fgts_ftour = task.fgts_ftour
    config.rts_window = task.rts_window
    config.association_size = task.association_size

    problem = _standard_problem_map()[task.problem_name]
    seed = make_run_seed(config, task.problem_name, task.population_size, task.run)

    result = run_ga(
        problem,
        "BMT",
        task.population_size,
        task.generations,
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
    return {
        "problem": problem.name,
        "population_size": task.population_size,
        "run": task.run,
        "algorithm": "BMT",
        "best_value": result["best_value"],
        "bipolarity": task.bipolarity,
    }

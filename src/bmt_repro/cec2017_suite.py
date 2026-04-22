from __future__ import annotations

from typing import List
import numpy as np

from .benchmarks import Problem


def _import_cec2017():
    try:
        from cec2017.functions import all_functions  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "cec2017-py is not installed. Install it first, for example:\n"
            "  pip install git+https://github.com/tilleyd/cec2017-py.git"
        ) from exc
    return all_functions


def get_cec2017_bound_problems(dimension: int = 10, exclude_deleted_f2: bool = True) -> List[Problem]:
    all_functions = _import_cec2017()
    lower = np.full(dimension, -100.0, dtype=float)
    upper = np.full(dimension, 100.0, dtype=float)

    problems: List[Problem] = []
    for func_id, fn in enumerate(all_functions, start=1):
        if exclude_deleted_f2 and func_id == 2:
            continue

        def evaluate_one(x: np.ndarray, fn=fn) -> float:
            arr = np.asarray(x, dtype=float).reshape(1, -1)
            return float(np.asarray(fn(arr))[0])

        problems.append(
            Problem(
                name=f"CEC2017_F{func_id}",
                dimension=dimension,
                lower=lower,
                upper=upper,
                evaluate=evaluate_one,
                optimum=float(100 * func_id),
                minimize=True,
            )
        )
    return problems

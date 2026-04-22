from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .benchmarks import Problem


def _clip(x, lower, upper):
    return np.clip(x, lower, upper)


@dataclass(frozen=True)
class ConstrainedProblem(Problem):
    penalty_scale: float = 1e7

    def constraint_violations(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def penalized(self, x: np.ndarray) -> float:
        return float(self.evaluate(x) + self.penalty_scale * np.sum(np.maximum(self.constraint_violations(x), 0.0)))


class PressureVesselProblem(ConstrainedProblem):
    def __init__(self):
        super().__init__(
            name='PressureVessel',
            dimension=4,
            lower=np.array([0.0625, 0.0625, 40.0, 20.0], dtype=float),
            upper=np.array([6.25, 6.25, 80.0, 60.0], dtype=float),
            evaluate=self._objective,
            optimum=None,
            minimize=True,
            penalty_scale=1e7,
        )

    def repair(self, x: np.ndarray) -> np.ndarray:
        y = _clip(x, self.lower, self.upper)
        y[0] = np.round(y[0] / 0.0625) * 0.0625
        y[1] = np.round(y[1] / 0.0625) * 0.0625
        return _clip(y, self.lower, self.upper)

    @staticmethod
    def _objective(x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x3 ** 2 * x2 + 3.1611 * x1 ** 2 * x4 + 19.8621 * x1 ** 2 * x3

    def constraint_violations(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3, x4 = x
        g1 = 0.0193 * x3 - x1
        g2 = 0.00954 * x3 - x2
        g3 = 750.0 * 1728.0 - np.pi * x3 ** 2 * x4 - (4.0 / 3.0) * np.pi * x3 ** 3
        g4 = x4 - 240.0
        g5 = 1.1 - x1
        g6 = 0.6 - x2
        return np.array([g1, g2, g3, g4, g5, g6], dtype=float)


class GearTrainProblem(ConstrainedProblem):
    def __init__(self):
        super().__init__(
            name='GearTrain',
            dimension=4,
            lower=np.array([12.0, 12.0, 12.0, 12.0], dtype=float),
            upper=np.array([60.0, 60.0, 60.0, 60.0], dtype=float),
            evaluate=self._objective,
            optimum=0.0,
            minimize=True,
            penalty_scale=0.0,
        )

    def repair(self, x: np.ndarray) -> np.ndarray:
        return np.round(_clip(x, self.lower, self.upper))

    @staticmethod
    def _objective(x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        return (1.0 / 6.931 - (x3 * x2) / (x1 * x4)) ** 2

    def constraint_violations(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(1, dtype=float)


class TensionCompressionSpringProblem(ConstrainedProblem):
    def __init__(self):
        super().__init__(
            name='TensionCompressionSpring',
            dimension=3,
            lower=np.array([0.05, 0.25, 2.0], dtype=float),
            upper=np.array([2.0, 1.30, 15.0], dtype=float),
            evaluate=self._objective,
            optimum=None,
            minimize=True,
            penalty_scale=1e7,
        )

    @staticmethod
    def _objective(x: np.ndarray) -> float:
        x1, x2, x3 = x
        return (x3 + 2.0) * x2 * x1 ** 2

    def constraint_violations(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        g1 = 1.0 - (x2 ** 3 * x3) / (71785.0 * x1 ** 4)
        g2 = (4.0 * x2 ** 2 - x1 * x2) / (12566.0 * (x2 * x1 ** 3 - x1 ** 4)) + 1.0 / (5108.0 * x1 ** 2) - 1.0
        g3 = 1.0 - 140.45 * x1 / (x2 ** 2 * x3)
        g4 = (x1 + x2) / 1.5 - 1.0
        return np.array([g1, g2, g3, g4], dtype=float)


def get_engineering_problems():
    return [PressureVesselProblem(), GearTrainProblem(), TensionCompressionSpringProblem()]

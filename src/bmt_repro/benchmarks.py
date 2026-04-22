from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import math
import numpy as np


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    lower: np.ndarray
    upper: np.ndarray
    evaluate: Callable[[np.ndarray], float]
    optimum: Optional[float] = None
    minimize: bool = True

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def repair(self, x: np.ndarray) -> np.ndarray:
        return self.clip(x)


def _arr(values):
    return np.asarray(values, dtype=float)


def ackley(x: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    d = x.size
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + math.e


def axis_parallel_hyper_ellipsoid(x: np.ndarray) -> float:
    idx = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(idx * x ** 2))


def branin(x: np.ndarray) -> float:
    x1, x2 = x
    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s


def booth(x: np.ndarray) -> float:
    x1, x2 = x
    return (x1 + 2.0 * x2 - 7.0) ** 2 + (2.0 * x1 + x2 - 5.0) ** 2


def bukin6(x: np.ndarray) -> float:
    x1, x2 = x
    return 100.0 * np.sqrt(abs(x2 - 0.01 * x1 ** 2)) + 0.01 * abs(x1 + 10.0)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


_A = np.array([
    [-32, -16, 0, 16, 32] * 5,
    [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5,
], dtype=float)

def de_jong5(x: np.ndarray) -> float:
    x1, x2 = x
    denom = np.arange(1, 26, dtype=float) + (x1 - _A[0]) ** 6 + (x2 - _A[1]) ** 6
    return 1.0 / (0.002 + np.sum(1.0 / denom))


def drop_wave(x: np.ndarray) -> float:
    r2 = float(np.sum(x ** 2))
    return -(1.0 + math.cos(12.0 * math.sqrt(r2))) / (0.5 * r2 + 2.0)


def easom(x: np.ndarray) -> float:
    x1, x2 = x
    return -math.cos(x1) * math.cos(x2) * math.exp(-((x1 - math.pi) ** 2 + (x2 - math.pi) ** 2))


def goldstein_price(x: np.ndarray) -> float:
    x1, x2 = x
    a = 1.0 + (x1 + x2 + 1.0) ** 2 * (19.0 - 14.0 * x1 + 3.0 * x1 ** 2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2 ** 2)
    b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (18.0 - 32.0 * x1 + 12.0 * x1 ** 2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2 ** 2)
    return a * b


def griewank(x: np.ndarray) -> float:
    idx = np.sqrt(np.arange(1, x.size + 1, dtype=float))
    return float(np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / idx)) + 1.0)


def holder_table(x: np.ndarray) -> float:
    x1, x2 = x
    return -abs(math.sin(x1) * math.cos(x2) * math.exp(abs(1.0 - math.sqrt(x1 ** 2 + x2 ** 2) / math.pi)))


_LANG_A = np.array([
    [3, 5, 2, 1, 7],
    [5, 2, 1, 4, 9],
], dtype=float)
_LANG_C = np.array([1, 2, 5, 2, 3], dtype=float)

def langermann(x: np.ndarray) -> float:
    total = 0.0
    for i in range(5):
        diff = np.sum((x - _LANG_A[:, i]) ** 2)
        total += _LANG_C[i] * math.exp(-diff / math.pi) * math.cos(math.pi * diff)
    return -total


def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
    idx = np.arange(1, x.size + 1, dtype=float)
    return -float(np.sum(np.sin(x) * np.sin(idx * x ** 2 / math.pi) ** (2.0 * m)))


def rastrigin(x: np.ndarray) -> float:
    return 10.0 * x.size + float(np.sum(x ** 2 - 10.0 * np.cos(2.0 * math.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def rotated_hyper_ellipsoid(x: np.ndarray) -> float:
    total = 0.0
    for i in range(x.size):
        total += np.sum(x[: i + 1]) ** 2
    return float(total)


def schubert(x: np.ndarray) -> float:
    def s(v: float) -> float:
        return sum(j * math.cos((j + 1) * v + j) for j in range(1, 6))
    return s(x[0]) * s(x[1])


def schwefel(x: np.ndarray) -> float:
    return float(418.9829 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def six_hump_camel(x: np.ndarray) -> float:
    x1, x2 = x
    return (4.0 - 2.1 * x1 ** 2 + x1 ** 4 / 3.0) * x1 ** 2 + x1 * x2 + (-4.0 + 4.0 * x2 ** 2) * x2 ** 2


def sum_different_powers(x: np.ndarray) -> float:
    idx = np.arange(2, 2 + x.size, dtype=float)
    return float(np.sum(np.abs(x) ** idx))


def get_standard_benchmarks() -> list[Problem]:
    return [
        Problem('F1_Ackley', 2, _arr([-32.768, -32.768]), _arr([32.768, 32.768]), ackley, 0.0),
        Problem('F2_AxisParallelHyperEllipsoid', 2, _arr([-5.12, -5.12]), _arr([5.12, 5.12]), axis_parallel_hyper_ellipsoid, 0.0),
        Problem('F3_Branin', 2, _arr([-5.0, 0.0]), _arr([10.0, 15.0]), branin, 0.397887),
        Problem('F4_Booth', 2, _arr([-10.0, -10.0]), _arr([10.0, 10.0]), booth, 0.0),
        Problem('F5_Bukin6', 2, _arr([-15.0, -3.0]), _arr([-5.0, 3.0]), bukin6, 0.0),
        Problem('F6_DeJong', 2, _arr([-5.12, -5.12]), _arr([5.12, 5.12]), sphere, 0.0),
        Problem('F7_DeJong5', 2, _arr([-65.536, -65.536]), _arr([65.536, 65.536]), de_jong5, 0.9980038388186492),
        Problem('F8_DropWave', 2, _arr([-5.12, -5.12]), _arr([5.12, 5.12]), drop_wave, -1.0),
        Problem('F9_Easom', 2, _arr([-100.0, -100.0]), _arr([100.0, 100.0]), easom, -1.0),
        Problem('F10_GoldsteinPrice', 2, _arr([-2.0, -2.0]), _arr([2.0, 2.0]), goldstein_price, 3.0),
        Problem('F11_Griewank', 2, _arr([-600.0, -600.0]), _arr([600.0, 600.0]), griewank, 0.0),
        Problem('F12_HolderTable', 2, _arr([-10.0, -10.0]), _arr([10.0, 10.0]), holder_table, -19.2085),
        Problem('F13_Langermann', 2, _arr([0.0, 0.0]), _arr([10.0, 10.0]), langermann, None),
        Problem('F14_Michalewicz', 2, _arr([0.0, 0.0]), _arr([math.pi, math.pi]), michalewicz, -1.8013034100985538),
        Problem('F15_Rastrigin', 2, _arr([-5.12, -5.12]), _arr([5.12, 5.12]), rastrigin, 0.0),
        Problem('F16_Rosenbrock', 2, _arr([-2.048, -2.048]), _arr([2.048, 2.048]), rosenbrock, 0.0),
        Problem('F17_RotatedHyperEllipsoid', 2, _arr([-65.536, -65.536]), _arr([65.536, 65.536]), rotated_hyper_ellipsoid, 0.0),
        Problem('F18_Schubert', 2, _arr([-10.0, -10.0]), _arr([10.0, 10.0]), schubert, None),
        Problem('F19_Schwefel', 2, _arr([-500.0, -500.0]), _arr([500.0, 500.0]), schwefel, -837.9658),
        Problem('F20_SixHumpCamelBack', 2, _arr([-3.0, -2.0]), _arr([3.0, 2.0]), six_hump_camel, -1.031628453489877),
        Problem('F21_SumDifferentPowers', 2, _arr([-1.0, -1.0]), _arr([1.0, 1.0]), sum_different_powers, 0.0),
    ]

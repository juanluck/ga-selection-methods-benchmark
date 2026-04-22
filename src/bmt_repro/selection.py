from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


def _better(values: np.ndarray, minimize: bool) -> int:
    return int(np.argmin(values) if minimize else np.argmax(values))


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


@dataclass
class TournamentSelection:
    tournament_size: int = 4
    minimize: bool = True
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    fitness: Optional[np.ndarray] = field(init=False, default=None)

    def start_generation(self, population: np.ndarray, fitness: np.ndarray) -> None:
        self.fitness = np.asarray(fitness, dtype=float)

    def select_one(self) -> int:
        n = len(self.fitness)
        contestants = self.rng.choice(n, size=self.tournament_size, replace=n < self.tournament_size)
        idx = _better(self.fitness[contestants], self.minimize)
        return int(contestants[idx])


@dataclass
class BMTSelection(TournamentSelection):
    bipolarity: float = 0.25

    def select_pair(self) -> tuple[int, int]:
        n = len(self.fitness)
        g1 = self.rng.choice(n, size=self.tournament_size, replace=n < self.tournament_size)
        g2 = self.rng.choice(n, size=self.tournament_size, replace=n < self.tournament_size)
        i_best = int(g1[_better(self.fitness[g1], self.minimize)])
        best2 = int(g2[_better(self.fitness[g2], self.minimize)])
        worst2 = int(g2[_better(self.fitness[g2], not self.minimize)])
        second = best2 if self.rng.random() <= self.bipolarity else worst2
        return i_best, second


@dataclass
class RTSSelection(TournamentSelection):
    window_size: int = 4

    def select_one(self) -> int:
        n = len(self.fitness)
        anchor = int(self.rng.integers(0, n))
        window = self.rng.choice(n, size=self.window_size, replace=n < self.window_size)
        candidates = np.unique(np.concatenate([[anchor], window]))
        idx = _better(self.fitness[candidates], self.minimize)
        return int(candidates[idx])


@dataclass
class UTSSelection(TournamentSelection):
    winners: Optional[np.ndarray] = field(init=False, default=None)
    cursor: int = field(init=False, default=0)

    def start_generation(self, population: np.ndarray, fitness: np.ndarray) -> None:
        super().start_generation(population, fitness)
        n = len(self.fitness)
        perm = _derangement(n, self.rng)
        winners = np.empty(n, dtype=int)
        for i in range(n):
            pair = np.array([i, perm[i]], dtype=int)
            idx = _better(self.fitness[pair], self.minimize)
            winners[i] = int(pair[idx])
        self.rng.shuffle(winners)
        self.winners = winners
        self.cursor = 0

    def select_one(self) -> int:
        if self.cursor >= len(self.winners):
            self.rng.shuffle(self.winners)
            self.cursor = 0
        value = int(self.winners[self.cursor])
        self.cursor += 1
        return value


@dataclass
class FGTSSelection(TournamentSelection):
    ftour: float = 4.5
    winners: Optional[np.ndarray] = field(init=False, default=None)
    cursor: int = field(init=False, default=0)

    def start_generation(self, population: np.ndarray, fitness: np.ndarray) -> None:
        super().start_generation(population, fitness)
        n = len(self.fitness)
        ft_minus = int(np.floor(self.ftour))
        ft_plus = ft_minus + 1
        n_minus = int(np.floor(n * (ft_plus - self.ftour)))
        n_plus = n - n_minus
        winners: list[int] = []
        for _ in range(n_minus):
            contestants = self.rng.choice(n, size=ft_minus, replace=n < ft_minus)
            winners.append(int(contestants[_better(self.fitness[contestants], self.minimize)]))
        for _ in range(n_plus):
            contestants = self.rng.choice(n, size=ft_plus, replace=n < ft_plus)
            winners.append(int(contestants[_better(self.fitness[contestants], self.minimize)]))
        self.winners = np.asarray(winners, dtype=int)
        self.rng.shuffle(self.winners)
        self.cursor = 0

    def select_one(self) -> int:
        if self.cursor >= len(self.winners):
            self.rng.shuffle(self.winners)
            self.cursor = 0
        value = int(self.winners[self.cursor])
        self.cursor += 1
        return value


@dataclass
class CooperativeSelection(TournamentSelection):
    association_size: int = 4
    coop_fitness: Optional[np.ndarray] = field(init=False, default=None)

    def start_generation(self, population: np.ndarray, fitness: np.ndarray) -> None:
        super().start_generation(population, fitness)
        self.coop_fitness = np.asarray(fitness, dtype=float).copy()

    def select_one(self) -> int:
        n = len(self.coop_fitness)
        size = max(3, self.association_size)
        contestants = self.rng.choice(n, size=size, replace=n < size)
        scores = self.coop_fitness[contestants]
        order = np.argsort(scores) if self.minimize else np.argsort(scores)[::-1]
        ranked = contestants[order]
        winner, second, third = ranked[:3]
        self.coop_fitness[winner] = 0.5 * (self.coop_fitness[second] + self.coop_fitness[third])
        return int(winner)


def make_selector(name: str, rng: np.random.Generator, *, tournament_size: int, bipolarity: float,
                  fgts_ftour: float, rts_window: int, association_size: int, minimize: bool):
    key = name.lower()
    if key == 'st':
        return TournamentSelection(tournament_size=tournament_size, minimize=minimize, rng=rng)
    if key == 'bmt':
        return BMTSelection(tournament_size=tournament_size, bipolarity=bipolarity, minimize=minimize, rng=rng)
    if key == 'rts':
        return RTSSelection(tournament_size=tournament_size, window_size=rts_window, minimize=minimize, rng=rng)
    if key == 'uts':
        return UTSSelection(tournament_size=tournament_size, minimize=minimize, rng=rng)
    if key == 'fgts':
        return FGTSSelection(tournament_size=tournament_size, ftour=fgts_ftour, minimize=minimize, rng=rng)
    if key == 'cs':
        return CooperativeSelection(tournament_size=tournament_size, association_size=association_size, minimize=minimize, rng=rng)
    raise ValueError(f'Unknown selection method: {name}')

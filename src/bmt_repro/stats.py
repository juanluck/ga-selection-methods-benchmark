from __future__ import annotations

import itertools
import math
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def summarize_median_std(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(['population_size', 'problem', 'algorithm'])['best_value']
        .agg(median='median', std='std', mean='mean')
        .reset_index()
    )


def friedman_by_population(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pop_size, sub in summary.groupby('population_size'):
        pivot = sub.pivot(index='problem', columns='algorithm', values='median')
        algs = list(pivot.columns)
        ranks = np.vstack([rankdata(row, method='average') for row in pivot.to_numpy()])
        mean_ranks = ranks.mean(axis=0)
        stat = friedmanchisquare(*[pivot[a].to_numpy() for a in algs])
        for alg, mr in zip(algs, mean_ranks):
            rows.append({
                'population_size': pop_size,
                'algorithm': alg,
                'mean_rank': float(mr),
                'friedman_statistic': float(stat.statistic),
                'p_value': float(stat.pvalue),
            })
    return pd.DataFrame(rows)


def _wilcoxon_ranks(x: np.ndarray, y: np.ndarray):
    d = x - y
    nz = d != 0
    d = d[nz]
    if d.size == 0:
        return 0.0, 0.0, 1.0
    ranks = rankdata(np.abs(d), method='average')
    r_plus = float(np.sum(ranks[d > 0]))
    r_minus = float(np.sum(ranks[d < 0]))
    p = float(wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', correction=False, mode='auto').pvalue)
    return r_plus, r_minus, p


def wilcoxon_by_population(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pop_size, sub in summary.groupby('population_size'):
        pivot = sub.pivot(index='problem', columns='algorithm', values='median')
        algs = list(pivot.columns)
        for a, b in itertools.combinations(algs, 2):
            r_plus, r_minus, p = _wilcoxon_ranks(pivot[a].to_numpy(), pivot[b].to_numpy())
            rows.append({
                'population_size': pop_size,
                'algorithm_a': a,
                'algorithm_b': b,
                'R_plus': r_plus,
                'R_minus': r_minus,
                'p_value': p,
            })
    return pd.DataFrame(rows)

from __future__ import annotations

import os


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def limit_library_threads(threads: int = 1, *, force: bool = False) -> None:
    """Limit BLAS/OpenMP-style thread pools before importing NumPy/SciPy."""
    value = str(int(threads))
    for name in _THREAD_ENV_VARS:
        if force or name not in os.environ:
            os.environ[name] = value


def default_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def recommended_jobs(max_jobs: int = 12) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(max_jobs, cpu_count))


def clamp_jobs(requested: int | None, *, default: int | None = None, max_jobs: int | None = 12) -> int:
    if requested is None:
        requested = default if default is not None else default_jobs()
    jobs = max(1, int(requested))
    if max_jobs is not None:
        jobs = min(jobs, int(max_jobs))
    return jobs

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, TypeVar


TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


def process_map(
    worker: Callable[[TaskT], ResultT],
    tasks: list[TaskT],
    *,
    jobs: int,
    initializer: Callable[[], None] | None = None,
    progress_label: str | None = None,
) -> list[ResultT]:
    if jobs <= 1:
        results: list[ResultT] = []
        total = len(tasks)
        for index, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if progress_label:
                print(f"[{index}/{total}] {progress_label}")
        return results

    chunksize = max(1, len(tasks) // max(1, jobs * 8))
    results = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=initializer) as executor:
        total = len(tasks)
        for index, result in enumerate(executor.map(worker, tasks, chunksize=chunksize), start=1):
            results.append(result)
            if progress_label:
                print(f"[{index}/{total}] {progress_label}")
    return results

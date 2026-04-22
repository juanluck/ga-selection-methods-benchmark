from __future__ import annotations

import hashlib


ALGORITHM_NAMES = ("BMT", "CS", "FGTS", "RTS", "ST", "UTS")


def stable_problem_offset(name: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo

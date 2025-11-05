"""Deterministic optimizer used in web tests."""

from __future__ import annotations

from lightbench.leaderboard import Item


class _DummyOptimizer:
    def __call__(self, item: Item) -> float:
        return 1.0 - float(item.difficulty)

    def metadata(self) -> dict[str, str]:
        return {"source": "dummy"}


def build_optimizer() -> _DummyOptimizer:
    return _DummyOptimizer()

"""Curated item bank for the web service."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from lightbench.leaderboard import Item

DEFAULT_AXES = ("big_data", "large_model", "dense")


DEFAULT_ITEM_BANK = (
    Item(
        dataset="xor_digit",
        model_size="tiny",
        budget=256,
        loadings=[0.9, 0.2, 0.1],
        difficulty=-0.2,
        cost=0.2,
        tags=("warmup", "tabular"),
    ),
    Item(
        dataset="mnist",
        model_size="small",
        budget=2048,
        loadings=[0.7, 0.2, 0.6],
        difficulty=0.1,
        cost=1.0,
        tags=("vision",),
    ),
    Item(
        dataset="cifar10",
        model_size="medium",
        budget=8192,
        loadings=[0.3, 0.8, 0.5],
        difficulty=0.4,
        cost=3.0,
        tags=("vision", "cnn"),
    ),
    Item(
        dataset="transfer_shift",
        model_size="large",
        budget=32768,
        loadings=[0.4, 0.9, 0.7],
        difficulty=0.7,
        cost=7.5,
        tags=("nlp", "transfer"),
    ),
)


def serialize_item(item: Item) -> Dict[str, Any]:
    return {
        "dataset": item.dataset,
        "model_size": item.model_size,
        "budget": item.budget,
        "loadings": list(item.loadings),
        "difficulty": item.difficulty,
        "cost": item.cost,
        "tags": list(item.tags),
        "name": item.name,
    }


def deserialize_item(payload: Mapping[str, Any]) -> Item:
    return Item(
        dataset=str(payload["dataset"]),
        model_size=str(payload["model_size"]),
        budget=int(payload["budget"]),
        loadings=[float(v) for v in payload["loadings"]],
        difficulty=float(payload["difficulty"]),
        cost=float(payload["cost"]),
        tags=tuple(payload.get("tags", [])),
        name=payload.get("name"),
    )

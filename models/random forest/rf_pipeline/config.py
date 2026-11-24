"""Versioned configuration; dates are week STARTS (Monday), not week ends."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import pandas as pd


def monday(value: str, name: str = "week") -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be an ISO date (Monday)") from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize() or result.weekday() != 0:
        raise ValueError(f"{name} must be a timezone-naive Monday at midnight: {value}")
    return result


@dataclass(frozen=True)
class Config:
    data_start_week: str = "2023-07-03"
    data_end_week: str = "2025-12-22"
    train_end_week: str = "2025-03-24"
    validation_end_week: str = "2025-08-04"
    assume_complete_weeks: bool = False
    min_history_weeks: int = 8
    validation_blocks: int = 3
    random_state: int = 317
    n_jobs: int = -1
    candidates: list[dict] = field(default_factory=lambda: [
        {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.8},
        {"n_estimators": 80, "max_depth": 14, "min_samples_leaf": 5, "max_features": 0.8},
    ])

    def __post_init__(self):
        dates = [monday(getattr(self, name), name) for name in (
            "data_start_week", "train_end_week", "validation_end_week", "data_end_week")]
        if not dates[0] < dates[1] < dates[2] < dates[3]:
            raise ValueError("Require data_start_week < train_end_week < validation_end_week < data_end_week")
        if type(self.assume_complete_weeks) is not bool:
            raise ValueError("assume_complete_weeks must be a JSON boolean")
        if type(self.min_history_weeks) is not int or self.min_history_weeks < 8:
            raise ValueError("min_history_weeks must be an integer >= 8")
        if type(self.validation_blocks) is not int or self.validation_blocks < 1:
            raise ValueError("validation_blocks must be a positive integer")
        if (dates[2] - dates[1]).days // 7 < self.validation_blocks:
            raise ValueError("Not enough validation weeks for validation_blocks")
        if (dates[1] - dates[0]).days // 7 < self.min_history_weeks:
            raise ValueError("Training period must include a full history window and a target week")
        if type(self.random_state) is not int or type(self.n_jobs) is not int or self.n_jobs == 0:
            raise ValueError("random_state/n_jobs must be integers and n_jobs must not be 0")
        if not self.candidates:
            raise ValueError("At least one Random Forest candidate is required")
        allowed = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
        for candidate in self.candidates:
            if not isinstance(candidate, dict) or set(candidate) != allowed:
                raise ValueError(f"Each candidate must specify exactly {sorted(allowed)}")
            for name in ("n_estimators", "min_samples_leaf"):
                if type(candidate[name]) is not int or candidate[name] < 1:
                    raise ValueError(f"{name} must be a positive integer")
            depth = candidate["max_depth"]
            if depth is not None and (type(depth) is not int or depth < 1):
                raise ValueError("max_depth must be null or a positive integer")
            value = candidate["max_features"]
            if type(value) not in (float, int) or not 0 < value <= 1:
                raise ValueError("max_features must be in (0, 1]")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Config must be a JSON object")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)

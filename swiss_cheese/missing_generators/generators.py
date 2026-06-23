from dataclasses import dataclass
from typing import override
from warnings import deprecated
import numpy as np
import pandas as pd
from enum import Enum
from .utils import max_missing_percentage, frequency_encode, Gauss


class MCAR:
    def __init__(self, random_seed: None | int = None):
        self.seed: int | None = random_seed

    def __call__(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        n = df.size
        n_features = df.shape[1]
        n_missing = round(n * alpha)
        if n_missing == 0:
            return df
        df = df.copy()  # .astype(np.float64)

        rng = np.random.default_rng(self.seed)
        indices = np.arange(n)
        grouped = indices.reshape(-1, n_features)
        safeguard = rng.integers(low=0, high=n_features, size=grouped.shape[0])
        keep = np.ones(grouped.shape, dtype=bool)
        keep[np.arange(grouped.shape[0]), safeguard] = False
        if keep.sum() < n_missing:
            raise ValueError(
                f"Cannot comply, max missing rate is {
                    max_missing_percentage(df)
                }, requested is: {alpha}"
            )
        indices = grouped[keep].flatten()
        indices = rng.choice(indices, size=n_missing, replace=False)
        values = np.zeros(n)
        values[indices] = 1
        mask = values.reshape(df.shape)
        df[mask.astype(bool)] = pd.NA
        return df

    @override
    def __repr__(self):
        return "MCAR"

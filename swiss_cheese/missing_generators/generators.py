from typing import override
from warnings import warn
import numpy as np
import pandas as pd
from .utils import max_missing_percentage  # , frequency_encode, Gauss


class MCAR:
    def __init__(
            self, 
            random_seed: None | int = None,
            *,
            n_features: int | None = None, 
            allowed_features: int | list[int] | None = None, # number of features that are allows missing
            spread: float = 1.0, # how many datapoints should be affected
    ):
        self.seed: int | None = random_seed
        self.rng = np.random.default_rng(seed=random_seed)
        self.spread = spread
        if isinstance(allowed_features, int):
            if n_features is None:
                raise ValueError(f"In order to drop random features provide a 'n_features' value, that gives an upper limit")
            self.allowed_features = rng.integers(low=0, high=n_features, size=allowed_features)
        else:
            self.allowed_features = allowed_features

    def _verify_input(self, df, missing_rate) -> pd.DataFrame:
        missing_rate = _adjust_missing_rate(df, missing_rate)
        n_features = df.shape[1]
        n_missing = round(df.size * missing_rate)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df = df.copy()  
        return df, n_missing

    def __call__(self, df: pd.DataFrame | np.ndarray, missing_rate: float) -> pd.DataFrame | np.ndarray:
        df, n_missing = self._verify_input(df, missing_rate)
        if n_missing == 0:
            return df

        indices = self._select_feature_to_keep_colwise(df, n_missing)
        mask = self._make_mask(indices, df, n_missing)
        df[mask] = pd.NA
        return df

    def _make_mask(self, indices, df, n_missing) -> np.ndarray[bool]:
        missing_indicies = self.rng.choice(indices, size=n_missing, replace=False)
        mask = np.zeros_like(df).flatten()
        mask[missing_indicies] = 1
        mask = mask.reshape(df.shape)
        return mask.astype(bool)

    def _select_feature_to_keep_colwise(self, df, n_missing) -> np.ndarray[int]:
        indices = np.arange(df.size)
        grouped = indices.reshape(-1, df.shape[1])
        safeguard = self.rng.integers(low=0, high=df.shape[1], size=grouped.shape[0])
        keep = np.ones(grouped.shape, dtype=bool)
        keep[np.arange(grouped.shape[0]), safeguard] = False
        if keep.sum() < n_missing:
            raise ValueError(
                f"Cannot comply, max missing rate is {
                    max_missing_percentage(df)
                }, requested is: {missing_rate}"
            )
        indices = grouped[keep].flatten()
        return indices

    @override
    def __repr__(self):
        return "MCAR"


def _adjust_missing_rate(df: pd.DataFrame | np.ndarray, missing_rate: float) -> float:
    max = 1.0 - (1.0 / df.shape[1])
    if missing_rate > max:
        warn(
            f"Warning: Missing rate too high to ensure MAR properties! Maximum missing rate: {
                max}", UserWarning
        )
        return max
    return missing_rate

from dataclasses import dataclass
import numpy as np
import pandas as pd
from enum import Enum
from .utils import max_missing_percentage, frequency_encode, Gauss

class MCAR:
    def __init__(self, random_seed=None):
        self.seed = random_seed

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


@dataclass
class MNARParamters:
    means: None | float = None
    variances: None | float = None
    # weights: None | list[float] = None
    randomize: bool = False

    def __post_init__(self):
        if self.means is None:
            self.means = 0.5 if not self.randomize else float(np.random.rand(1))
        if self.variances is None:
            self.variances = 0 if not self.randomize else float(np.random.rand(1))
        # if self.weights is None:
        #     self.weights = [1 / len(self.means)]


class MnarType(Enum):
    MoG = "Mixture-of-Gaussians"


class MNAR:
    def __init__(
        self,
        params: MNARParamters,
        mnar_type: MnarType = MnarType.MoG,
        random_seed=None,
    ):
        self.mnar_type = mnar_type
        self.params = params
        self.seed = random_seed

    def __call__(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        match self.mnar_type:
            case MnarType.MoG:
                return self.mog(df, alpha)
            case _:
                raise ValueError(f"Unkown MNAR type: {self.mnar_type}")

    def mog(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        df, old = frequency_encode(df, False), df.copy()

        def drop(df, gaussians, dimensions):
            g = gaussians()
            diff = np.abs(df - g)
            diff[np.isnan(diff)] = np.inf
            closest = np.argmin(diff, axis=0)[dimensions]
            # Modify the DataFrame in-place
            for row_idx, col_idx in zip(closest, dimensions):
                df.iloc[row_idx, col_idx] = np.nan
            # df.values[closest, dimensions] = np.nan
            assert df.isna().any().any(), f"no values missing\ndiff:\n{diff}\nclosest:{len(
                closest)} shape:{df.shape}\n{closest}\ndim{dimensions}\n{df.values[closest, dimensions]}"

        def n_features(df, alpha):
            step_size = 1 / df.size
            mp = df.isna().sum().sum() / df.size
            d = df.shape[1]
            if mp + d * step_size <= alpha:
                return np.arange(df.shape[1])
            else:
                return np.random.choice(
                    np.arange(df.shape[1]), int(
                        np.ceil((alpha - mp) / step_size))
                )
        df = df.astype(float).copy()
        n = df.size

        n_missing = alpha * n
        step = df.shape[1]/n
        total_steps = n_missing / step
        i = 0

        quantiles = df.quantile(self.params.means)
        variances = self.params.variances * (df.max() - df.min())
        gaussians = Gauss(quantiles, variances, self.seed)
        while df.isna().sum().sum() / n < alpha:
            drop(df, gaussians, n_features(df, alpha))
            i += 1
            if i > total_steps:
                assert False, f"steps:{i}, missing:{df.isna().sum().sum()}, total_steps:{
                    total_steps}, should be missing: {n_missing}"
        mask = df.isna()
        old[mask] = pd.NA
        return old

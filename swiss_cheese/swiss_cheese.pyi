from enum import Enum
from numpy import uint64
import pandas as pd


class MnarType(Enum):
    MoG = "Mixture-of-Gaussians"


class MCAR:
    def __init__(self, random_seed: uint64 | None = None) -> None: ...
    def __call__(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame: ...


class MNARParamters:
    means: None | float = None
    variances: None | float = None
    randomize: bool = False


class MNAR:
    def __init__(
        self,
        params: MNARParamters,
        mnar_type: MnarType = MnarType.MoG,
        random_seed: uint64 | None = None,
    ) -> None:
        ...

    def __call__(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame: ...


class MNARrs:
    def __init__(
        self,
        mean: float | None = None,
        variance: float | None = None,
        seed: uint64 | None = None,
        n_workers: uint64 | None = None,
    ) -> None:
        ...

    def __call__(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame: ...


class MAR:
    def __init__(
        self,
        seed: uint64 | None = None,
        n_workers: uint64 | None = None,
    ) -> None:
        ...

    def __call__(self, df: pd.DataFrame, alpha: float) -> pd.DataFrame: ...

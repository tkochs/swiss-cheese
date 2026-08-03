from enum import Enum
# from numpy import uint64
import pandas as pd


class MCAR:
    def __init__(self, random_seed: int | None = None) -> None: ...
    def __call__(self, df: pd.DataFrame,
                 missing_rate: float) -> pd.DataFrame: ...


class MNAR:
    def __init__(
        self,
        mean: float | None = None,
        variance: float | None = None,
        mode: str = "GM",
        max_missing_per_column: float = 0.8,
        random_seed: int | None = None,
    ) -> None:
        ...

    def __call__(self, df: pd.DataFrame,
                 missing_rate: float) -> pd.DataFrame: ...


class MAR:
    def __init__(
        self,
        mean: float | None = None,
        variance: float | None = None,
        mode: str = "GM",
        random_seed: int | None = None,
        max_missing_per_column: float = 0.8,
    ) -> None:
        ...

    def __call__(self, df: pd.DataFrame,
                 missing_rate: float) -> pd.DataFrame: ...

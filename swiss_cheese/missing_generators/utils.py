import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from scipy.stats import norm


def max_missing_percentage(df: pd.DataFrame) -> float:
    rows, _ = df.shape
    n = df.size
    return 1 - rows / n


class Gauss:
    def __init__(self, means, variances, seed):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.means = means
        self.variances = variances
        self.gaussians = [lambda: self.rng.normal(
            q, v) for q, v in zip(means, variances)]

    def __call__(self):
        samples = np.array([g() for g in self.gaussians])
        assert not np.isnan(samples).all(), f"error while sampling from gaussians: \n{
            self.means}\n{self.variances}"
        return samples


def frequency_encode(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    """
    Replaces categorical values with their frequency (count) in each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    inplace : bool
        If True, modifies df in place. Otherwise returns a copy.

    Returns
    -------
    pd.DataFrame
        Frequency-encoded dataframe
    """
    if not inplace:
        df = df.copy()

    cat_cols = df.select_dtypes(exclude=["number"]).columns

    for col in cat_cols:
        freq = df[col].value_counts(dropna=False)
        df[col] = df[col].map(freq)

    return df


def fselect(df, target, k=5, random_state=None):
    """
    Returns top k features ranked by information gain (mutual information).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features and target
    target_col : str
        Name of the target column
    k : int
        Number of top features to return
    random_state : int, optional
        For reproducibility

    Returns:
    --------
    pd.DataFrame
        Top k features sorted by information gain (descending)
    """
    # Split features and target
    X = df
    y = target

    # Compute mutual information
    mi = mutual_info_classif(X, y, random_state=random_state)

    # Create sorted dataframe
    feature_importance = (
        pd.DataFrame({
            "feature": X.columns,
            "information_gain": mi
        })
        .sort_values(by="information_gain", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )

    return feature_importance

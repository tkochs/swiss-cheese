from swiss_cheese.missing_generators import MCAR, MNAR, MNARParamters
from sklearn.datasets import make_classification
from swiss_cheese.missing_generators import max_missing_percentage
import numpy as np
import pandas as pd
import pytest


def data(kind="MCAR"):
    rng = np.random.default_rng()
    match kind:
        case "MCAR":
            return pd.DataFrame(
                rng.random((10, 5)),  # 10 rows × 5 columns
                columns=[f"col{i}" for i in range(1, 6)],
            )
        case "WithStr":
            return pd.DataFrame(
                {
                    "col1": np.arange(10),
                    "col2": np.arange(10),
                    "col3": np.arange(10),
                    "col4": np.arange(10),
                    "col5": ["a"]*5+["b"]*5,
                }  # 10 rows × 5 columns
            )
        case "MNAR":
            return pd.DataFrame(
                {
                    "col1": np.arange(10),
                    "col2": np.arange(10),
                    "col3": np.arange(10),
                    "col4": np.arange(10),
                    "col5": np.arange(10),
                }  # 10 rows × 5 columns
            )
        case _:
            raise ValueError(f"No dataframe for this type: {kind}")


def test_mcar():
    df = data()
    n = df.size
    missing = MCAR()(df, 0.5)
    assert not df.isna().any().any(), "introduced missing in original data"

    # ensure no row or column is fully missing
    rows_all_nan = missing.isna().all(axis=1).sum()
    cols_all_nan = missing.isna().all(axis=0).sum()
    assert rows_all_nan == 0, "Some rows are fully NaN"
    assert cols_all_nan == 0, "Some columns are fully NaN"
    assert np.isclose(missing.isna().sum().sum() / n, 0.5, atol=1 / n)


def test_mcar_strdata():
    df = data("WithStr")
    n = df.size
    missing = MCAR()(df, 0.5)
    assert not df.isna().any().any(), "introduced missing in original data"

    # ensure no row or column is fully missing
    rows_all_nan = missing.isna().all(axis=1).sum()
    cols_all_nan = missing.isna().all(axis=0).sum()
    assert rows_all_nan == 0, "Some rows are fully NaN"
    assert cols_all_nan == 0, "Some columns are fully NaN"
    assert np.isclose(missing.isna().sum().sum() / n, 0.5, atol=1 / n)


def test_mcar_error():
    with pytest.raises(ValueError):
        df = data()
        missing = MCAR()(df, 0.9)


def test_max_percentage():
    p = max_missing_percentage(data())
    print(p)
    assert np.isclose(p, 1 - 10 / 50)


def test_mnar_min():
    df = data("MNAR")
    missing = MNAR(MNARParamters(means=[1]))(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(df)
    print(missing)
    print(df.max())
    print(missing < df.max())
    assert (missing[~missing.isna()].max() < df.max()).all(), "max test"
    assert missing.isna().sum().sum() == 5


def test_mnar_max():
    df = data("MNAR")
    missing = MNAR(MNARParamters(means=[0]))(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    print(missing > df.min())
    assert (missing[~missing.isna()].min() > df.min()).all(), "min test"
    assert missing.isna().sum().sum() == 5


def test_mnar_median():
    df = data("MNAR")
    missing = MNAR(MNARParamters(means=[0.5]))(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(df)
    print(missing)
    # print(df.quantile(0.5))
    print(missing.iloc[4, 2])
    assert np.isnan(missing.iloc[4, :]).all(), "median test"
    assert missing.isna().sum().sum() == 5


def test_mnar_var():
    df = data("MNAR")
    missing = MNAR(MNARParamters(means=[0], variances=[1]))(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 5
    assert not missing.iloc[0, :].isna().all()


def test_mnar_alpha():
    df = data()
    missing = MNAR(MNARParamters())(df, 0.1)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 5

    missing = MNAR(MNARParamters())(df, 0.2)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 10

    missing = MNAR(MNARParamters())(df, 0.15)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 8
    assert not df.isna().any().any(), "introduced missing in original data"


from swiss_cheese import MCAR, MNAR, MAR
from swiss_cheese.utils import max_missing_percentage
import numpy as np
import pandas as pd
import pytest


def data(kind: str = "MCAR"):
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
    with pytest.warns(UserWarning):
        df = data()
        missing = MCAR()(df, 0.9)
    n_miss = missing.isna().sum().sum()
    print(missing)
    print(n_miss)
    assert n_miss == df.size * 0.8


def test_mnar_warn():
    with pytest.warns(UserWarning):
        df = data()
        missing = MNAR(max_missing_per_column=1.0)(df, 1.0)
    n_miss = missing.isna().sum().sum()
    print(missing)
    print(n_miss)
    assert n_miss == df.size * 0.8


def test_max_percentage():
    p = max_missing_percentage(data())
    print(p)
    assert np.isclose(p, 1 - 10 / 50)


def test_mnar_min():
    df = data("MNAR")
    missing = MNAR(mean=1)(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(df)
    print(missing)
    print(df.max())
    print(missing < df.max())
    # assert (missing[~missing.isna()].max() < df.max()).all(), "min test"
    missing = missing.iloc[-1]
    assert missing.isna().sum() == 4, f"Expected 4 missing values, got {
        missing.isna().sum()}"


def test_mnar_max():
    df = data("MNAR")
    missing = MNAR(mean=0)(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    print(missing > df.min())
    assert missing.isna().sum().sum() == 5, "Wrong total"
    # assert (missing[~missing.isna()].min() > df.min()).all(), "max test"
    missing = missing.iloc[0]
    assert missing.isna().sum() == 4, f"Expected 4 missing values, got {
        missing.isna().sum()}"


def test_mnar_median():
    df = data("MNAR")
    missing = MNAR(mean=0.5)(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(df)
    print(missing)
    print(df.quantile(0.5))
    print(missing.iloc[4, 2])
    assert np.isnan(missing.iloc[4, :]).sum() == 4, "Complete row is missing"
    assert missing.isna().sum().sum() == 5


def test_mnar_var():
    df = data("MNAR")
    missing = MNAR(0.5, 1)(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 5
    assert not missing.iloc[4, :].isna().all(), "Entire row is NaN!"


def test_mnar_alpha():
    df = data()
    missing = MNAR()(df, 0.1)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 5

    missing = MNAR()(df, 0.2)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 10

    missing = MNAR()(df, 0.15)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 8
    assert not df.isna().any().any(), "introduced missing in original data"


def test_mnarrs():
    df = data()
    missing = MNAR(mean=0.5)(df, 0.1)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 5


def test_rs_str():
    df = data("WithStr")
    print(df.dtypes)
    missing = MNAR(mean=0.5)(df, 0.1)
    print(missing)
    print(df.min())
    assert missing.isna().sum().sum() == 5


def test_rs_stable():
    df = data()
    missing = MNAR()(df, 0.1)
    print(df)
    print(missing)
    print(df.min())
    mm = missing.isna()

    pd.testing.assert_series_equal(
        df[~mm].stack(),
        missing[~mm].stack(),
        check_names=False,
    )

    df = data("WithStr")
    missing = MNAR()(df, 0.1)
    print(df)
    print(missing)
    print(df.min())
    mm = missing.isna()

    pd.testing.assert_series_equal(
        df[~mm].stack(),
        missing[~mm].stack(),
        check_names=False,
    )


def test_mnar_max_det():
    df = data("MNAR")
    missing = MNAR(mode="max")(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    print(missing > df.min())
    # assert (missing[~missing.isna()].min() > df.min()).all(), "max test"
    assert missing.isna().sum().sum() == 5, "Wrong total"
    missing = missing.iloc[0]
    assert missing.isna().sum() == 4, f"Expected 4 missing values, got {
        missing.isna().sum()}"


def test_mnar_min_det():
    df = data("MNAR")
    missing = MNAR(mode="min")(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    print(missing > df.min())
    # assert (missing[~missing.isna()].max() < df.max()).all(), "min test"
    # assert missing.isna().sum().sum() == 4
    assert missing.isna().sum().sum() == 5, "Wrong total"
    missing = missing.iloc[-1]
    assert missing.isna().sum() == 4, f"Expected 4 missing values, got {
        missing.isna().sum()}"


def test_gm():
    df = data()
    old = df.quantile(0.75)
    missing = MNAR(mean=1., random_seed=42)(df, 0.1)
    miss = missing.quantile(0.75)

    print("Clean quantiles")
    print(old)
    print("Missing quantiles")
    print(miss)
    assert (old > miss).all(), "old quantile should be larger"


def test_mar():
    df = data()
    missing = MAR(mean=0.5)(df, 0.1)
    print(missing)
    print(df.min())
    assert missing.isna().any().any(), "No Missing values"
    assert missing.isna().sum().sum() == 5, \
        f"Wrong amount expected 5, got {missing.isna().sum().sum()}"


def test_mar_max():
    df = data()
    missing = MAR(mode="max")(df, 0.1)
    print(missing)
    print(df.min())
    assert missing.isna().any().any(), "No Missing values"
    assert missing.isna().sum().sum() == 5, \
        f"Wrong amount expected 5, got {missing.isna().sum().sum()}"


def test_mar_warning():
    # should emit warning as missing rate is too high
    df = data()
    with pytest.warns(UserWarning):
        missing = MAR()(df, 1.0)
    print(missing)
    print(df.min())
    expected = df.shape[0] * 0.8 * (df.shape[1] - 1)
    assert missing.isna().any().any(), "No Missing values"
    assert missing.isna().sum().sum() == expected, \
        f"Wrong amount expected {expected}, got {missing.isna().sum().sum()}"


def test_block():
    df = data()
    missing = MNAR(mode="block", random_seed=42)(df, 0.1)
    assert missing.isna().any().any(), "No Missing values"
    assert missing.isna().sum().sum() == 5, \
        f"Wrong amount expected 5, got {missing.isna().sum().sum()}"

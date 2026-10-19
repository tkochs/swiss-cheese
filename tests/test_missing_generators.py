from swiss_cheese import MCAR, MNAR, MAR
from swiss_cheese.utils import max_missing_percentage
import numpy as np
import pandas as pd
import pytest
import warnings

ALL_GENERATORS = [
    MCAR(),
    MNAR(0.75, 0.01),
    MAR(0.75, 0.01),
    MNAR(mode="block"),
    MNAR(mode="blob"),
]

ALL_GENERATORS_DET = [
    MCAR,
    MNAR,
    MAR,
]


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


################################################## ALL #########################################################

class TestGeneralProperties:
    @pytest.mark.parametrize("generator", ALL_GENERATORS, ids=repr)
    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.15, 0.2, 0.4, 0.5])
    def test_alpha(self, generator, alpha):
        df = data()
        missing: pd.DataFrame = generator(df, alpha)
        print(missing)
        print(df.min())
        assert missing.isna().sum().sum() == np.ceil(df.size*alpha), f"Expected: {np.ceil(df.size*alpha)} Actual {missing.isna().sum().sum()}"
        assert not df.isna().any().any(), "introduced missing in original data"

    @pytest.mark.parametrize("generator", ALL_GENERATORS, ids=repr)
    def test_n_misisng(self, generator):
        df = data()
        n = df.size
        missing = generator(df, 0.5)
        assert not df.isna().any().any(), "introduced missing in original data"

        # ensure no row or column is fully missing
        rows_all_nan = missing.isna().all(axis=1).sum()
        cols_all_nan = missing.isna().all(axis=0).sum()
        assert rows_all_nan == 0, "Some rows are fully NaN"
        assert cols_all_nan == 0, "Some columns are fully NaN"
        assert np.isclose(missing.isna().sum().sum() / n, 0.5, atol=1 / n), f"{
            generator} produces wrong n missing expected: {0.5}, actual: {missing.isna().sum().sum() / n}"


    @pytest.mark.parametrize("generator", ALL_GENERATORS, ids=repr)
    def test_strdata(self, generator):
        df = data("WithStr")
        n = df.size
        missing = generator(df, 0.5)
        assert not df.isna().any().any(), "introduced missing in original data"

        # ensure no row or column is fully missing
        rows_all_nan = missing.isna().all(axis=1).sum()
        cols_all_nan = missing.isna().all(axis=0).sum()
        assert rows_all_nan == 0, "Some rows are fully NaN"
        assert cols_all_nan == 0, "Some columns are fully NaN"
        assert np.isclose(missing.isna().sum().sum() / n, 0.5, atol=1 / n)

    @pytest.mark.parametrize("generator", ALL_GENERATORS_DET, ids=repr)
    def test_determernism(self, generator):
        g = generator(random_seed=42)
        df = data()
        mm1 = g(df, 0.2).isna()
        g = generator(random_seed=42)
        mm2 = g(df, 0.2).isna()
        assert mm1.equals(mm2), f"MV generator {generator.__name__} produced non-deterministic values"

################################################## MCAR #########################################################

def test_mcar_error():
    with pytest.warns(UserWarning):
        df = data()
        missing = MCAR()(df, 0.9)
    n_miss = missing.isna().sum().sum()
    print(missing)
    print(n_miss)
    assert n_miss == df.size * 0.8

def test_mcar_allowed():
    df = data()
    missing = MCAR(allowed_features=[0])(df, 0.1)
    assert not missing.isna().iloc[:, 1:].any().any(), "missing values in not allowed column"
    assert missing.isna().iloc[:, 0].any().any(), "no missing values allowed column"

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_mcar_n_allowed(n):
    df = data()
    # more missing values than fit into n-1 columns, so all n columns must be hit
    missing_rate = (n - 0.5) / df.shape[1]
    missing = MCAR(n_features=df.shape[1], allowed_features=n)(df, missing_rate)
    n_missing_cols = missing.isna().any(axis=0).sum()
    assert n_missing_cols == n, f"Expected {n} columns with missing values, actual: {n_missing_cols}"

@pytest.mark.parametrize("spread", [0.4, 0.6, 0.8])
def test_mcar_spread(spread):
    df = data()
    # 15 missing values need at least 4 rows (4 removable cells per row), all spreads fit
    missing = MCAR(spread=spread)(df, 0.3)
    n_missing_rows = missing.isna().any(axis=1).sum()
    assert missing.isna().sum().sum() == 15
    assert n_missing_rows <= round(spread * df.shape[0]), f"Expected at most {
        round(spread * df.shape[0])} rows with missing values, actual: {n_missing_rows}"

@pytest.mark.parametrize("spread", [0.0, 0.1])
def test_mcar_spread_overflow(spread):
    df = data()
    # 20 missing values don't fit into the targeted rows, so exactly 5 full rows (4 cells each) must be used
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        missing = MCAR(spread=spread)(df, 0.4)
    n_missing_rows = missing.isna().any(axis=1).sum()
    assert missing.isna().sum().sum() == 20
    assert n_missing_rows == 5, f"Expected 5 rows with missing values, actual: {n_missing_rows}"

def test_mcar_spread_allowed():
    df = data()
    # 5 missing values in 2 allowed columns need 3 rows, more than the 1 targeted row
    missing = MCAR(allowed_features=[0, 1], spread=0.1)(df, 0.1)
    n_missing_rows = missing.isna().any(axis=1).sum()
    assert not missing.isna().iloc[:, 2:].any().any(), "missing values in not allowed column"
    assert n_missing_rows == 3, f"Expected 3 rows with missing values, actual: {n_missing_rows}"

@pytest.mark.parametrize("spread", [0.0, 0.1, 0.5, 0.8])
@pytest.mark.parametrize("missing_rate", [0.1, 0.4, 0.6])
def test_mcar_spread_keeps_one_per_row(spread, missing_rate):
    df = data()
    # spread 0.0 with rate 0.4 fills the 5 targeted rows completely up to the safeguard
    missing = MCAR(spread=spread)(df, missing_rate)
    rows_all_nan = missing.isna().all(axis=1).sum()
    assert rows_all_nan == 0, "Some rows are fully NaN"

@pytest.mark.parametrize("spread", [-0.1, 1.5])
def test_mcar_spread_invalid(spread):
    with pytest.raises(ValueError):
        MCAR(spread=spread)

# masks produced before spread was implemented, spread=1.0 must reproduce them exactly
@pytest.mark.parametrize(
    "seed, allowed, expected",
    [
        (0, None, [0, 11, 18, 19, 20, 22, 24, 26, 28, 31, 32, 37, 38, 41, 43]),
        (0, [0, 2], [0, 5, 7, 10, 12, 17, 22, 25, 27, 30, 32, 35, 37, 45, 47]),
        (42, None, [5, 7, 16, 19, 20, 23, 25, 26, 27, 31, 33, 35, 39, 46, 48]),
        (42, [0, 2], [0, 2, 5, 7, 10, 12, 17, 20, 22, 30, 32, 35, 40, 45, 47]),
    ],
)
def test_mcar_spread_one_unchanged(seed, allowed, expected):
    df = data()
    missing = MCAR(seed, allowed_features=allowed, spread=1.0)(df, 0.3)
    assert np.flatnonzero(missing.isna().to_numpy()).tolist() == expected

def test_max_percentage():
    p = max_missing_percentage(data())
    print(p)
    assert np.isclose(p, 1 - 10 / 50)

##################################### MAR #################################################

def test_mnar_warn():
    with pytest.warns(UserWarning):
        df = data()
        missing = MNAR(max_missing_per_column=1.0)(df, 1.0)
    n_miss = missing.isna().sum().sum()
    print(missing)
    print(n_miss)
    assert n_miss == df.size * 0.8

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




def test_mnar_max_det():
    df = data("MNAR")
    assert (df.iloc[0] < df.iloc[-1]).all()
    missing = MNAR(mode="max")(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    print(missing > df.min())
    # assert (missing[~missing.isna()].min() > df.min()).all(), "max test"
    assert missing.isna().sum().sum() == 5, "Wrong total"
    missing = missing.iloc[-1]
    assert missing.isna().sum() == 4, f"Expected 4 missing values, got {
        missing.isna().sum()}"


def test_mnar_min_det():
    df = data("MNAR")
    assert (df.iloc[0] < df.iloc[-1]).all()
    missing = MNAR(mode="min")(df, 0.1)
    assert not df.isna().any().any(), "introduced missing in original data"
    print(missing)
    print(df.min())
    print(missing > df.min())
    # assert (missing[~missing.isna()].max() < df.max()).all(), "min test"
    # assert missing.isna().sum().sum() == 4
    assert missing.isna().sum().sum() == 5, "Wrong total"
    missing = missing.iloc[0]
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

##################################### MAR #################################################

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

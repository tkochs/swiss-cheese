# swiss-cheese

`swiss-cheese` generates missing values in tabular data for research and testing.
It supports pandas DataFrames containing numerical or categorical values, as well
as NumPy arrays.

## Installation

```bash
uv add swiss-cheese
```

or:

```bash
pip install swiss-cheese
```

Python 3.11 or newer is required.

## Quick start

```python
import pandas as pd

from swiss_cheese import MNAR

df = pd.DataFrame(
    {
        "age": [24, 31, 46, 52, 63],
        "group": ["A", "B", "A", "B", "A"],
    }
)

generator = MNAR(random_seed=42)
missing_df = generator(df, 0.4)
```

The second argument is the requested fraction of values to replace with missing
values. The input is not modified; the generator returns a copy with missing
values inserted.

Set `random_seed` when results need to be reproducible:

```python
from swiss_cheese import MCAR

missing_df = MCAR(random_seed=42)(df, 0.2)
```

## Generators

### Missing Completely At Random (MCAR)

`MCAR` selects values independently of the data values:

```python
from swiss_cheese import MCAR

missing_df = MCAR(random_seed=42)(df, 0.2)
```

### Missing Not At Random (MNAR)

`MNAR` selects missing values using information from the same columns. Five
modes are available; mode names are case-insensitive:

- `"gm"` (default): sample from a Gaussian distribution and remove nearby
  values. Configure it with `mean` and `variance`.
- `"min"`: prefer the smallest values.
- `"max"`: prefer the largest values.
- `"block"`: remove rectangular regions. `block_size=(width, height)` sets
  their maximum relative dimensions.
- `"blob"`: remove blob-shaped regions. `n_blobs` controls their number.

```python
from swiss_cheese import MNAR

gaussian = MNAR(mean=0.75, variance=0.01, random_seed=42)
blocks = MNAR(mode="block", block_size=(0.3, 0.5), random_seed=42)
blobs = MNAR(mode="blob", n_blobs=4, random_seed=42)

missing_df = blobs(df, 0.3)
```

### Missing At Random (MAR)

`MAR` pairs observed and target columns. It ranks rows using values in an
observed column and inserts missing values only into its paired target column.
It supports the `"gm"` (default), `"min"`, and `"max"` modes.

```python
from swiss_cheese import MAR

missing_df = MAR(mode="max", random_seed=42)(df, 0.3)
```

## Missingness constraints

All generators try to reach the requested missingness rate while keeping at
least one observed value in every row. `MNAR` and `MAR` additionally default to
`max_missing_per_column=0.8`.

If these constraints make the requested rate impossible, the rate is reduced to
the highest permitted value and a `UserWarning` is emitted. The realized rate
can also differ slightly for structured modes such as `"block"` and `"blob"`,
because their geometric regions do not necessarily contain the exact requested
number of cells.

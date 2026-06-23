# swiss-cheese
Making missing values for research purposes.
Focuses on tabular data.
Supports categorical and numerical values.

__installation:__
``` bash
uv add swiss-cheese
pip install swiss-cheese
```

__Usage:__
``` python
from swiss_cheese import MNAR

generator = MNAR()
missing_df = generator(df, 0.4) # returns np.array or DataFrame with 40% missingness
```

# Missing Completely At Random (MCAR)

  Sets $\alpha$ percentage of values to missing completely at random.
  Ensures that every element has at least one feature.

# Missing Not At Random (MNAR)
  
  Sets $\alpha$ percentage of values missing by sampling from a normal distribution and matching to the nearest data value.
  Currently $3$ modes are supported:
  #### Min
  Removes the minimum value of each feature until desired $\alpha$ has been achieved.
  #### Max
  Removes the maximum value of each feature until desired $\alpha$ has been achieved.
  #### Gaussian Missing (GM)
  Samples from a gaussian (mean, var given as parameter) and romves closest value.
  
# Missing At Random (MAR)

  Create pairs of observed and missing columns, then apply the MNAR schemes to the observed column but only set missing column to missing.
  Default mode is _GM_ but _Max_ and _Min_ are also availble.

  _Attention:_ *max_missing_per_column* defines a maximum percentage that can be miissing per column
  It will _always_ be assured that at least one column remains observed. 
  If $\alpha$ is higher than the maximum possible missing rate, it will be set to the maximum and a warning is emitted.


## Generators
All generators try to achieve the desired missingness rate, however they will ensure that every item has at least one observed feature.
If it that constraint does not allow for the desired missingness rate it will warn the user and set the missing rate to maximal possible missing rate.

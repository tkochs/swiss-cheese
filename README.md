# swiss-cheese
Making missing values for research purposes.
Focuses on tabular data.

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
  

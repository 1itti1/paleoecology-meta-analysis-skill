# Validation and uncertainty

## 1. Diagnostics

`check_normality_bootstrap()` is descriptive. Non-normal raw data do not by
themselves invalidate every bootstrap; examine the statistic, sample size,
dependence, outliers, and tail behavior.

`check_temporal_independence()` can detrend before reporting AR1 and
Durbin–Watson diagnostics. Its block-length suggestion is a starting point,
not a universal estimator. Use residuals or a time-series model when trend and
autocorrelation are both important.

`check_spatial_independence()` converts latitude/longitude to approximate local
kilometres and uses a declared radius or a k-nearest fallback. Report the
neighborhood rule, number of sites, and p-value limitations for small networks.

## 2. Moving-block bootstrap

`block_bootstrap()` implements moving circular blocks and returns an explicit
`moving_block_percentile`, `moving_block_basic`, or labelled percentile fallback
when `method='bca'` is requested. It does not claim an exact BCa interval for
dependent data. Select block length from autocorrelation diagnostics and perform
a block-length sensitivity analysis.

## 3. Robustness

`leave_one_out_validation()` evaluates the available age members rather than
only the first member. `sensitivity_analysis()` should vary one scientifically
meaningful choice at a time. `dual_indicator_check()` compares indicator sets,
but agreement does not prove a causal mechanism.

## 4. Three uncertainty layers

`propagate_three_layer_uncertainty()` can combine:

1. chronology-member sampling;
2. calibration or measurement residual noise;
3. site/sample bootstrap.

These layers are only as credible as the supplied error model. Independent
Gaussian noise is a convenience assumption, not a default law of proxy error.
Record the layer actually used, the random seed, member count, and interval
interpretation.

`propagate_continuous_uncertainty()` keeps measured sample positions by default.
Select `sampling_method='iid'` only when paired age/value observations can be
treated as independent for the target. Degenerate resamples are counted and
excluded, not silently repaired; finite member counts accompany each bin's
quantiles. A bin supported by very few members is not equivalent to a bin with
complete posterior support.

Validate numerically meaningful invariants as well as output shapes: correct
error scaling under unit changes, neutral zero-change indices, preservation of
all-missing samples, independent chronology-group draws and reference-model
agreement. See `tests/test_effect_models.py`, `test_multi_proxy.py`,
`test_workflows.py`, and `test_analysis_contracts.py`.

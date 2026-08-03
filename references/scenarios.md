# Scenario selection and orchestration

## 1. Paired comparison

Use `scenario1_proxy_validation()` when each proxy value has a defensible
reference value. The function removes non-finite pairs, selects LRR when all
values are positive, and otherwise uses Hedges' g. It returns dictionaries from
`effect_size_bca()` and `rmsep()` without converting them to tuples or scalars.

For proxy calibration, report out-of-sample RMSEP and R². Use `timeseries`
validation for ordered calibration data when temporal ordering matters.

## 2. Multi-site time-series synthesis

Use `scenario2_multi_site_synthesis()` for multiple records without a paired
study design. Each site supplies `ages` and `values`, and may supply its own age
ensemble and weight. Ragged records are retained as lists; they are not padded
with zeros.

The workflow is:

1. validate and standardize each site when appropriate;
2. align each age member to the declared grid;
3. combine site curves with missing-aware weights;
4. return the ensemble, percentile band, effective member count, and spatial
   grouping metadata.

SCC, CPS, GAM, and weighted mean are alternative sensitivity methods. A GAM
output is the `predicted` curve.

## 3. Event-window comparison

Use `scenario3_human_attribution()` and `before_after_test()` to compare windows
around a declared event. The bootstrap statistic accepts separate before and
after arrays, so the BCa callback is compatible with SciPy. The output is
labelled as an association check. Autocorrelation, overlapping windows,
confounders, and multiple testing must be discussed.

`build_indicators()` accepts user-provided taxon groups and returns
`result['indicators']`, coverage metadata, and missing requested taxa. A missing
taxon group is not silently converted to zero.

`multi_window_robustness()` is a sensitivity summary. It is not a multiplicity-
adjusted hypothesis test and should not be used as the only evidence of a
stable effect.

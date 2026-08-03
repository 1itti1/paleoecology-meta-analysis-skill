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

## 3. Heterogeneous multi-proxy synthesis

Use `scripts.multi_proxy.prepare_proxy_records()` and
`scripts.multi_proxy.multi_proxy_synthesis()` when several proxy types are
intended to inform one declared target. Provide one record per `site_id` ×
`proxy_id`, including `target`, `direction`, `unit`, and any measurement error.
The function retains raw values and creates a transformed value separately.

The estimand is deliberately two-stage:

1. standardize, orient, and align each proxy record;
2. combine proxies within site, then combine site composites regionally.

This prevents a site with four proxies from receiving four times the regional
weight of a site with one proxy. `site_weight` controls the second stage and
`weight` controls the within-site proxy stage. `weighting='precision'` uses the
median positive standardized measurement error as an optional precision factor;
use it only when the reported errors are comparable and defensible.

`measurement_correlation` is an optional correlation matrix or pair mapping for
standardized measurement-error draws within a site. It does not model shared
chronology error or proxy-process covariance. Use `chronology_group` and common
age-ensemble member indices for shared chronology structure, and use an
archive-specific hierarchical model when covariance is central to the
estimand.

Inspect `proxy_concordance`, coverage counts, and the uncertainty-layer labels.
Run `leave_one_proxy_out()` before interpreting a common composite. Proxy
agreement is descriptive evidence, not proof of a common latent mechanism.

## 4. Event-window comparison

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

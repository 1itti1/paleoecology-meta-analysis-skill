# Scenario selection and orchestration

## 1. Paired comparison

Use `scenario1_proxy_validation()` when each proxy value has a defensible
reference value. The function removes non-finite pairs, applies a supplied
calibration first, and then evaluates values in the observation unit. Default
`auto` means paired prediction bias/mean difference and RMSEP; it does not
infer a ratio-scale variable from positivity. Explicit `lnrr` and
`paired_log_ratio` use paired resampling. Paired SMD is unsupported.

For proxy calibration, report out-of-sample RMSEP and R². Use `timeseries`
validation for ordered calibration data when temporal ordering matters.

## 2. Multi-site time-series synthesis

Use `scenario2_multi_site_synthesis()` for multiple records without a paired
study design. Each site supplies `ages` and `values`, and may supply its own age
ensemble, measurement error and weight. Ragged records are retained as lists;
they are not padded with zeros. Taxa error is divided by the same scale as
the proxy value. Supply `random_state` for reproducible propagation.

The workflow is:

1. validate and standardize each site when appropriate;
2. align each age member to the declared grid;
3. combine site curves with missing-aware weights;
4. return the ensemble, finite-member percentile band, effective member count, and spatial
   grouping metadata.

SCC, CPS, GAM, and weighted mean are alternative sensitivity methods. A GAM
output is the `predicted` curve. All paths use the declared site weights.
Spatial clusters are descriptive metadata, not an automatic spatial adjustment.

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
aligned age-ensemble member IDs for shared chronology structure, and use an
archive-specific hierarchical model when covariance is central to the
estimand.

The common-target composite uses conditional Gaussian measurement errors on
the aligned grid, independent between time bins. This is an explicit working
error model, not propagation of a full sample-level observation covariance.
Keep the returned `measurement_error_model` with the uncertainty band; use a
proxy observation model when temporal error covariance or interpolation-error
covariance is part of the target.

Inspect `proxy_concordance`, coverage counts, and the uncertainty-layer labels.
Run `leave_one_proxy_out()` before interpreting a common composite. Proxy
agreement is descriptive evidence, not proof of a common latent mechanism.

## 4. Event-window comparison

Use `scenario3_human_attribution()` and `before_after_test()` to compare windows
around a declared event. The default interval uses circular-block percentile
resampling within each time window. Specify block length and inspect its
sensitivity; the default cube-root length is a starting heuristic. IID
resampling requires explicit selection. The output reports mean difference,
interval method, block length and association language; it does not attach an
independent-sample P value to a block result. With BP ages use
`time_direction='BP'` so numerically larger ages occur before the event.

`build_indicators()` accepts user-provided taxon groups and returns
`result['indicators']`, coverage metadata, and missing requested taxa. A missing
taxon group or a wholly unmeasured sample is not silently converted to zero.

`multi_window_robustness()` returns direction consistency and Bonferroni-adjusted
intervals across requested windows. It does not vote on P values or issue an
overall binary robustness verdict; overlapping windows remain dependent.

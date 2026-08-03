# Effect sizes and calibration validation

## 1. Design gate

Use effect sizes only when the data provide a defensible comparison unit:

- paired proxy versus reference observations;
- independent groups with defined sampling units;
- study-level estimates with reported sampling variances.

Do not treat every depth or time bin from a correlated record as an independent
study. For stacked time series, use dependence-aware synthesis instead.

## 2. Implemented measures

`log_response_ratio()` returns paired elementwise `log(treatment/control)` and
requires positive finite values. For independent study-level LRR, calculate the
group means, standard deviations, sample sizes, and sampling variance explicitly
before using `r_bridge.rma_random_effects()`.

`hedges_d()` returns a small-sample corrected standardized mean difference and
its approximate sampling variance. Zero pooled variance is handled explicitly;
it is not silently converted to infinity.

`effect_size_bca()` uses separate bootstrap resampling for treatment and control
when calculating Hedges' g, and reports the actual interval method. BCa may fall
back to percentile only with an explicit warning. Check small-sample and
constant-value diagnostics.

`rmsep()` returns a dictionary with `rmsep`, `bias`, `n`, and `n_missing`.
`loocv()` must refit the calibration model inside every training fold.

## 3. Interpretation

Report the estimand, whether the comparison is paired, the sampling unit, the
interval type, and the treatment of zeros. A before/after LRR or Hedges' g is an
association strength unless the design supports a counterfactual.

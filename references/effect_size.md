# Effect sizes and calibration validation

## 1. Design gate

Use effect sizes only when the data provide a defensible comparison unit:

- paired proxy versus reference observations;
- independent groups with defined sampling units;
- study-level estimates with reported sampling variances.

Do not treat every depth or time bin from a correlated record as an independent
study. For stacked time series, use dependence-aware synthesis instead.

## 2. Implemented measures

`log_response_ratio()` retains the elementwise paired `log(T_i/C_i)` array.
Its mean is a geometric-mean comparison. `effect_size_bca(..., effect_type='lnrr')`
estimates the different study-level target `log(mean(T)/mean(C))`. Choose
`paired_log_ratio` with `design='paired'` for the former mean target. Do not
combine them under one effect definition: their signs can differ.

`log_response_ratio_variance(..., design='independent')` uses independent-group
delta variance. With `design='paired'`, supply the individual-pair covariance
explicitly as `paired_covariance=Cov(T_i,C_i)`. This is a ROM variance, not the
variance of the mean paired log-ratio.

`hedges_d()` implements independent-group Hedges' g and its large-sample
sampling variance. Paired SMD has several possible denominators and is
explicitly unsupported here. Zero pooled variance is handled explicitly.

`effect_size_bca()` defaults to `design='independent'` and permits unequal
group sizes. Set `design='paired'` for shared-index resampling and jackknife.
It supports paired ROM and paired mean log-ratio, and independent Hedges' g.
BCa degeneracy can fall back to percentile with a recorded warning. Reports
include the estimand, design, resampling unit and actual interval method.

`rmsep()` returns a dictionary with `rmsep`, `bias`, `n`, and `n_missing`.
`loocv()` must refit the calibration model inside every training fold.

## 3. Interpretation

Report the estimand, whether the comparison is paired, the sampling unit, the
interval type, and the treatment of zeros. A before/after LRR or Hedges' g is an
association strength unless the design supports a counterfactual.

## 4. Study-level models

Use `analysis_contracts.run_analysis()` for the staged effect-table and routing
checks. The low-level `rma_random_effects()` additionally checks optional
`study_ids` and `sampling_covariance`; repeated IDs or non-diagonal V are
handed off to a multivariate/multilevel implementation.

The Python fallback implements DL with Wald or unmodified KH inference,
including k=2 with t degrees of freedom k-1. It reports the actual estimator,
requested estimator, test and any fallback warning. A zero KH residual scale
does not support automatic inference. The R result records R/metafor versions.

Egger uses the same standardized-effect/precision parameterization and
intercept semantics across backends. Small independent-study counts and narrow
precision ranges are explicitly exploratory; a funnel-asymmetry result is not
a binary publication-bias determination.

Formula and model references:
[metafor escalc](https://wviechtb.github.io/metafor/reference/escalc.html),
[rma.uni](https://wviechtb.github.io/metafor/reference/rma.uni.html), and
[regtest](https://wviechtb.github.io/metafor/reference/regtest.html).

# Synthesis methods

## 1. Match the method to the estimand

Use the following as transparent sensitivity methods, not as interchangeable
estimators:

| Method | Input | Interpretation |
|---|---|---|
| SCC | Comparable calibrated values | Weighted level composite |
| DCC | Values plus time-aware calibration function | Dynamically calibrated composite |
| CPS | Common-scale z-scores plus external baseline | Composite-plus-scale estimate |
| PAI | Directional changes | Agreement in sign, not magnitude |
| GAM | Time and a composite curve | Descriptive nonlinear smooth |

`scc_composite()` renormalizes weights for each time bin when observations are
missing. It requires a site × time matrix. `cps_composite()` should only be used
when the baseline mean and scale have a substantive calibration meaning.

## 2. Alignment and missingness

Align each site independently with `resample_to_grid()` or
`interpolate_no_extrapolation()`. Values outside the observed age range remain
missing. A regional estimate at a time bin is based only on sites with valid
values at that bin; report the effective site count.

## 3. GAM and LOESS

`gam_composite()` uses PyGAM when installed and returns the fitted model and the
`predicted` curve. `loess_trend()` is for visualization. Neither is a causal
model by itself. Choose smoothing complexity using a declared rule and report
the grid, effective sample size, and dependence treatment.

## 4. Monte Carlo ensembles

`monte_carlo_ensemble()` can consume a shared age ensemble or a per-site age
ensemble. Use a local `random_state`; do not rely on the global NumPy RNG. The
result is an uncertainty ensemble, not automatically a confidence interval or
Bayesian posterior. Check member-count sensitivity and label percentile bands
accurately.

`uncertainty_band()` accepts three ordered percentiles and returns lower,
median, and upper arrays. It does not infer the meaning of the interval.

## 5. Method comparison

`multi_method_cross_validation()` compares requested synthesis methods on the
same aligned inputs. Agreement among methods is sensitivity evidence, not three
independent replications. If methods disagree, inspect calibration, missingness,
age uncertainty, site weighting, and the estimand before choosing a preferred
curve.

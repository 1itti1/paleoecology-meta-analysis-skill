---
name: paleoecology-meta-analysis
description: >-
  Generic, reproducible synthesis of multi-site paleoecology and paleoclimate
  proxy records. Use for data audits, chronology-ensemble alignment, taxa or
  continuous-proxy standardization, multi-proxy common-target synthesis,
  regional time-series synthesis, leave-one-proxy validation, uncertainty
  propagation, dependence checks, and cautious climate-versus-human-activity
  comparisons. Supports pollen, diatoms,
  foraminifera, plant macrofossils, charcoal, biomarkers, isotopes, and other
  stratigraphic proxies without assuming a region, archive, proxy calibration,
  or causal interpretation.
---

# Generic paleoecology proxy synthesis

Use this skill as an analysis workflow, not as an automatic attribution engine.
Start by identifying the estimand, archive, proxy type, chronology source, and
dependence structure. Keep observed data, transformations, model outputs, and
interpretations separate.

## Operating rules

1. Audit the data before selecting a method.
2. Prefer an existing calibrated age-model posterior or ensemble. Do not label
   random perturbation of dated horizons as Bacon, Bchron, Clam, or BAM.
3. Preserve raw counts, denominators, units, site IDs, sample IDs, and source
   metadata. Never overwrite raw values with percentages or z-scores.
4. Treat compositional taxa data and continuous proxies as different channels.
5. Propagate uncertainty independently for each site before regional synthesis;
   do not silently reuse one site's age ensemble for all sites.
6. Renormalize weights when values are missing at a time bin. Do not silently
   extrapolate outside an observed age range.
7. Treat LOESS/GAM as descriptive unless the model includes an explicit
   inferential likelihood, uncertainty, and dependence structure.
8. Treat before/after results as associations unless a design includes controls,
   covariates, lags, and a defensible counterfactual.
9. Report the actual backend, package versions, random seed, member count,
   missingness, excluded samples, and all sensitivity choices.
10. Fail loudly on invalid shapes, non-positive variances, missing time axes,
    unsupported age layouts, and non-finite effect-size inputs.
11. Do not combine proxies with different targets. Keep all proxies from one
    archive in one site bootstrap cluster unless a design explicitly justifies
    a different unit of replication.

## Required data contract

Normalize inputs to a table or dictionary with these fields before analysis:

| Field | Requirement |
|---|---|
| `site_id` | Stable site/core/archive identifier |
| `sample_id` | Stable sample or depth identifier |
| `depth` | Optional but required for depth-based age modelling |
| `age` | One age per sample, with units and calendar convention |
| `age_error` | Optional 1-sigma horizon uncertainty; not an age model |
| `age_ensembles` | Preferred posterior/ensemble, shaped per site as members × samples |
| `value` or taxa columns | Raw measurement or count data |
| `count_sum` | Required for count-based taxa data when available |
| `lat`, `lon`, `elevation` | Required only for spatial analyses |
| `source`, `method`, `unit` | Provenance and measurement metadata |

For a heterogeneous multi-proxy synthesis, use one record per `site_id` ×
`proxy_id` and add `proxy_type`, `target`, `direction`, `measurement_error`,
`weight`, `site_weight`, and `chronology_group`. `target` must identify the
common construct being synthesized (for example, hydroclimate or fire
activity), not merely the study region. `direction` must make a larger
transformed value support the same target interpretation across proxies.

For ragged multi-site data, keep a list of per-site arrays or long-form rows.
Do not pad observations with zeros. Use `NaN` only for genuinely missing values.

## Workflow

### 1. Define the question and design

Write down the response variable, comparison unit, time direction, target
estimand, and whether the analysis is descriptive, predictive, associational,
or causal. Select one of these designs:

| Design | Use | Do not do |
|---|---|---|
| `paired_comparison` | Proxy versus an observed/reference value | Pool unrelated time-series points as independent replicates |
| `time_series_stacking` | Multiple records aligned to a common time grid | Apply classic independent-study effect sizes to site stacks |
| `before_after` | Event-window comparison | Call a raw pre/post contrast a causal effect |
| `calibration` | Proxy–environment training/validation | Report in-sample fit as predictive skill |

Use `scripts/scenarios.py::select_scenario` only after this design decision.

### 2. Audit and harmonize

Use `harmonize_names` for an explicit mapping table. Record unmapped names and
do not infer that absent taxa are true zeros. Check duplicate ages, age order,
units, sample counts, detection limits, taxonomic resolution, laboratory
methods, aquatic/local taxa, sediment mixing, and preservation context.

For taxa data, retain counts and calculate percentages only after checking the
pollen/proxy sum. Consider log-ratio or count-based models when closure,
zero-inflation, or unequal count sums affect the estimand. `build_indicators`
is a user-defined aggregation helper; it is not an ecological classification
validated for every region.

Use `record_preservation_bias` to document a hypothesized bias and its evidence.
Its presets are flags and sensitivity-analysis inputs, not automatic corrections.

### 3. Handle chronology

Preferred order:

1. Load a published or externally fitted posterior/ensemble with
   `consume_bacon_ages` or an equivalent loader.
2. Validate dimensions, monotonicity, age direction, calibration, and the
   number of members.
3. Use `resample_to_grid` or
   `scripts/_utils.py::interpolate_no_extrapolation` for each site/member.
4. Use `age_ensemble_from_errors` only as a transparent sensitivity analysis
   when no posterior exists. Its output must be reported as perturbed-horizon
   uncertainty, not as a fitted chronology.

The compatibility wrapper `bam_age_ensemble` is retained for old notebooks but
is deprecated and reports `method='age_perturbation'`.

### 4. Standardize without losing meaning

Use `zscore_standardize` or `standardize_continuous_proxy` only after choosing
the baseline and supplying an explicit age vector when a baseline period is
used. Record the baseline, mean, scale, missing-value rule, and denominator.

Do not standardize age, depth, latitude, or other metadata as if they were proxy
variables. For cross-site synthesis, standardize within the declared site or
calibration group and justify any cross-site scaling.

### 5. Align and synthesize

For multi-site records:

1. Interpolate each site separately onto the declared grid.
2. Use `scc_composite` or a weighted mean for already comparable calibrated
   values; weights are re-normalized per time bin.
3. Use `cps_composite` only when a common calibration scale and baseline are
   defensible.
4. Use `gam_composite` for a descriptive smooth and report smoothing choices.
5. Use `pai_composite` only for directional agreement; ties and missing values
   must remain neutral/undefined rather than being counted as decreases.
6. Compare methods as a sensitivity analysis, not as independent confirmations.

For continuous proxies, use `composite_continuous_proxy` with ragged per-site
arrays or a documented common shape. A shared age ensemble is valid only when
the same chronology applies to all sites; otherwise pass one ensemble per site.

For genuinely heterogeneous proxies, use `scripts/multi_proxy.py`:

1. Call `prepare_proxy_records()` to validate the proxy-level contract while
   preserving raw values.
2. Use `multi_proxy_synthesis()` only when records have a defensible common
   target. It standardizes and orients each record, aligns each chronology,
   combines proxies within site, and then combines sites.
3. Treat `measurement_correlation` as standardized measurement-error
   correlation within a site; provide it only when supported by the data.
4. Inspect `proxy_concordance`, `effective_proxy_count`, and
   `effective_site_count`. A composite is not evidence that disagreeing proxies
   measure one latent process.
5. Run `leave_one_proxy_out()` and report proxy-specific sensitivity. Do not
   use a common z-score to erase proxy-specific calibration, transport,
   preservation, or ecological mechanisms.

This common-target synthesis is a transparent evidence-combination layer. It
is not a proxy forward model, REVEALS, a latent-variable model, or a causal
multi-proxy attribution model. Use an archive-specific or hierarchical model
when the estimand requires those mechanisms.

### 6. Propagate uncertainty

Use `monte_carlo_ensemble`, `propagate_continuous_uncertainty`,
`multi_proxy_synthesis`, or
`propagate_three_layer_uncertainty` only after specifying which uncertainty
layers are actually available:

- chronology: posterior/ensemble member;
- measurement or calibration: reported error model or residual distribution;
- sampling/site uncertainty: an appropriate bootstrap or hierarchical model.

Use a local `random_state`. Label percentile bands as uncertainty quantiles or
bootstrap intervals; do not call them Bayesian credible intervals unless a
Bayesian model generated them. Check member-count sensitivity and Monte Carlo
stability.

### 7. Validate dependence and robustness

Run the relevant checks in `validation.py`:

- finite-value and sample-size checks;
- temporal autocorrelation after considering trend/residual structure;
- spatial autocorrelation with a declared coordinate distance and neighborhood;
- block or moving-block bootstrap for dependent series;
- leave-one-site-out and parameter sensitivity checks;
- multiple windows, indicators, taxa, or proxies with multiplicity acknowledged.
- leave-one-proxy-out sensitivity and proxy disagreement diagnostics for any
  heterogeneous synthesis.

Normality tests are diagnostics, not a prerequisite for every bootstrap. Do not
use arbitrary p-value thresholds as the sole robustness criterion.

### 8. Report and interpret

Return a dictionary containing estimates, interval type, method, input shape,
valid sample count, site count, seed, backend, and warnings. Produce a compact
provenance record with:

- input files and hashes when available;
- preprocessing and taxon mappings;
- chronology source and member shape;
- grid, baseline, weights, smoothing, bootstrap, and multiple-testing choices;
- excluded or extrapolated bins (normally none);
- limitations and alternative explanations.

Use `quasi_experiment_effect_size` and `scenario3_human_attribution` with
association language. Climate and human indicators should be compared in a
joint temporal framework when the scientific question is attribution.

## Module map

- `scripts/preprocessing.py`: validation-aware chronology consumption,
  perturbation sensitivity, standardization, interpolation, naming, and bias
  documentation.
- `scripts/synthesis.py`: SCC, DCC, CPS, PAI, descriptive GAM/LOESS, Monte
  Carlo ensembles, and method sensitivity.
- `scripts/continuous_proxy.py`: continuous-proxy calibration, ragged-site
  synthesis, uncertainty, validation, and paired comparison.
- `scripts/multi_proxy.py`: proxy-level data contract, target/direction
  harmonization, site-clustered common-target synthesis, correlated
  measurement-error propagation, concordance, and leave-one-proxy-out checks.
- `scripts/effect_size.py`: paired log-ratios, Hedges' g, BCa/percentile
  intervals, RMSEP, and LOOCV. Use only when the design supports it.
- `scripts/scenarios.py`: explicit orchestration for paired, stacked, and
  before/after designs.
- `scripts/validation.py`: dependence, bootstrap, leave-one-site-out,
  sensitivity, and uncertainty diagnostics.
- `scripts/r_bridge.py`: optional metafor backend. Treat Python fallbacks as a
  documented subset, never as numerically identical replacements.

Read only the relevant reference before a specialized analysis:
`references/preprocessing.md`, `synthesis_methods.md`, `effect_size.md`,
`scenarios.md`, `validation.md`, `python_toolchain.md`, and
`methodology_gaps.md`.

## Optional dependencies and failure policy

Core array/data operations require NumPy, SciPy, pandas, and statsmodels. See
`requirements-optional.txt` for optional Python methods. PyGAM, scikit-learn,
PySAL/ESDA, and R/metafor are optional;
detect them at call time and state the fallback. Do not import optional modules
at package import time.

If a requested method is unavailable or its assumptions are not met, return a
clear error or an explicitly labeled limited result. Never silently downgrade a
chronology model, confidence-interval type, spatial model, or causal claim.

The skill is MIT-licensed. The bundled code is a reproducible starting point,
not a substitute for archive-specific proxy ecology, age modelling, or expert
review.

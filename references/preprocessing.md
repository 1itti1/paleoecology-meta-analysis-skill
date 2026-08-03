# Preprocessing and chronology

## 1. Data audit

Before transforming a record, preserve the raw table and document:

- site/core/archive and sample identifiers;
- depth, age, calendar convention, units, and age direction;
- raw counts and denominators for compositional proxies;
- taxonomic resolution, laboratory method, detection limit, and missingness;
- local/aquatic taxa, sediment mixing, redeposition, and preservation context.

Do not interpret an absent taxon as a true zero without a detection rule.

## 2. Age ensembles

`consume_bacon_ages()` loads an externally generated age ensemble. The expected
shape is `(members, samples)`, with one complete monotonic age curve per member.
Validate the sample order and age direction before interpolation.

`age_ensemble_from_errors()` perturbs dated horizons and enforces monotonicity.
It is a sensitivity analysis, not a fitted chronology. The legacy
`bam_age_ensemble()` wrapper is deprecated and returns
`method='age_perturbation'` with an explicit warning. Do not use it as a generic
replacement for Bacon, Bchron, Clam, OxCal, or another archive-specific model.

## 3. Standardization

`zscore_standardize()` requires an explicit `ages` vector whenever
`baseline_period` is supplied. For DataFrames, pass `value_columns` so age,
depth, coordinates, and identifiers are not accidentally standardized.
Document the baseline, standard deviation convention, within-site/group rule,
and treatment of zero-variance variables.

`standardize_continuous_proxy()` provides z-score, min-max, and robust scaling.
These are scale transformations, not proxy calibrations and not ecological
models.

## 4. Alignment

`resample_to_grid()` sorts age/value pairs, averages duplicate ages, and returns
`NaN` outside the observed range. It can process one age ensemble per site when
called separately. Never pad ragged records with zeros before interpolation.

For multi-site continuous records, `composite_continuous_proxy()` accepts a list
of per-site arrays and supports shared `(members, samples)` or per-site
`(sites, members, samples)` age ensembles.

## 5. Spatial and preservation metadata

`spatial_clustering()` is a grouping aid, not a spatial ecological model. Verify
coordinate order, distance scale, cluster sensitivity, and site independence.
`record_preservation_bias()` records hypotheses and computes descriptive ratios;
its presets are not automatic corrections. Prefer a documented, archive-specific
sensitivity analysis when preservation affects the estimand.

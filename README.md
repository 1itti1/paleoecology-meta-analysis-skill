# Generic Paleoecology Meta-Analysis Skill

Reusable, validation-aware workflows for multi-site paleoecology and
paleoclimate proxy records. The skill is designed to be archive-, region-, and
proxy-agnostic: pollen, diatoms, foraminifera, plant macrofossils, charcoal,
biomarkers, isotopes, and other stratigraphic records can use the same data
contract while retaining archive-specific assumptions.

> Development release: 2.3.0. The package is a reproducible starting point,
> not a substitute for archive-specific age modelling, proxy ecology, or
> expert review.

## What changed in 2.3

- Added a proxy-level contract and a conservative heterogeneous-proxy
  synthesis layer. It standardizes and orients each record, aligns chronologies,
  combines proxies within site, and combines sites without double-counting an
  archive.
- Added site-cluster bootstrap, optional standardized measurement-error
  correlation, proxy concordance diagnostics, and leave-one-proxy-out checks.

## What changed in 2.2

- Replaced the misleading pure-Python “BAM” claim with an explicitly labelled
  dated-horizon perturbation sensitivity analysis.
- Added shared validation, safe non-extrapolating interpolation, per-bin weight
  renormalization, local random states, and ragged-site support.
- Fixed LOWESS imports, bootstrap callback signatures, Hedges' g resampling,
  continuous-proxy synthesis, GAM output contracts, and time-series diagnostics.
- Added package metadata and regression tests under `tests/`.
- Reframed event analyses as association checks unless a defensible causal
  design is supplied.

## Use it when

- auditing and harmonizing multi-site proxy records;
- aligning records with existing chronology ensembles;
- standardizing taxa or continuous proxies;
- synthesizing irregular, unequal-length time series;
- validating paired proxy predictions against observations;
- propagating chronology, calibration, and sampling uncertainty;
- checking temporal/spatial dependence and leave-one-site-out sensitivity;
- combining pollen, charcoal, isotopes, biomarkers, or other proxies only when
  they have a defensible common target;
- checking proxy disagreement and leave-one-proxy-out sensitivity;
- comparing climate and human-activity indicators without overstating causality.

## Core guardrails

1. Keep raw counts, denominators, units, and provenance separate from derived
   percentages, z-scores, and model outputs.
2. Use an externally fitted age-model posterior whenever possible.
3. Treat `age_ensemble_from_errors` as horizon-perturbation sensitivity only;
   the legacy `bam_age_ensemble` name is deprecated.
4. Interpolate each site and age member independently. Values outside the
   observed age range become `NaN`, never silent edge extrapolations.
5. Renormalize site weights at every time bin with missing data.
6. Use classic effect sizes only for a design that supports paired or
   independent-study comparisons; do not apply them to a stack of correlated
   time series.
7. Report descriptive GAM/LOESS curves as descriptive unless a full inferential
   model is specified.
8. Report before/after results as associations, not causal effects, unless the
   design includes a counterfactual, controls, and dependence-aware modelling.
9. Keep all proxies from one archive in one site bootstrap cluster. A larger
   number of measured proxies does not create more independent regional sites.

## Quick start

Run from the repository root:

```python
import numpy as np

from scripts.preprocessing import resample_to_grid
from scripts.synthesis import scc_composite
from scripts.effect_size import effect_size_bca

ages = np.array([0., 100., 200.])
values = np.array([1., 2., 3.])
grid = np.array([-50., 50., 150., 250.])

aligned = resample_to_grid(ages, values, grid)
print(aligned["resampled"])  # edge bins are NaN, not extrapolated values

sites = np.array([[1., 2., np.nan], [3., 4., 5.]])
composite = scc_composite(sites)
print(composite["composite"])

effect = effect_size_bca(
    np.array([1.2, 1.4, 1.3, 1.5] * 8),
    np.array([1.0, 1.1, 1.2, 1.0] * 8),
    n_boot=2000,
    random_state=42,
)
print(effect["effect_size"], effect["ci_lower"], effect["ci_upper"])
```

For an existing chronology ensemble:

```python
from scripts.preprocessing import consume_bacon_ages

chronology = consume_bacon_ages("age_ensemble.csv")
# chronology["age_ensembles"] has shape (members, samples)
```

When no posterior is available, use the explicitly limited sensitivity helper:

```python
from scripts.preprocessing import age_ensemble_from_errors

perturbed = age_ensemble_from_errors(
    depths, dated_ages, dated_age_errors,
    n_members=500, random_state=42,
)
```

For heterogeneous proxies with a shared target, use the explicit proxy-level
workflow:

```python
import numpy as np

from scripts.multi_proxy import multi_proxy_synthesis, leave_one_proxy_out

records = [
    {
        "site_id": "core_A",
        "proxy_id": "pollen",
        "proxy_type": "taxa",
        "target": "vegetation_change",
        "ages": np.array([0., 100., 200.]),
        "values": np.array([12., 18., 24.]),
        "direction": "positive",
        "measurement_error": 0.5,
    },
    {
        "site_id": "core_A",
        "proxy_id": "charcoal",
        "proxy_type": "continuous",
        "target": "vegetation_change",
        "ages": np.array([0., 100., 200.]),
        "values": np.array([8., 5., 2.]),
        "direction": "negative",
        "measurement_error": 0.2,
    },
]

result = multi_proxy_synthesis(
    records,
    np.array([0., 50., 100., 150., 200.]),
    n_members=500,
    random_state=42,
)
loo = leave_one_proxy_out(records, np.array([0., 50., 100., 150., 200.]))
```

This is a transparent common-target evidence synthesis, not a latent-variable,
REVEALS, or proxy-specific forward model. Different targets must be analyzed
separately and compared in a joint temporal framework.

## Data contract

Each observation should retain `site_id`, `sample_id`, `age`, optional `depth`,
raw proxy/count values, units, source, and measurement method. For taxa data,
retain the pollen/proxy sum or equivalent denominator. For multi-site analyses,
pass ragged per-site arrays rather than padding with zeros. A per-site age
ensemble is represented as members × samples; a shared ensemble is valid only
when the chronology truly applies to every site.

For multi-proxy analyses, use one record per `site_id` × `proxy_id` and retain
`proxy_type`, `target`, `direction`, `unit`, `measurement_error`, `weight`,
`site_weight`, and `chronology_group`. Do not combine records with different
targets or silently infer the direction of a proxy response.

## Modules

| Module | Purpose |
|---|---|
| `scripts/preprocessing.py` | Age-ensemble consumption, perturbation sensitivity, standardization, safe interpolation, naming, bias documentation |
| `scripts/synthesis.py` | SCC, DCC, CPS, PAI, descriptive GAM/LOESS, Monte Carlo ensembles |
| `scripts/continuous_proxy.py` | Calibration, ragged-site synthesis, uncertainty, time-aware validation, paired comparison |
| `scripts/multi_proxy.py` | Proxy-level validation, target/direction harmonization, site-clustered synthesis, correlated error propagation, concordance, leave-one-proxy-out |
| `scripts/effect_size.py` | Paired log-ratio, Hedges' g, BCa/percentile intervals, RMSEP, LOOCV |
| `scripts/scenarios.py` | Paired validation, multi-site synthesis, and event-window association workflows |
| `scripts/validation.py` | Normality diagnostics, temporal/spatial dependence, moving-block bootstrap, sensitivity |
| `scripts/r_bridge.py` | Optional `metafor` backend; Python fallbacks are explicitly limited subsets |

Read only the relevant file in `references/` for specialized method details.

## Installation

```bash
python -m pip install -r requirements.txt
# Optional methods:
python -m pip install -r requirements-optional.txt
```

Core operations use NumPy, pandas, SciPy, and statsmodels. PyGAM,
scikit-learn, PySAL/ESDA, and R/metafor are optional and are detected only
when their methods are called. A missing optional backend must be reported in
the result rather than silently treated as an equivalent implementation.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover all public module imports and the highest-risk paths: safe
interpolation, missing-weight handling, age perturbation labelling, effect-size
intervals, ragged continuous sites, mixed-proxy site clustering, correlated
measurement error, leave-one-proxy-out sensitivity, event bootstrap callbacks,
and moving-block bootstrap.

## References

The `references/` directory separates workflow instructions from method notes.
The most important distinction is between an archive-specific chronology model,
an uncertainty ensemble consumed by this skill, and a simple sensitivity
perturbation. The skill does not implement REVEALS or claim that z-score
standardization is a biological equivalent of REVEALS.

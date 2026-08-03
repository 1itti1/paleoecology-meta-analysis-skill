# Python and optional backends

## Core dependencies

| Capability | Dependency | Policy |
|---|---|---|
| Arrays and missing-aware operations | NumPy | Required |
| Tables and long-form data | pandas | Required |
| Bootstrap and tests | SciPy | Required |
| LOWESS | statsmodels | Required by `synthesis.py` |

The code uses runtime validation rather than assuming a particular minor
version. Pin versions in the consuming project when reproducible publication
results are required.

## Optional dependencies

| Capability | Dependency | Behavior when absent |
|---|---|---|
| Descriptive GAM | PyGAM | Raise a clear method-specific error |
| Calibration CV | scikit-learn | Use LOOCV or install the package |
| Spatial Moran diagnostics | libpysal + esda | Use the labelled simplified fallback |
| Classical meta analysis/plots | R + metafor | Use only the documented Python subset |

Do not import optional modules at package import time. Test all modules, not
just the preprocessing path.

## Reproducibility

Pass `random_state` to stochastic functions, record the seed and package
versions, and check member-count sensitivity. The number 500 is a configurable
ensemble size, not a universal requirement.

## Method gaps

There is no generic Python implementation here for REVEALS, pollen source-area
models, archive-specific Bayesian age-depth modelling, compositional count
models, latent-variable proxy models, correlated multivariate random-effects
meta-analysis, or full spatial hierarchical synthesis. The bundled
`multi_proxy.py` module is a transparent common-target, site-clustered evidence
synthesis with optional standardized measurement-error correlation; it is not a
substitute for those models. Use a validated external backend or state the
limitation; do not replace a missing model with a z-score and call it
equivalent.

# Analysis contracts and provenance

## Route fields

Start from `assets/analysis_plan.json`. Its fields are deliberately compact:

| Field | Meaning |
|---|---|
| `schema_version` | `1.0` for this skill's contract; not EasyMeta's schema version |
| `product`, `target`, `estimand` | Requested product, common construct and quantity to estimate |
| `data_level` | `proxy_records`, `paired_observations`, or `study_effects` |
| `design` | `multi_proxy`, `time_series_stacking`, `paired_comparison`, `calibration`, `before_after`, `study_meta`, or `narrative` |
| `analysis_scale` | Explicit scale such as `standardized`, `identity`, `log`, or `fisher-z` |
| `age_unit` | Optional declared time unit/calendar convention; conflicting record declarations are rejected |
| `independent_cluster` | Field identifying independent replication, usually `site_id` or `study_id` |
| `dependence` | `independent`, `within_site`, `within_study`, `spatiotemporal`, or `unknown` |
| `sampling_covariance` | `not_applicable`, `not_needed_verified`, `available`, or `unavailable` |
| `pooling_eligible`, `pooling_reason` | Scientific pooling decision; an explicit reason is required when not pooling |

`runner_allowed` refers to the selected runner. `ordinary_meta_allowed` is true
only for a permitted study-level independent-effect model. Raw proxy records
cannot enter that model by changing a display label.

The input plan is a declaration to be checked, not evidence that independence
is true. The run also validates actual study/cluster IDs and effect scales.
Unknown dependence and explicitly modelled spatiotemporal processes produce a
specialist handoff. A supported site-cluster bootstrap remains available for
the common-target proxy index.

The data argument depends on the selected design:

| Design | Data argument |
|---|---|
| `multi_proxy` | List of proxy record dictionaries |
| `time_series_stacking` | Site-ID dictionary containing `ages`, `values` and optional errors/chronology |
| `paired_comparison` | Dictionary with `proxy_values`, `observed_values`, optional calibration arrays |
| `calibration` | Dictionary with `proxy_values`, `calibration_x`, `calibration_y` |
| `before_after` | Dictionary with `time_series`, `ages`, `event_year` |
| `study_meta` | List of staged effect rows described below |

Source and sample metadata remain part of the input hash. Put operational
choices such as grid, bootstrap count and an integer seed in keyword parameters.

## Invocation

```python
import json
import numpy as np
from scripts.analysis_contracts import run_analysis

with open('assets/analysis_plan.json', encoding='utf-8') as stream:
    plan = json.load(stream)

records = [
    dict(site_id='A', proxy_id='proxy_1', target='hydroclimate',
         age_unit='cal_year_BP', ages=[0., 100., 200.], values=[1., 2., 4.],
         direction='positive', source='synthetic demonstration'),
    dict(site_id='B', proxy_id='proxy_1', target='hydroclimate',
         age_unit='cal_year_BP', ages=[0., 100., 200.], values=[2., 3., 5.],
         direction='positive', source='synthetic demonstration'),
]
result = run_analysis(plan, records, time_grid=np.array([50., 150.]),
                      n_members=200, random_state=42)
```

The CLI checks a plan without fitting:

```bash
python scripts/analysis_contracts.py assets/analysis_plan.json
```

## Study effects

Use `design='study_meta'`, `data_level='study_effects'`,
`independent_cluster='study_id'`, `dependence='independent'`, and
`sampling_covariance='not_needed_verified'` only when justified.

Each input row retains:

- `study_id`, `report_id`, unique `effect_id`, and the declared cluster field;
- `source`, `target`, `measure`, `data_stage='analysis_effect'`;
- `analysis_scale`, finite `yi`, positive finite `vi`, and `sei` when reported;
- a common explicit `unit` for `identity`-scale effects.

`validate_effect_records()` preserves other extraction fields. It checks scale,
variance/SE consistency, IDs, common target and measure. Repeated independent
clusters or repeated studies are not sent to the ordinary model. Use a
specialist implementation with the appropriate sampling V and true-effect
structure; the bundled route returns that handoff without claiming a fit.

Common measures have an explicit analysis-scale check: ROM/RR/OR/HR/IRR use
`log`, MD/SMD use `identity`, and Fisher-transformed correlations use
`fisher-z`. Other measure families require a `calculation_method` in each row;
the contract records that choice without inventing an uncertainty conversion.

Keep `raw_extraction` as the original reported numbers. Derive a separate
analysis-effect table with the formula, orientation and uncertainty conversion
recorded. `lnRR` needs a meaningful ratio scale; a positive value alone is
insufficient. Align variance and standard error with the transformed estimate.

## Actual-run manifest

`run_analysis()` attaches `provenance` with content hashes, code-file hashes,
input/source identity, actual parameters, package versions, and UTC run time.
Pass `source_files=[...]` when files should be fingerprinted. In-memory source
metadata remains part of the input hash; it is not replaced by a guessed DOI.

All original values and source locators should remain in raw input records.
Keep the plan, route decision, result, and source verification together when
freezing an analysis. A reproducible hash does not itself verify extraction,
eligibility, publication integrity or ecological interpretation.

## Source and implementation boundary

The routing, staged extraction, independent-cluster and provenance concepts
were adapted from the locally reviewed EasyMeta workflow on 2026-10-22. This
skill is self-contained and does not copy EasyMeta's full systematic-review
pipeline or require a separate installation. PRISMA/ROSES reporting, study
selection and risk-of-bias appraisal belong in a review project when those
products are requested.

For dependent study effects, distinguish sampling V, random-effects structure
and coefficient inference: [metafor rma.mv documentation](https://wviechtb.github.io/metafor/reference/rma.mv.html).
For paired resampling, preserve shared indices:
[SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

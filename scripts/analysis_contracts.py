"""Analysis routing and raw-to-analysis contracts adapted to paleo records.

The concepts follow EasyMeta: declare the estimand, data level, scale and
independent cluster before fitting. This module has no dependency on an
installed EasyMeta skill and does not run a systematic-review pipeline.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    from .provenance import build_run_manifest
except ImportError:  # pragma: no cover
    from provenance import build_run_manifest


_ROUTES = {
    'multi_proxy': ('proxy_records', 'multi_proxy_synthesis'),
    'time_series_stacking': ('proxy_records', 'scenario2_multi_site_synthesis'),
    'paired_comparison': ('paired_observations', 'scenario1_proxy_validation'),
    'calibration': ('paired_observations', 'calibrate_continuous_proxy'),
    'before_after': ('proxy_records', 'before_after_test'),
    'study_meta': ('study_effects', 'rma_random_effects'),
    'narrative': ('any', None),
}


def route_analysis(plan: Mapping) -> dict:
    """Return the permitted runner, or a concrete non-pooling/handoff decision.

    See assets/analysis_plan.json for the complete compact contract. Route
    decisions are data-independent; the runner also checks the actual data.
    """
    if not isinstance(plan, Mapping):
        raise ValueError('analysis plan must be a mapping.')
    if plan.get('schema_version') != '1.0':
        raise ValueError('analysis plan schema_version must be 1.0.')
    required = ('product', 'target', 'estimand', 'data_level', 'design',
                'analysis_scale', 'independent_cluster', 'dependence', 'sampling_covariance')
    for field in required:
        if not isinstance(plan.get(field), str) or not plan[field].strip():
            raise ValueError(f'An explicit {field} is required in the analysis plan.')
    if not isinstance(plan.get('pooling_eligible'), bool):
        raise ValueError('pooling_eligible must be true or false, not a numeric/string flag.')
    design = plan['design']
    if design not in _ROUTES:
        raise ValueError(f'Unsupported design {design!r}; choose one of {sorted(_ROUTES)}.')
    level, runner = _ROUTES[design]
    if level != 'any' and plan['data_level'] != level:
        raise ValueError(f'{design} requires data_level={level!r}.')
    if plan['dependence'] not in {'independent', 'within_site', 'within_study', 'spatiotemporal', 'unknown'}:
        raise ValueError('Unknown dependence structure; use unknown if it has not been established.')
    if plan['sampling_covariance'] not in {'not_applicable', 'not_needed_verified', 'available', 'unavailable'}:
        raise ValueError('sampling_covariance has an unsupported status.')
    result = {
        'schema_version': '1.0', 'route': design, 'runner': runner,
        'runner_allowed': runner is not None, 'ordinary_meta_allowed': design == 'study_meta',
        'independent_cluster': plan['independent_cluster'],
        'estimand': plan['estimand'], 'analysis_scale': plan['analysis_scale'],
        'stop_reason': None, 'required_handoff': [],
    }
    if not plan['pooling_eligible'] or design == 'narrative':
        reason = plan.get('pooling_reason')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A pooling_reason is required for a non-pooling decision.')
        result.update(route='narrative', runner=None, runner_allowed=False,
                      ordinary_meta_allowed=False, stop_reason=reason)
    elif design == 'study_meta' and (
        plan['dependence'] != 'independent'
        or plan['sampling_covariance'] != 'not_needed_verified'
    ):
        result.update(
            route='specialist', runner=None, runner_allowed=False, ordinary_meta_allowed=False,
            stop_reason='The implemented ordinary Meta runner requires verified independent sampling errors and clusters.',
            required_handoff=['sampling_V', 'multilevel_or_multivariate_model', 'cluster_inference'],
        )
    elif plan['dependence'] in {'spatiotemporal', 'unknown'}:
        result.update(
            route='specialist', runner=None, runner_allowed=False, ordinary_meta_allowed=False,
            stop_reason='The declared dependence requires a specified observation/process model.',
            required_handoff=['dependence_model'],
        )
    return result


def validate_effect_records(records, *, analysis_scale: str, independent_cluster: str, target=None) -> dict:
    """Validate one row per derived study effect without inventing missing SEs.

    All original fields are retained. Repeated independent-cluster IDs trigger
    a specialist handoff at run time; this validator reports their presence.
    """
    rows = [deepcopy(dict(record)) for record in records]
    if len(rows) < 2:
        raise ValueError('Study Meta analysis needs at least two effects.')
    seen, clusters = set(), []
    for index, row in enumerate(rows):
        for field in ('study_id', 'report_id', 'effect_id', independent_cluster, 'source', 'measure', 'target'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Effect {index} requires a non-empty {field}.')
        if row['effect_id'] in seen:
            raise ValueError(f'Duplicate effect_id: {row["effect_id"]}')
        seen.add(row['effect_id'])
        if row.get('data_stage') != 'analysis_effect':
            raise ValueError('Keep raw_extraction separate; the model needs data_stage=analysis_effect.')
        if row.get('analysis_scale') != analysis_scale:
            raise ValueError('Effects and variances must use the plan analysis_scale.')
        if target is not None and row['target'] != target:
            raise ValueError('Effect targets must match the plan target.')
        yi, vi = float(row['yi']), float(row['vi'])
        if not np.isfinite(yi) or not np.isfinite(vi) or vi <= 0:
            raise ValueError('Each yi must be finite and vi must be finite and positive.')
        if row.get('sei') is not None:
            sei = float(row['sei'])
            if not np.isfinite(sei) or sei <= 0 or not np.isclose(sei ** 2, vi, rtol=1e-6, atol=1e-12):
                raise ValueError('sei must be positive and vi must equal sei squared on the analysis scale.')
        row.update(yi=yi, vi=vi)
        clusters.append(row[independent_cluster])
    if len({row['measure'] for row in rows}) != 1 or len({row['target'] for row in rows}) != 1:
        raise ValueError('Different measures or outcome targets must not be pooled in one model.')
    measure = rows[0]['measure'].upper()
    scales = {
        'MD': {'identity'}, 'SMD': {'identity'}, 'HEDGES_G': {'identity'},
        'ROM': {'log'}, 'LNRR': {'log'}, 'RR': {'log'}, 'OR': {'log'},
        'HR': {'log'}, 'IRR': {'log'}, 'RD': {'identity'}, 'ZCOR': {'fisher-z'},
        'FISHER_Z': {'fisher-z'},
    }
    if measure in scales and analysis_scale not in scales[measure]:
        raise ValueError(f'{measure} requires analysis_scale in {sorted(scales[measure])}; transform estimates and uncertainty together.')
    if measure not in scales and any(not row.get('calculation_method') for row in rows):
        raise ValueError('Custom effect measures require an explicit calculation_method and declared analysis_scale.')
    if analysis_scale == 'identity':
        units = {row.get('unit') for row in rows}
        if None in units or len(units) != 1:
            raise ValueError('Natural-scale effects require an explicit common unit.')
    return {
        'records': rows, 'yi': np.array([row['yi'] for row in rows]),
        'vi': np.array([row['vi'] for row in rows]), 'cluster_ids': clusters,
        'n_effects': len(rows), 'n_independent_clusters': len(set(clusters)),
        'has_dependent_effects': len(set(clusters)) != len(rows),
    }


def _check_time_units(data, plan):
    if plan['design'] == 'multi_proxy':
        records = list(data)
    elif plan['design'] == 'time_series_stacking':
        records = list(data.values())
    else:
        records = [data]
    declared = {row['age_unit'] for row in records if row.get('age_unit') is not None}
    if plan.get('age_unit'):
        declared.add(plan['age_unit'])
    if len(declared) > 1:
        raise ValueError('Convert age units/calendar conventions explicitly before time alignment.')


def run_analysis(plan: Mapping, data, *, source_files=(), **parameters) -> dict:
    """Run an allowed route and return its decision plus a SHA-256 run manifest.

    For narrative/specialist routes, return the decision without fitting. This
    makes an explicit non-pooling decision a usable output rather than a silent
    downgrade into an unrelated model.
    """
    decision = route_analysis(plan)
    raw = deepcopy(data)
    if not decision['runner_allowed']:
        result = {'status': 'not_fitted', 'analysis_route': decision}
        result['provenance'] = build_run_manifest(raw, {'plan': dict(plan), **parameters}, result, source_files=source_files)
        return result
    design = plan['design']
    try:
        from . import scenarios, continuous_proxy, multi_proxy, r_bridge
    except ImportError:  # pragma: no cover
        import scenarios, continuous_proxy, multi_proxy, r_bridge
    if design == 'study_meta':
        effects = validate_effect_records(data, analysis_scale=plan['analysis_scale'], independent_cluster=plan['independent_cluster'], target=plan['target'])
        if effects['has_dependent_effects'] or len({r['study_id'] for r in effects['records']}) < len(effects['records']):
            raise ValueError('Repeated study/cluster effects require sampling V and a multilevel/multivariate model; ordinary pooling was not run.')
        result = r_bridge.rma_random_effects(effects['yi'], effects['vi'], **parameters)
        result.update(effect_records=effects['records'], n_effects=effects['n_effects'],
                      n_independent_clusters=effects['n_independent_clusters'], analysis_scale=plan['analysis_scale'])
    elif design == 'multi_proxy':
        _check_time_units(data, plan)
        if plan['independent_cluster'] != 'site_id':
            raise ValueError('The bundled multi-proxy bootstrap implements site_id clusters only.')
        raw_scale = [record.get('standardization', parameters.get('default_standardization', 'zscore')) == 'none' for record in data]
        if (plan['analysis_scale'] == 'standardized' and any(raw_scale)
                or plan['analysis_scale'] == 'identity' and not all(raw_scale)):
            raise ValueError('The plan analysis_scale must match the actual proxy transformations.')
        result = multi_proxy.multi_proxy_synthesis(data, target=plan['target'], **parameters)
    elif design == 'time_series_stacking':
        _check_time_units(data, plan)
        if plan['independent_cluster'] != 'site_id':
            raise ValueError('The bundled site synthesis implements site_id replication only.')
        result = scenarios.scenario2_multi_site_synthesis(data, **parameters)
    elif design == 'paired_comparison':
        numeric = {key: value for key, value in data.items()
                   if key in {'proxy_values', 'observed_values', 'proxy_type', 'calibration_x', 'calibration_y'}}
        result = scenarios.scenario1_proxy_validation(**numeric, **parameters)
    elif design == 'calibration':
        numeric = {key: value for key, value in data.items()
                   if key in {'proxy_values', 'calibration_x', 'calibration_y'}}
        result = continuous_proxy.calibrate_continuous_proxy(**numeric, **parameters)
    else:
        _check_time_units(data, plan)
        event_data = {key: value for key, value in data.items() if key in {'time_series', 'ages', 'event_year'}}
        result = scenarios.before_after_test(**event_data, **parameters)
    result['status'] = 'failed' if 'error' in result else 'fitted'
    result['analysis_route'] = decision
    result['provenance'] = build_run_manifest(raw, {'plan': dict(plan), **parameters}, result, source_files=source_files)
    return result


def main():
    parser = argparse.ArgumentParser(description='Validate and route a paleoecology analysis plan.')
    parser.add_argument('plan', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(route_analysis(json.loads(arguments.plan.read_text(encoding='utf-8-sig'))), indent=2))


if __name__ == '__main__':
    main()

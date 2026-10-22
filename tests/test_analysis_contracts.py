import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts.analysis_contracts import route_analysis, run_analysis, validate_effect_records
from scripts.provenance import content_hash


def plan(design='study_meta'):
    return {
        'schema_version': '1.0', 'product': 'quantitative_synthesis',
        'target': 'temperature', 'estimand': 'mean_anomaly',
        'data_level': 'study_effects', 'design': design, 'analysis_scale': 'identity',
        'independent_cluster': 'study_id', 'dependence': 'independent',
        'sampling_covariance': 'not_needed_verified', 'pooling_eligible': True,
    }


def effects():
    return [
        dict(study_id=f's{i}', report_id=f'r{i}', effect_id=f'e{i}', source='synthetic table',
             target='temperature', measure='MD', unit='degC', data_stage='analysis_effect',
             analysis_scale='identity', yi=yi, vi=0.01, sei=0.1)
        for i, yi in enumerate([0.1, 0.2, 0.3])
    ]


class AnalysisContractTests(unittest.TestCase):
    def test_specialist_dependence_never_reaches_ordinary_runner(self):
        specification = plan()
        specification.update(dependence='within_study', sampling_covariance='unavailable')
        with patch('scripts.r_bridge.rma_random_effects') as runner:
            result = run_analysis(specification, effects())
        runner.assert_not_called()
        self.assertEqual(result['status'], 'not_fitted')
        self.assertFalse(result['analysis_route']['ordinary_meta_allowed'])

    def test_declared_independence_is_checked_against_actual_ids(self):
        data = effects()
        data[1]['study_id'] = data[0]['study_id']
        with patch('scripts.r_bridge.rma_random_effects') as runner:
            with self.assertRaisesRegex(ValueError, 'Repeated study'):
                run_analysis(plan(), data)
        runner.assert_not_called()

    def test_scale_and_variance_checks_reject_mismatched_effects(self):
        data = effects()
        data[0]['sei'] = 0.01
        with self.assertRaisesRegex(ValueError, 'squared'):
            validate_effect_records(data, analysis_scale='identity', independent_cluster='study_id')
        data = effects()
        data[0]['analysis_scale'] = 'log'
        with self.assertRaisesRegex(ValueError, 'analysis_scale'):
            validate_effect_records(data, analysis_scale='identity', independent_cluster='study_id')
        data = effects()
        data[0]['data_stage'] = 'raw_extraction'
        with self.assertRaisesRegex(ValueError, 'raw_extraction'):
            validate_effect_records(data, analysis_scale='identity', independent_cluster='study_id')

    def test_common_scale_does_not_permit_mixed_outcomes(self):
        data = effects()
        data[1]['target'] = 'precipitation'
        with self.assertRaisesRegex(ValueError, 'targets'):
            validate_effect_records(data, analysis_scale='identity', independent_cluster='study_id', target='temperature')

    def test_ratio_effects_cannot_be_accepted_on_natural_scale(self):
        data = effects()
        for row in data:
            row['measure'] = 'ROM'
        with self.assertRaisesRegex(ValueError, 'requires analysis_scale'):
            validate_effect_records(data, analysis_scale='identity', independent_cluster='study_id')

    def test_paired_source_metadata_is_hashed_without_entering_model_arguments(self):
        specification = plan('paired_comparison')
        specification.update(data_level='paired_observations', independent_cluster='sample_id')
        raw = {'proxy_values': np.arange(20.), 'observed_values': np.arange(20.) + 1,
               'source': 'synthetic validation table', 'sample_id': list(range(20))}
        result = run_analysis(specification, raw, n_boot=200, random_state=5)
        self.assertAlmostEqual(result['rmsep']['rmsep'], 1.)
        self.assertEqual(result['provenance']['input_sha256'], content_hash(raw))

    def test_narrative_output_is_a_recorded_non_pooling_decision(self):
        specification = plan()
        specification.update(pooling_eligible=False, pooling_reason='No comparable reference periods')
        result = run_analysis(specification, effects())
        self.assertEqual(result['status'], 'not_fitted')
        self.assertEqual(result['analysis_route']['stop_reason'], 'No comparable reference periods')

    def test_proxy_route_and_manifest_describe_actual_inputs(self):
        specification = plan('multi_proxy')
        specification.update(data_level='proxy_records', analysis_scale='standardized',
                             independent_cluster='site_id', dependence='within_site',
                             sampling_covariance='not_applicable', age_unit='cal_year_BP')
        records = [dict(site_id='A', proxy_id='p', target='temperature', age_unit='cal_year_BP',
                        ages=[0.0, 1.0, 2.0], values=[1.0, 2.0, 3.0], source='synthetic data')]
        result = run_analysis(specification, records, time_grid=np.array([0.5, 1.5]),
                              bootstrap_sites=False, propagate_age=False, propagate_measurement=False)
        self.assertEqual(result['status'], 'fitted')
        self.assertEqual(result['provenance']['input_sha256'], content_hash(records))
        without_manifest = {key: value for key, value in result.items() if key != 'provenance'}
        self.assertEqual(result['provenance']['output_sha256'], content_hash(without_manifest))
        changed = copy.deepcopy(records)
        changed[0]['values'][0] = 99.0
        self.assertNotEqual(content_hash(records), content_hash(changed))
        records[0]['age_unit'] = 'year_CE'
        with self.assertRaisesRegex(ValueError, 'age units'):
            run_analysis(specification, records, time_grid=np.array([0.5]))

    def test_proxy_table_cannot_route_as_study_effects(self):
        specification = plan()
        specification['data_level'] = 'proxy_records'
        with self.assertRaises(ValueError):
            route_analysis(specification)


if __name__ == '__main__':
    unittest.main()

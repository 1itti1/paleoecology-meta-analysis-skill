import io
import unittest

import numpy as np
import pandas as pd

from scripts.continuous_proxy import propagate_continuous_uncertainty
from scripts.preprocessing import consume_bacon_ages
from scripts.scenarios import (
    before_after_test, build_indicators, multi_window_robustness,
    scenario1_proxy_validation, scenario2_multi_site_synthesis, select_scenario,
)
from scripts.synthesis import pai_composite


class WorkflowRegressionTests(unittest.TestCase):
    def test_calibration_is_applied_before_prediction_validation(self):
        x = np.linspace(1, 10, 20)
        result = scenario1_proxy_validation(x, 2 * x + 10, calibration_x=x,
                                             calibration_y=2 * x + 10, n_boot=200, random_state=5)
        self.assertLess(result['rmsep']['rmsep'], 1e-10)
        self.assertEqual(result['effect_type'], 'mean_difference')
        self.assertEqual(select_scenario('calibration')['scenario'], 4)

    def test_legacy_site_error_enters_the_ensemble(self):
        ages = np.arange(4.0)
        records = {'A': {'ages': ages, 'values': ages + 1, 'proxy_error': 1.0}}
        result = scenario2_multi_site_synthesis(records, ages, proxy_type='continuous',
                                                synthesis_method='weighted_mean', n_members=200, random_state=9)
        self.assertEqual(result['n_members'], 200)
        self.assertIn('measurement', result['uncertainty_layers'])
        self.assertTrue(np.all(result['uncertainty_band']['upper'] > result['uncertainty_band']['lower']))

    def test_cps_obeys_site_weights(self):
        ages = np.arange(4.0)
        records = {'A': {'ages': ages, 'values': np.zeros(4), 'weight': 1.0},
                   'B': {'ages': ages, 'values': np.full(4, 10.0), 'weight': 9.0}}
        result = scenario2_multi_site_synthesis(records, ages, proxy_type='continuous', synthesis_method='cps')
        np.testing.assert_allclose(result['composite'], 9.0)
        records['A']['weight'], records['B']['weight'] = 9.0, 1.0
        result = scenario2_multi_site_synthesis(records, ages, proxy_type='continuous', synthesis_method='cps')
        np.testing.assert_allclose(result['composite'], 1.0)

    def test_pai_zero_change_is_neutral_and_first_bin_undefined(self):
        result = pai_composite(np.array([[1., 1., 1.], [2., 2., 2.]]))
        self.assertTrue(np.isnan(result['composite'][0]))
        np.testing.assert_allclose(result['composite'][1:], [0., 0.])

    def test_missing_taxa_measurements_are_not_absences(self):
        result = build_indicators(pd.DataFrame({'a': [np.nan, 1.], 'b': [np.nan, 2.]}), {'trees': ['a', 'b']})
        self.assertTrue(np.isnan(result['indicators']['trees'][0]))
        self.assertEqual(result['indicators']['trees'][1], 3.)

    def test_csv_and_whitespace_chronology_have_same_values(self):
        csv = consume_bacon_ages(io.StringIO('depth,m1,m2\n0,100,110\n1,200,210\n'))
        space = consume_bacon_ages(io.StringIO('depth m1 m2\n0 100 110\n1 200 210\n'))
        np.testing.assert_array_equal(csv['age_ensembles'], [[100, 200], [110, 210]])
        np.testing.assert_array_equal(csv['age_ensembles'], space['age_ensembles'])

    def test_fixed_sample_ensemble_keeps_endpoint_support(self):
        ages = np.arange(20.)
        result = propagate_continuous_uncertainty(ages, np.array([ages]), 0., np.array([0., 10., 19.]), n_members=100, random_state=42)
        np.testing.assert_allclose(result['uncertainty_band']['median'], [0., 10., 19.])
        np.testing.assert_array_equal(result['uncertainty_band']['valid_member_count'], [100, 100, 100])
        self.assertNotIn('iid_pairs_sampling', result['layers'])

    def test_iid_pairs_bootstrap_counts_degenerate_members_without_crashing(self):
        result = propagate_continuous_uncertainty(np.array([1., 2.]), np.array([[0., 1.]]), 0.,
                                                  np.array([0., .5, 1.]), n_members=100, random_state=1, sampling_method='iid')
        self.assertGreater(result['skipped_members'], 0)
        self.assertTrue(np.all(np.isfinite(result['uncertainty_band']['median'])))

    def test_bp_event_comparison_uses_older_ages_as_before(self):
        ages = np.arange(90., 111.)
        result = before_after_test(ages, ages, 100., window=10, n_boot=200, random_state=2, time_direction='BP')
        self.assertAlmostEqual(result['difference'], -10.5)
        self.assertEqual(result['method'], 'circular_block_percentile')
        self.assertIsNone(result['p_value'])

    def test_overlapping_windows_do_not_vote_by_p_values(self):
        ages = np.arange(100.)
        result = multi_window_robustness(ages, ages, 50., windows=[10, 20], n_boot=200, random_state=3)
        self.assertTrue(result['direction_consistent'])
        self.assertIsNone(result['robust'])
        for row in result['results'].values():
            self.assertAlmostEqual(row['confidence_level'], .975)


if __name__ == '__main__':
    unittest.main()

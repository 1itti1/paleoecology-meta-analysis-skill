"""Small, dependency-light regression tests for the public workflow."""

import unittest

import numpy as np

from scripts.continuous_proxy import composite_continuous_proxy, proxy_comparison
from scripts.effect_size import effect_size_bca, hedges_d
from scripts.preprocessing import age_ensemble_from_errors, resample_to_grid
from scripts.scenarios import scenario2_multi_site_synthesis, scenario3_human_attribution
from scripts.synthesis import scc_composite
from scripts.validation import block_bootstrap


class CoreWorkflowTests(unittest.TestCase):
    def test_age_ensemble_is_explicitly_not_a_model(self):
        result = age_ensemble_from_errors(
            np.arange(4), np.array([0.0, 100.0, 200.0, 300.0]),
            np.ones(4), n_members=8, random_state=7,
        )
        self.assertEqual(result['age_ensembles'].shape, (8, 4))
        self.assertEqual(result['method'], 'age_perturbation')

    def test_interpolation_does_not_extrapolate(self):
        result = resample_to_grid(
            np.array([0.0, 100.0, 200.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, 50.0, 250.0]),
        )
        self.assertTrue(np.isnan(result['resampled'][0]))
        self.assertTrue(np.isnan(result['resampled'][-1]))
        self.assertAlmostEqual(result['resampled'][1], 1.5)

    def test_weighted_composite_renormalizes_missing_sites(self):
        result = scc_composite(np.array([[1.0, np.nan], [3.0, 5.0]]))
        np.testing.assert_allclose(result['composite'], [2.0, 5.0])

    def test_effect_size_intervals_are_finite(self):
        treatment = np.linspace(1.0, 3.0, 24)
        control = np.linspace(0.8, 2.8, 24)
        result = effect_size_bca(
            treatment, control, effect_type='lnrr', n_boot=200, random_state=1,
        )
        self.assertTrue(np.isfinite(result['ci_lower']))
        self.assertTrue(np.isfinite(result['ci_upper']))
        self.assertAlmostEqual(hedges_d(np.ones(3), np.ones(3))['d'], 0.0)

    def test_ragged_continuous_sites(self):
        result = composite_continuous_proxy(
            [np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0, 8.0])],
            [np.array([0.0, 100.0, 200.0]), np.array([0.0, 100.0, 200.0, 300.0])],
            np.array([0.0, 50.0, 150.0, 250.0]),
        )
        self.assertEqual(result['composite'].shape, (4,))

    def test_scenario_bootstrap_callback(self):
        result = scenario3_human_attribution(
            np.arange(24.0), np.arange(24.0) + 1.0,
            event_year=12.0, n_boot=200, random_state=3,
        )
        ci = result['results'][0]['ci']
        self.assertTrue(np.all(np.isfinite(ci)))

    def test_proxy_comparison_reports_association_and_error(self):
        result = proxy_comparison(
            np.arange(24.0), np.arange(24.0) + 0.1,
            n_boot=200, random_state=4, agreement_threshold=0.2,
        )
        self.assertTrue(result['agreement'])
        self.assertTrue(np.isfinite(result['rmse']))

    def test_scenario_two_accepts_ragged_site_records(self):
        site_data = {
            'a': {'ages': np.array([0.0, 100.0, 200.0]), 'values': np.array([1.0, 2.0, 3.0])},
            'b': {'ages': np.array([0.0, 100.0, 200.0, 300.0]), 'values': np.array([3.0, 2.0, 1.0, 0.0])},
        }
        result = scenario2_multi_site_synthesis(
            site_data, np.array([0.0, 50.0, 150.0, 250.0]),
            proxy_type='continuous', synthesis_method='weighted_mean',
        )
        self.assertEqual(result['composite'].shape, (4,))

    def test_block_bootstrap_has_explicit_method(self):
        result = block_bootstrap(
            np.arange(40.0), np.mean, block_length=4,
            n_resamples=200, random_state=5,
        )
        self.assertEqual(result['method'], 'moving_block_percentile')


if __name__ == '__main__':
    unittest.main()

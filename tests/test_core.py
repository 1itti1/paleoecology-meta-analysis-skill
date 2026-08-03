"""Small, dependency-light regression tests for the public workflow."""

import unittest

import numpy as np

from scripts.continuous_proxy import composite_continuous_proxy, proxy_comparison
from scripts.effect_size import effect_size_bca, hedges_d
from scripts.multi_proxy import (
    leave_one_proxy_out,
    multi_proxy_synthesis,
    prepare_proxy_records,
)
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

    def test_multi_proxy_synthesis_clusters_proxies_by_site(self):
        records = [
            {
                'site_id': 'A',
                'proxy_id': 'pollen',
                'proxy_type': 'taxa',
                'target': 'vegetation_change',
                'ages': np.array([0.0, 100.0, 200.0, 300.0]),
                'values': np.array([1.0, 2.0, 3.0, 4.0]),
                'age_ensembles': np.array([
                    [0.0, 100.0, 200.0, 300.0],
                    [1.0, 101.0, 201.0, 301.0],
                ]),
                'measurement_error': 0.05,
                'site_weight': 2.0,
            },
            {
                'site_id': 'A',
                'proxy_id': 'charcoal',
                'proxy_type': 'continuous',
                'target': 'vegetation_change',
                'direction': 'negative',
                'ages': np.array([0.0, 150.0, 300.0]),
                'values': np.array([4.0, 3.0, 1.0]),
                'measurement_error': 0.10,
                'site_weight': 2.0,
            },
            {
                'site_id': 'B',
                'proxy_id': 'pollen',
                'proxy_type': 'taxa',
                'target': 'vegetation_change',
                'ages': np.array([0.0, 120.0, 240.0, 360.0]),
                'values': np.array([0.5, 1.5, 2.5, 3.5]),
                'measurement_error': 0.05,
            },
            {
                'site_id': 'B',
                'proxy_id': 'charcoal',
                'proxy_type': 'continuous',
                'target': 'vegetation_change',
                'direction': 'negative',
                'ages': np.array([0.0, 180.0, 360.0]),
                'values': np.array([3.5, 2.5, 1.0]),
                'measurement_error': 0.10,
            },
        ]
        grid = np.array([-20.0, 0.0, 60.0, 180.0, 320.0, 400.0])
        result = multi_proxy_synthesis(
            records,
            grid,
            weighting='precision',
            measurement_correlation={('pollen', 'charcoal'): 0.25},
            n_members=40,
            random_state=7,
        )
        self.assertEqual(result['n_sites'], 2)
        self.assertEqual(result['n_proxies'], 2)
        self.assertEqual(result['ensembles'].shape, (40, len(grid)))
        self.assertTrue(np.isnan(result['composite'][0]))
        self.assertTrue(np.isnan(result['composite'][-1]))
        self.assertEqual(result['bootstrap_unit'], 'site_cluster')
        self.assertIn('pollen__vs__charcoal', result['proxy_concordance']['pairs'])
        self.assertEqual(result['effective_proxy_count'][2], 2)

    def test_multi_proxy_rejects_mixed_targets(self):
        base = {
            'site_id': 'A',
            'ages': np.array([0.0, 1.0, 2.0]),
            'values': np.array([1.0, 2.0, 3.0]),
        }
        records = [
            dict(base, proxy_id='pollen', target='climate'),
            dict(base, site_id='B', proxy_id='charcoal', target='human_activity'),
        ]
        with self.assertRaises(ValueError):
            prepare_proxy_records(records)
        no_target = [
            dict(base, proxy_id='pollen'),
            dict(base, site_id='B', proxy_id='charcoal'),
        ]
        with self.assertRaises(ValueError):
            multi_proxy_synthesis(no_target, np.array([0.0, 1.0, 2.0]), bootstrap_sites=False)

    def test_leave_one_proxy_out_is_deterministic(self):
        records = [
            {
                'site_id': 'A', 'proxy_id': 'pollen', 'target': 'shared',
                'ages': np.array([0.0, 100.0, 200.0]),
                'values': np.array([0.0, 1.0, 2.0]),
            },
            {
                'site_id': 'A', 'proxy_id': 'isotope', 'target': 'shared',
                'ages': np.array([0.0, 100.0, 200.0]),
                'values': np.array([0.0, 2.0, 4.0]),
            },
            {
                'site_id': 'B', 'proxy_id': 'pollen', 'target': 'shared',
                'ages': np.array([0.0, 100.0, 200.0]),
                'values': np.array([1.0, 2.0, 3.0]),
            },
            {
                'site_id': 'B', 'proxy_id': 'isotope', 'target': 'shared',
                'ages': np.array([0.0, 100.0, 200.0]),
                'values': np.array([2.0, 3.0, 4.0]),
            },
        ]
        result = leave_one_proxy_out(
            records,
            np.array([0.0, 100.0, 200.0]),
            measurement_correlation={('pollen', 'isotope'): 0.25},
        )
        self.assertFalse(result['stochastic'])
        self.assertEqual(set(result['by_excluded_proxy']), {'pollen', 'isotope'})
        for summary in result['by_excluded_proxy'].values():
            self.assertTrue(np.isfinite(summary['rmse']))

    def test_block_bootstrap_has_explicit_method(self):
        result = block_bootstrap(
            np.arange(40.0), np.mean, block_length=4,
            n_resamples=200, random_state=5,
        )
        self.assertEqual(result['method'], 'moving_block_percentile')


if __name__ == '__main__':
    unittest.main()

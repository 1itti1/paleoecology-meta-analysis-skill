"""Numerical contracts for effect estimands and independent-study models.

Run with ``python -B -m unittest tests.test_effect_models -v``.  R comparisons
are optional and skip unless Rscript, jsonlite and metafor are available.
All observations here are synthetic regression fixtures, not research data.
"""

import math
import os
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy import stats

from scripts import r_bridge
from scripts.effect_size import (
    effect_size_bca,
    hedges_d,
    log_response_ratio,
    log_response_ratio_variance,
)


NO_R = {'success': False, 'result': None, 'error': 'deliberately unavailable'}


def manual_dl(yi, vi, kh):
    """Independent formula-level reference, including nonuniform weights."""
    k = len(yi)
    fixed_weights = [1 / value for value in vi]
    fixed_mean = sum(w * y for w, y in zip(fixed_weights, yi)) / sum(fixed_weights)
    q_fixed = sum(w * (y - fixed_mean) ** 2 for w, y in zip(fixed_weights, yi))
    c = sum(fixed_weights) - sum(w * w for w in fixed_weights) / sum(fixed_weights)
    tau2 = max(0, (q_fixed - k + 1) / c)
    weights = [1 / (value + tau2) for value in vi]
    mean = sum(w * y for w, y in zip(weights, yi)) / sum(weights)
    scale = sum(w * (y - mean) ** 2 for w, y in zip(weights, yi)) / (k - 1) if kh else 1
    se = math.sqrt(scale / sum(weights))
    critical = stats.t.ppf(0.975, k - 1) if kh else stats.norm.ppf(0.975)
    p = 2 * (stats.t.sf(abs(mean / se), k - 1) if kh else stats.norm.sf(abs(mean / se)))
    return mean, se, mean - critical * se, mean + critical * se, p, tau2


class EffectEstimandTests(unittest.TestCase):
    def test_opposite_directions_are_distinct_estimands(self):
        treatment = np.tile([1., 100.], 50)
        control = np.full(100, 20.)
        rom = effect_size_bca(treatment, control, n_boot=300, random_state=11)
        paired = effect_size_bca(
            treatment, control, effect_type='paired_log_ratio', design='paired',
            n_boot=300, random_state=11,
        )
        self.assertAlmostEqual(rom['effect_size'], math.log(50.5 / 20), places=13)
        self.assertAlmostEqual(paired['effect_size'], math.log(0.5), places=13)
        self.assertGreater(rom['effect_size'], 0)
        self.assertLess(paired['effect_size'], 0)
        self.assertEqual(rom['estimand'], 'log(mean(T)/mean(C))')
        self.assertEqual(paired['estimand'], 'mean(log(T/C))')
        self.assertEqual(paired['resampling_unit'], 'pair')
        np.testing.assert_allclose(
            log_response_ratio(treatment, control), np.log(treatment / control), atol=1e-14,
        )

    def test_independent_groups_can_have_different_lengths(self):
        treatment = np.linspace(1., 4., 24)
        control = np.linspace(.8, 3., 31)
        actual = effect_size_bca(treatment, control, design='independent', n_boot=350, random_state=17)

        def statistic(t, c, axis=-1):
            return np.log(np.mean(t, axis=axis) / np.mean(c, axis=axis))

        reference = stats.bootstrap(
            (treatment, control), statistic, method='BCa', paired=False,
            n_resamples=350, random_state=np.random.default_rng(17),
        )
        np.testing.assert_allclose(
            [actual['ci_lower'], actual['ci_upper']], reference.confidence_interval, atol=1e-13,
        )
        self.assertEqual((actual['n_t'], actual['n_c']), (24, 31))

    def test_paired_log_ratio_matches_single_sample_bootstrap(self):
        control = np.linspace(2., 20., 32)
        treatment = control * np.exp(.2 + .08 * np.sin(np.arange(32)))
        actual = effect_size_bca(
            treatment, control, effect_type='paired_log_ratio', design='paired',
            n_boot=500, random_state=19,
        )
        reference = stats.bootstrap(
            (np.log(treatment / control),), np.mean, method='BCa',
            n_resamples=500, random_state=np.random.default_rng(19),
        )
        np.testing.assert_allclose(
            [actual['ci_lower'], actual['ci_upper']], reference.confidence_interval, atol=1e-13,
        )

    def test_paired_rom_resamples_the_same_indices(self):
        control = np.linspace(2., 40., 32)
        treatment = control + 2 + .2 * np.sin(np.arange(32))
        paired = effect_size_bca(treatment, control, design='paired', n_boot=500, random_state=21)

        def paired_statistic(index, axis=-1):
            return np.log(np.mean(treatment[index], axis=axis) / np.mean(control[index], axis=axis))

        reference = stats.bootstrap(
            (np.arange(len(control)),), paired_statistic, method='BCa',
            n_resamples=500, random_state=np.random.default_rng(21),
        )
        np.testing.assert_allclose(
            [paired['ci_lower'], paired['ci_upper']], reference.confidence_interval, atol=1e-13,
        )
        independent = effect_size_bca(treatment, control, design='independent', n_boot=500, random_state=21)
        self.assertLess(
            paired['ci_upper'] - paired['ci_lower'],
            (independent['ci_upper'] - independent['ci_lower']) / 3,
        )

    def test_design_and_estimand_errors_are_explicit(self):
        cases = [
            ({'effect_type': 'paired_log_ratio'}, 'design'),
            ({'effect_type': 'd', 'design': 'paired'}, 'SMD'),
            ({'design': 'BACI'}, '专用模型'),
            ({'effect_type': 'unknown'}, 'effect_type'),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                effect_size_bca(np.arange(24.) + 1, np.arange(24.) + 2, **kwargs)
        with self.assertRaisesRegex(ValueError, '长度'):
            effect_size_bca(np.ones(24), np.ones(25), design='paired')
        with self.assertRaises(ValueError):
            effect_size_bca(np.ones(24), np.r_[0., np.ones(23)])
        with self.assertRaisesRegex(ValueError, '整数'):
            effect_size_bca(np.ones(24), np.ones(24), n_boot=100.5)

    def test_degenerate_bca_retains_finite_percentile_result_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = effect_size_bca(np.ones(24), np.ones(24), design='paired', n_boot=100, random_state=3)
        self.assertEqual(result['method'], 'percentile_fallback')
        self.assertEqual((result['ci_lower'], result['ci_upper']), (0., 0.))
        self.assertFalse(result['significant'])
        self.assertTrue(caught)
        self.assertTrue(result['warnings'])

    def test_rom_variance_independent_and_paired_hand_calculation(self):
        t = np.array([8., 10., 12.])
        c = np.array([9., 11., 13.])
        independent = 4 / (3 * 10 ** 2) + 4 / (3 * 11 ** 2)
        paired = independent - 8 / (3 * 10 * 11)
        self.assertAlmostEqual(log_response_ratio_variance(t, c, 2, 2, 3, 3), independent, places=15)
        self.assertAlmostEqual(
            log_response_ratio_variance(t, c, 2, 2, 3, 3, design='paired', paired_covariance=4),
            paired, places=15,
        )
        self.assertGreater(independent / paired, 200)
        self.assertAlmostEqual(log_response_ratio_variance(10., 11., 2, 2, 3, 3), independent)

    def test_rom_variance_rejects_missing_or_incompatible_covariance(self):
        cases = [
            ({'design': 'paired'}, 'paired_covariance'),
            ({'paired_covariance': 0}, 'independent|独立组'),
            ({'design': 'paired', 'paired_covariance': 5}, 'Cov'),
            ({'design': 'paired', 'paired_covariance': np.nan}, '有限'),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                log_response_ratio_variance([10.], [11.], 2, 2, 3, 3, **kwargs)
        with self.assertRaisesRegex(ValueError, '样本量'):
            log_response_ratio_variance([10.], [11.], 2, 2, 3, 4, design='paired', paired_covariance=0)
        for invalid_n in (0, 2.5, np.inf, True):
            with self.subTest(n=invalid_n), self.assertRaises(ValueError):
                log_response_ratio_variance([10.], [11.], 2, 2, invalid_n, 3)

    def test_hedges_g_independent_formula_and_paired_rejection(self):
        t, c = np.array([1., 2., 3., 4.]), np.array([2., 4., 5.])
        pooled = math.sqrt((3 * np.var(t, ddof=1) + 2 * np.var(c, ddof=1)) / 5)
        correction = 1 - 3 / 19
        expected = correction * (np.mean(t) - np.mean(c)) / pooled
        result = hedges_d(t, c)
        self.assertAlmostEqual(result['d'], expected)
        self.assertAlmostEqual(result['variance'], 7 / 12 + expected ** 2 / 14)
        self.assertEqual(result['design'], 'independent')
        self.assertAlmostEqual(hedges_d(np.ones(3), np.ones(3))['d'], 0)
        with self.assertRaisesRegex(ValueError, '配对 SMD'):
            hedges_d(t, t + 1, design='paired')

    def test_hedges_bca_uses_actual_jackknife_sample_sizes(self):
        t = np.linspace(1., 5., 24)
        c = np.linspace(.5, 4., 30)

        def independent_g(t, c, axis=-1):
            nt, nc = t.shape[axis], c.shape[axis]
            df = nt + nc - 2
            pooled = np.sqrt(((nt - 1) * np.var(t, axis=axis, ddof=1)
                              + (nc - 1) * np.var(c, axis=axis, ddof=1)) / df)
            return (1 - 3 / (4 * df - 1)) * (np.mean(t, axis=axis) - np.mean(c, axis=axis)) / pooled

        reference = stats.bootstrap(
            (t, c), independent_g, paired=False, method='BCa', n_resamples=300,
            random_state=np.random.default_rng(29),
        )
        result = effect_size_bca(t, c, effect_type='d', n_boot=300, random_state=29)
        np.testing.assert_allclose([result['ci_lower'], result['ci_upper']], reference.confidence_interval)


class IndependentModelTests(unittest.TestCase):
    def test_kh_audit_example_independent_hand_calculation(self):
        result = r_bridge._rma_python_fallback([.1, .2, .3], [.01] * 3)
        se = math.sqrt(1 / 300)
        test = .2 / se
        # For t(df=2), two-sided p = 1 - |t|/sqrt(t²+2).
        self.assertAlmostEqual(result['se'], se, places=14)
        self.assertAlmostEqual(result['p_value'], 1 - test / math.sqrt(test * test + 2), places=14)
        # Closed-form t(df=2) quantile; tolerate numerical inverse-CDF error.
        critical = .95 * math.sqrt(2 / (1 - .95 ** 2))
        self.assertAlmostEqual(result['ci_lower'], .2 - critical * se, places=11)
        self.assertEqual((result['test'], result['df']), ('knha', 2))

    def test_kh_with_two_studies_uses_t_one_df(self):
        result = r_bridge._rma_python_fallback([.1, .3], [.01, .01])
        critical = math.tan(math.pi * (.975 - .5))
        self.assertAlmostEqual(result['se'], .1, places=14)
        self.assertAlmostEqual(result['ci_lower'], .2 - critical * .1, places=12)
        self.assertAlmostEqual(result['p_value'], 1 - 2 * math.atan(2) / math.pi, places=14)
        self.assertEqual((result['knha'], result['test'], result['df']), (True, 'knha', 1))

    def test_dl_wald_and_kh_nonuniform_variance_reference(self):
        y, v = [-.8, .1, .4, 1.3], [.03, .12, .04, .08]
        for kh in (False, True):
            with self.subTest(khna=kh):
                result = r_bridge._rma_python_fallback(y, v, knha=kh)
                expected = manual_dl(y, v, kh)
                np.testing.assert_allclose(
                    [result[key] for key in ('pooled_effect', 'se', 'ci_lower', 'ci_upper', 'p_value', 'tau2')],
                    expected, rtol=1e-13, atol=1e-14,
                )
                self.assertEqual(result['test'], 'knha' if kh else 'z')
                if not kh:
                    self.assertAlmostEqual(result['p_value'], math.erfc(abs(expected[0] / expected[1]) / math.sqrt(2)))

    def test_kh_is_scale_equivariant(self):
        y, v = np.array([.1, .5, -.2, .4]), np.array([.01, .04, .02, .08])
        original = r_bridge._rma_python_fallback(y, v)
        scaled = r_bridge._rma_python_fallback(y * 100, v * 10000)
        self.assertAlmostEqual(scaled['se'], original['se'] * 100)
        self.assertAlmostEqual(scaled['p_value'], original['p_value'])

    def test_zero_kh_scale_does_not_crash_or_claim_inference(self):
        result = r_bridge._rma_python_fallback([.25] * 3, [.01] * 3)
        self.assertEqual(result['se'], 0)
        self.assertFalse(result['inference_available'])
        self.assertIsNone(result['p_value'])
        self.assertTrue(result['warnings'])

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_fallback_reports_actual_not_requested_estimator(self, _):
        with self.assertWarnsRegex(RuntimeWarning, 'DL'):
            result = r_bridge.rma_random_effects([.1, .2, .3], [.01] * 3, method='REML')
        self.assertEqual((result['method'], result['requested_method'], result['backend']), ('DL', 'REML', 'python'))

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_dependence_is_rejected_before_either_backend(self, runner):
        vi = [.01, .02, .03]
        tiny_covariance = np.diag(vi)
        tiny_covariance[0, 1] = tiny_covariance[1, 0] = 1e-30
        for extra in ({'study_ids': ['a', 'a', 'b']}, {'sampling_covariance': tiny_covariance}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, '多层/多变量'):
                r_bridge.rma_random_effects([.1, .2, .3], vi, **extra)
        runner.assert_not_called()

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_unique_studies_and_diagonal_covariance_can_use_fallback(self, _):
        vi = [.01, .02, .03]
        result = r_bridge.rma_random_effects(
            [.1, .2, .3], vi, method='DL', study_ids=['a', 'b', 'c'], sampling_covariance=np.diag(vi),
        )
        self.assertEqual(result['n_independent'], 3)
        self.assertEqual(result['sampling_covariance'], 'validated_diagonal')

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_covariance_ids_and_model_input_validation(self, runner):
        cases = [
            {'study_ids': ['a', 'b']}, {'study_ids': ['a', None, 'b']},
            {'study_ids': ['a', float('nan'), 'b']}, {'sampling_covariance': np.eye(2)},
            {'sampling_covariance': np.diag([.01, .02, .04])},
            {'sampling_covariance': [[.01, .001, 0], [0, .02, 0], [0, 0, .03]]},
        ]
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                r_bridge.rma_random_effects([.1, .2, .3], [.01, .02, .03], **extra)
        for y, v in (([.1], [.01]), ([.1, .2], [0, .01]), ([[.1, .2]], [.01, .01])):
            with self.subTest(y=y, v=v), self.assertRaises(ValueError):
                r_bridge.rma_random_effects(y, v)
        runner.assert_not_called()

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_subgroup_fallback_preserves_actual_model_labels(self, _):
        result = r_bridge.subgroup_analysis(
            [.1, .3, .2, .4], [.01] * 4, ['a', 'a', 'b', 'b'], method='DL',
        )
        self.assertEqual(result['model'], 'subgroup_random_effects')
        self.assertIsNone(result['Q_between'])
        self.assertEqual(result['method'], 'DL')
        for subgroup in result['subgroups'].values():
            self.assertEqual((subgroup['method'], subgroup['test'], subgroup['df']), ('DL', 'knha', 1))
            self.assertEqual(subgroup['model'], 'random_effects')
            self.assertEqual(subgroup['backend'], 'python')


class EggerTests(unittest.TestCase):
    @staticmethod
    def fixture(k=10):
        precision = np.arange(1., k + 1)
        standardized = .5 * precision + .1 * (-1.) ** (precision + 1)
        return standardized / precision, precision ** -2

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_intercept_standard_error_and_p_not_slope(self, _):
        y, v = self.fixture()
        result = r_bridge.egger_test(y, v)
        precision = 1 / np.sqrt(v)
        design = np.column_stack([np.ones(10), precision])
        beta = np.linalg.solve(design.T @ design, design.T @ (y / np.sqrt(v)))
        residual = y / np.sqrt(v) - design @ beta
        covariance = (residual @ residual / 8) * np.linalg.inv(design.T @ design)
        expected_se = math.sqrt(covariance[0, 0])
        expected_p = 2 * stats.t.sf(abs(beta[0] / expected_se), 8)
        self.assertAlmostEqual(result['intercept'], beta[0], places=13)
        self.assertAlmostEqual(result['intercept_se'], expected_se, places=13)
        self.assertAlmostEqual(result['p_value'], expected_p, places=12)
        self.assertAlmostEqual(result['p_value'], .6693599, places=6)
        self.assertLess(result['slope_p_value'], 1e-8)
        self.assertGreater(result['p_value'], .5)
        self.assertEqual((result['test'], result['df']), ('t', 8))
        self.assertEqual(result['z_value'], result['t_value'])
        self.assertEqual(result['bias_intercept'], result['intercept'])
        self.assertEqual(result['bias_intercept_se'], result['intercept_se'])
        self.assertEqual(result['intercept_p_value'], result['p_value'])
        self.assertTrue(result['inference_allowed'])
        self.assertNotIn('publication_bias', result)
        self.assertNotIn('significant', result)

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_small_independent_sample_retains_finite_diagnostics_without_auto_inference(self, _):
        y, v = self.fixture(5)
        with self.assertWarnsRegex(RuntimeWarning, '独立研究'):
            result = r_bridge.egger_test(y, v, study_ids=list('abcde'))
        self.assertTrue(result['success'])
        self.assertFalse(result['inference_allowed'])
        self.assertTrue(result['exploratory'])
        self.assertTrue(np.all(np.isfinite([result['intercept'], result['intercept_se'], result['p_value']])))
        self.assertTrue(result['warnings'])

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_dependent_rows_cannot_inflate_independent_count(self, runner):
        y, v = self.fixture(10)
        with self.assertRaisesRegex(ValueError, '同研究多效应'):
            r_bridge.egger_test(y, v, study_ids=['a'] * 10)
        runner.assert_not_called()

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_unidentifiable_regressions_return_explicit_failure(self, runner):
        for y, v in (([.1, .2], [.01, .02]), ([.1, .2, .3], [.01] * 3)):
            with self.subTest(y=y):
                result = r_bridge.egger_test(y, v)
                self.assertFalse(result['success'])
                self.assertFalse(result['inference_allowed'])
        runner.assert_not_called()

    @patch.object(r_bridge, '_run_r_script', return_value=NO_R)
    def test_near_constant_precision_is_exploratory(self, _):
        precision = 1 + np.arange(10) * 1e-6
        y = (.5 * precision + .1 * (-1.) ** np.arange(10)) / precision
        with self.assertWarnsRegex(RuntimeWarning, '跨度'):
            result = r_bridge.egger_test(y, precision ** -2)
        self.assertTrue(result['success'])
        self.assertFalse(result['inference_allowed'])
        self.assertTrue(np.isfinite(result['intercept_se']))


class RDiscoveryTests(unittest.TestCase):
    def test_environment_override_precedes_path(self):
        with patch.dict(os.environ, {'R_SCRIPT': r'Q:\custom R\Rscript.exe'}), patch.object(
            r_bridge.shutil, 'which', return_value=r'Q:\custom R\Rscript.exe'
        ) as which:
            self.assertEqual(r_bridge.find_rscript(), r'Q:\custom R\Rscript.exe')
            which.assert_called_once_with(r'Q:\custom R\Rscript.exe')

    def test_path_precedes_legacy_locations(self):
        with patch.dict(os.environ, {'R_SCRIPT': ''}), patch.object(
            r_bridge.shutil, 'which', return_value=r'Q:\path R\Rscript.exe'
        ):
            self.assertEqual(r_bridge.find_rscript(), r'Q:\path R\Rscript.exe')

    def test_invalid_explicit_configuration_is_not_silently_ignored(self):
        with patch.dict(os.environ, {'R_SCRIPT': r'Q:\missing\Rscript.exe'}), patch.object(
            r_bridge.shutil, 'which', return_value=None
        ), patch.object(r_bridge.os.path, 'isfile', return_value=False):
            with self.assertRaisesRegex(ValueError, 'R_SCRIPT'):
                r_bridge.find_rscript()

    def test_missing_explicit_executable_returns_failure_for_fallback(self):
        with patch.object(r_bridge.subprocess, 'run', side_effect=FileNotFoundError('missing R')):
            result = r_bridge._run_r_script('', rscript_path='missing-Rscript')
        self.assertFalse(result['success'])
        self.assertIn('missing R', result['error'])


class OptionalMetaforComparisons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.rscript = r_bridge.find_rscript()
        except ValueError as exc:
            raise unittest.SkipTest(str(exc))
        if cls.rscript is None:
            raise unittest.SkipTest('Rscript unavailable; Python numerical tests still run')
        probe = r_bridge._run_r_script(
            'cat(jsonlite::toJSON(list(success=requireNamespace("metafor", quietly=TRUE)), auto_unbox=TRUE))',
            cls.rscript,
        )
        if not probe['success']:
            raise unittest.SkipTest('metafor/jsonlite unavailable')

    def test_r_dl_wald_and_kh_match_python_including_k_two(self):
        fixtures = [([.1, .2, .3], [.01] * 3), ([.1, .3], [.01] * 2),
                    ([-.8, .1, .4, 1.3], [.03, .12, .04, .08])]
        for y, v in fixtures:
            for kh in (False, True):
                with self.subTest(k=len(y), kh=kh):
                    actual = r_bridge.rma_random_effects(y, v, method='DL', knha=kh, rscript_path=self.rscript)
                    reference = r_bridge._rma_python_fallback(y, v, knha=kh)
                    self.assertEqual(actual['backend'], 'metafor', actual)
                    for key in ('pooled_effect', 'se', 'ci_lower', 'ci_upper', 'p_value', 'tau2', 'I2', 'Q'):
                        np.testing.assert_allclose(actual[key], reference[key], rtol=1e-12, atol=1e-13, err_msg=key)
                    self.assertEqual(actual['test'], reference['test'])
                    self.assertEqual(actual['df'], reference['df'])
                    self.assertIn('metafor_version', actual)

    def test_r_egger_has_same_intercept_semantics_and_precision(self):
        y, v = EggerTests.fixture()
        actual = r_bridge.egger_test(y, v, rscript_path=self.rscript)
        with patch.object(r_bridge, '_run_r_script', return_value=NO_R):
            reference = r_bridge.egger_test(y, v)
        self.assertEqual(actual['backend'], 'metafor', actual)
        for key in ('intercept', 'intercept_se', 'p_value', 'slope', 'slope_se', 'slope_p_value', 'test_statistic'):
            np.testing.assert_allclose(actual[key], reference[key], rtol=1e-11, atol=1e-14, err_msg=key)
        self.assertGreater(actual['slope_p_value'], 0)  # default JSON rounding used to erase tiny p-values

    def test_r_subgroups_report_actual_models_and_group_comparison(self):
        result = r_bridge.subgroup_analysis(
            [.1, .3, .2, .4, .6, .9], [.01, .02, .03, .02, .04, .05],
            ['a', 'a', 'a', 'b', 'b', 'b'], rscript_path=self.rscript, method='DL', knha=True,
        )
        self.assertEqual(result['backend'], 'metafor', result)
        self.assertEqual(result['Q_between_test'], 'F')
        for subgroup in result['subgroups'].values():
            self.assertEqual(subgroup['model'], 'random_effects')
            self.assertEqual((subgroup['method'], subgroup['test'], subgroup['df']), ('DL', 'knha', 2))

    def test_r_rom_variance_matches_metafor_escalc(self):
        reference = r_bridge._run_r_script('''
            x <- metafor::escalc(measure="ROM", m1i=10, m2i=11,
                                 sd1i=2, sd2i=2, n1i=3, n2i=3, correct=FALSE)
            cat(jsonlite::toJSON(list(success=TRUE, yi=x$yi, vi=x$vi),
                                auto_unbox=TRUE, digits=NA))
        ''', self.rscript)
        self.assertTrue(reference['success'], reference)
        self.assertAlmostEqual(reference['result']['yi'], math.log(10 / 11), places=14)
        self.assertAlmostEqual(reference['result']['vi'], log_response_ratio_variance(10, 11, 2, 2, 3, 3), places=14)

    def test_r_errors_have_boolean_failure_not_truthy_list(self):
        result = r_bridge._run_r_script('stop("intentional numerical-test error")', self.rscript)
        self.assertIs(result['success'], False)
        self.assertIn('intentional numerical-test error', result['error'])


if __name__ == '__main__':
    unittest.main()

"""Independent numerical regressions for proxy synthesis (synthetic data only)."""

from copy import deepcopy
from itertools import product
import math
import unittest

import numpy as np

from scripts.multi_proxy import (
    leave_one_proxy_out,
    multi_proxy_synthesis,
    prepare_proxy_records,
)


def record(site="A", proxy="p", **updates):
    data = {
        "site_id": site,
        "proxy_id": proxy,
        "target": "shared",
        "ages": np.array([0.0, 1.0, 2.0]),
        "values": np.array([1.0, 2.0, 3.0]),
    }
    data.update(updates)
    return data


def deterministic(records, grid=(0.0, 1.0, 2.0), **kwargs):
    options = dict(
        bootstrap_sites=False, propagate_age=False, propagate_measurement=False
    )
    options.update(kwargs)
    return multi_proxy_synthesis(records, np.asarray(grid), **options)


def age_records(*, shared=False, same_site=False):
    ages = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]])
    records = [
        record(age_ensembles=ages),
        record("A" if same_site else "B", "q", age_ensembles=ages.copy()),
    ]
    if shared:
        for item in records:
            item.update(chronology_group="joint", age_member_ids=["young", "old"])
    return records


class StandardizationTests(unittest.TestCase):
    def test_zscore_location_scale_and_error_against_hand_calculation(self):
        prepared = prepare_proxy_records(record(measurement_error=0.2))["records"][0]
        scale = math.sqrt(2.0 / 3.0)
        self.assertEqual(prepared["location"], 2.0)
        self.assertAlmostEqual(prepared["scale"], scale)
        np.testing.assert_allclose(
            prepared["standardized_values"], [-1.0 / scale, 0.0, 1.0 / scale]
        )
        np.testing.assert_allclose(prepared["measurement_error"], 0.24494897427831783)
        np.testing.assert_array_equal(prepared["measurement_error_raw"], [0.2] * 3)

    def test_translation_and_unit_scaling_leave_zscores_and_errors_invariant(self):
        original = record(measurement_error=[0.1, 0.2, 0.3])
        a = prepare_proxy_records(original)["records"][0]
        for multiplier, offset in [(1.0, 10.0), (1000.0, -37.0), (0.25, 12.0)]:
            with self.subTest(multiplier=multiplier, offset=offset):
                converted = dict(
                    original,
                    values=original["values"] * multiplier + offset,
                    measurement_error=np.array([0.1, 0.2, 0.3]) * multiplier,
                )
                b = prepare_proxy_records(converted)["records"][0]
                np.testing.assert_allclose(a["standardized_values"], b["standardized_values"])
                np.testing.assert_allclose(a["measurement_error"], b["measurement_error"])
                self.assertAlmostEqual(b["location"], multiplier * 2.0 + offset)
                self.assertAlmostEqual(b["scale"], multiplier * math.sqrt(2.0 / 3.0))

    def test_robust_standardization_and_error_use_mad_scale(self):
        item = record(values=[1.0, 2.0, 100.0], standardization="robust", error=0.2)
        prepared = prepare_proxy_records(item)["records"][0]
        self.assertEqual(prepared["location"], 2.0)
        self.assertAlmostEqual(prepared["scale"], 1.4826)
        np.testing.assert_allclose(prepared["standardized_values"], np.array([-1, 0, 98]) / 1.4826)
        np.testing.assert_allclose(prepared["measurement_error"], 0.2 / 1.4826)

    def test_robust_iqr_fallback_has_consistent_location_and_scale(self):
        item = record(
            ages=[0.0, 1.0, 2.0, 3.0], values=[0.0, 0.0, 0.0, 4.0],
            standardization="robust", measurement_error=0.2,
        )
        prepared = prepare_proxy_records(item)["records"][0]
        self.assertEqual(prepared["location"], 0.0)
        self.assertAlmostEqual(prepared["scale"], 1.0 / 1.349)
        np.testing.assert_allclose(prepared["measurement_error"], 0.2 * 1.349)

    def test_none_retains_units_and_error_without_division_by_zero(self):
        item = record(
            values=[11.0, 12.0, 13.0], standardization="none", unit="degC",
            measurement_error=0.2, direction="negative",
        )
        info = prepare_proxy_records(item)
        prepared = info["records"][0]
        self.assertEqual((prepared["location"], prepared["scale"]), (0.0, 1.0))
        np.testing.assert_array_equal(prepared["standardized_values"], [-11.0, -12.0, -13.0])
        np.testing.assert_array_equal(prepared["measurement_error"], [0.2] * 3)
        self.assertEqual((info["analysis_scale"], info["analysis_unit"]), ("raw", "degC"))

    def test_baseline_period_fits_the_declared_observations_only(self):
        prepared = prepare_proxy_records(
            record(baseline_period=(0.0, 1.0), measurement_error=0.2)
        )["records"][0]
        self.assertEqual((prepared["location"], prepared["scale"]), (1.5, 0.5))
        np.testing.assert_array_equal(prepared["standardized_values"], [-1.0, 1.0, 3.0])
        np.testing.assert_allclose(prepared["measurement_error"], 0.4)

    def test_preparation_round_trip_does_not_rescale_errors_twice(self):
        for error_field in ("measurement_error", "error", "proxy_error"):
            for method in ("zscore", "robust", "none"):
                with self.subTest(error_field=error_field, method=method):
                    raw = record(standardization=method, **{error_field: [0.1, 0.2, 0.3]})
                    first = prepare_proxy_records(raw)
                    for reusable in (first, first["records"], first["records"][0]):
                        second = prepare_proxy_records(reusable)["records"][0]
                        for key in ("values", "standardized_values", "measurement_error_raw", "measurement_error"):
                            np.testing.assert_array_equal(second[key], first["records"][0][key])
                        self.assertEqual(second["scale"], first["records"][0]["scale"])

    def test_prepared_and_raw_synthesis_have_identical_seeded_distributions(self):
        records = [
            record(measurement_error=0.2),
            record(proxy="q", values=[2.0, 4.0, 9.0], proxy_error=0.3),
        ]
        options = dict(
            bootstrap_sites=False, propagate_age=False, n_members=48,
            weighting="precision", random_state=19,
        )
        a = multi_proxy_synthesis(records, np.array([0.5, 1.5]), **options)
        b = multi_proxy_synthesis(prepare_proxy_records(records), np.array([0.5, 1.5]), **options)
        np.testing.assert_array_equal(a["ensembles"], b["ensembles"])

    def test_precision_synthesis_is_translation_and_unit_invariant(self):
        raw = [
            record(measurement_error=0.2),
            record(proxy="q", values=[2.0, 5.0, 4.0], measurement_error=0.3),
        ]
        converted = deepcopy(raw)
        for item, multiplier, offset in zip(converted, [100.0, 0.5], [40.0, -20.0]):
            item["values"] = np.asarray(item["values"]) * multiplier + offset
            item["measurement_error"] *= multiplier
        options = dict(
            weighting="precision", bootstrap_sites=False, propagate_age=False,
            n_members=48, random_state=41,
        )
        a = multi_proxy_synthesis(raw, np.array([0.5, 1.5]), **options)
        b = multi_proxy_synthesis(converted, np.array([0.5, 1.5]), **options)
        np.testing.assert_allclose(a["ensembles"], b["ensembles"], atol=1e-12)

    def test_raw_and_standardized_scales_cannot_be_mixed_even_if_units_match(self):
        for method in ("zscore", "robust"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "Cannot mix raw"):
                prepare_proxy_records([
                    record(standardization="none", unit="dimensionless"),
                    record("B", standardization=method, unit="dimensionless"),
                ])

    def test_multiple_raw_records_require_complete_matching_units(self):
        for units in [(None, None), ("degC", None), ("", ""), ("degC", "K")]:
            with self.subTest(units=units), self.assertRaisesRegex(ValueError, "same unit"):
                prepare_proxy_records([
                    record(standardization="none", unit=units[0]),
                    record("B", standardization="none", unit=units[1]),
                ])

    def test_ambiguous_scale_input_is_explicitly_rejected(self):
        for scale in ("zscore", "none", 2.0):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "standardization alias"):
                prepare_proxy_records(record(scale=scale))

    def test_numeric_missing_units_are_not_stringified_as_matching_raw_units(self):
        for unit in (np.nan, 1.0):
            with self.subTest(unit=unit), self.assertRaisesRegex(ValueError, "unit labels must be strings"):
                prepare_proxy_records([
                    record(standardization="none", unit=unit),
                    record("B", standardization="none", unit=unit),
                ])

    def test_provenance_is_copied_into_preparation_and_synthesis_metadata(self):
        fields = dict(
            source={"doi": "synthetic", "table": 2}, method="synthetic-method",
            sample_id=["s1", "s2", "s3"], age_unit="cal yr BP",
            custom_metadata={"core": "test-core", "flags": [1, 2]},
        )
        raw = record(**deepcopy(fields))
        info = prepare_proxy_records(raw)
        result = deterministic(info)
        for key, expected in fields.items():
            self.assertEqual(info["records"][0][key], expected)
            self.assertEqual(result["record_metadata"][0][key], expected)
        info["records"][0]["source"]["table"] = 999
        result["record_metadata"][0]["sample_id"][0] = "changed"
        self.assertEqual(raw["source"]["table"], 2)
        self.assertEqual(raw["sample_id"][0], "s1")
        np.testing.assert_array_equal(raw["values"], [1.0, 2.0, 3.0])


class ChronologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.options = dict(
            n_members=6000, bootstrap_sites=False,
            propagate_measurement=False, random_state=1729,
        )
        cls.independent = multi_proxy_synthesis(age_records(), np.array([0.5]), **cls.options)
        shared_records = age_records(shared=True)
        shared_records[1]["age_ensembles"] = shared_records[1]["age_ensembles"][::-1].copy()
        shared_records[1]["age_member_ids"] = ["old", "young"]
        cls.shared = multi_proxy_synthesis(shared_records, np.array([0.5]), **cls.options)

    def test_independent_chronologies_match_cartesian_variance_not_row_pairing(self):
        # Each z-scored interpolant is +/-sqrt(3/8). Enumerate the independent
        # product distribution, without calling any standardization helper.
        outcomes = [-math.sqrt(3.0 / 8.0), math.sqrt(3.0 / 8.0)]
        combinations = np.array([(left + right) / 2 for left, right in product(outcomes, repeat=2)])
        self.assertAlmostEqual(float(np.var(combinations)), 0.1875)
        self.assertAlmostEqual(float(np.var(self.independent["ensembles"])), 0.1875, delta=0.012)
        self.assertAlmostEqual(float(np.mean(self.independent["ensembles"])), 0.0, delta=0.025)
        indices = self.independent["chronology_member_indices"]
        pairs = np.column_stack([indices["A:p"], indices["B:q"]])
        self.assertEqual(len(np.unique(pairs, axis=0)), 4)
        for a, b in product([0, 1], repeat=2):
            self.assertAlmostEqual(float(np.mean((pairs[:, 0] == a) & (pairs[:, 1] == b))), 0.25, delta=0.025)

    def test_shared_members_match_identity_even_with_reversed_storage_order(self):
        self.assertAlmostEqual(float(np.var(self.shared["ensembles"])), 0.375, delta=0.002)
        indices = self.shared["chronology_member_indices"]
        np.testing.assert_array_equal(indices["A:p"], 1 - indices["B:q"])
        np.testing.assert_allclose(np.abs(self.shared["ensembles"]), math.sqrt(3.0 / 8.0))
        self.assertEqual(self.shared["chronology_groups"][0]["member_ids"], ["old", "young"])

    def test_undeclared_records_at_one_site_are_still_independent(self):
        result = multi_proxy_synthesis(age_records(same_site=True), np.array([0.5]), **self.options)
        self.assertAlmostEqual(float(np.var(result["ensembles"])), 0.1875, delta=0.012)
        self.assertEqual(result["independent_chronology_records"], ["A:p", "A:q"])
        self.assertTrue(all(item["chronology_group"] is None for item in result["record_metadata"]))
        self.assertIn("same site", " ".join(result["chronology_assumptions"]))

    def test_different_explicit_groups_are_independent(self):
        records = age_records()
        records[0]["chronology_group"] = "first"
        records[1]["chronology_group"] = "second"
        result = multi_proxy_synthesis(records, np.array([0.5]), **self.options)
        self.assertAlmostEqual(float(np.var(result["ensembles"])), 0.1875, delta=0.012)

    def test_independent_member_row_permutation_is_seed_invariant(self):
        records = age_records()
        options = dict(self.options, n_members=128)
        a = multi_proxy_synthesis(records, np.array([0.5]), **options)
        records[1]["age_ensembles"] = records[1]["age_ensembles"][::-1].copy()
        b = multi_proxy_synthesis(records, np.array([0.5]), **options)
        np.testing.assert_array_equal(a["ensembles"], b["ensembles"])

    def test_record_order_does_not_assign_different_draws_to_groups(self):
        records = age_records()
        options = dict(self.options, n_members=128)
        a = multi_proxy_synthesis(records, np.array([0.5]), **options)
        b = multi_proxy_synthesis(records[::-1], np.array([0.5]), **options)
        np.testing.assert_array_equal(a["ensembles"], b["ensembles"])

    def test_member_draws_preserve_the_entire_age_trajectory(self):
        result = multi_proxy_synthesis(
            age_records()[:1], np.array([0.25, 0.75]), **dict(self.options, n_members=64)
        )
        # Both age members have slope 1/sqrt(2/3); drawing ages sample-by-sample
        # would fail this identity even if marginal variances happened to fit.
        differences = result["ensembles"][:, 1] - result["ensembles"][:, 0]
        np.testing.assert_allclose(differences, 0.5 / math.sqrt(2.0 / 3.0))

    def test_shared_groups_require_member_identity_not_equal_row_counts(self):
        records = age_records(shared=True)
        del records[1]["age_member_ids"]
        with self.assertRaisesRegex(ValueError, "requires explicit age_member_ids"):
            prepare_proxy_records(records)

    def test_shared_groups_reject_incompatible_member_sets_and_counts(self):
        for ids in (["old", "other"], ["old"]):
            records = age_records(shared=True)
            records[1]["age_member_ids"] = ids
            if len(ids) == 1:
                records[1]["age_ensembles"] = records[1]["age_ensembles"][:1]
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "same age_member_ids"):
                prepare_proxy_records(records)

    def test_shared_groups_reject_missing_age_ensembles(self):
        records = age_records(shared=True)
        del records[1]["age_ensembles"]
        del records[1]["age_member_ids"]
        with self.assertRaisesRegex(ValueError, "with and without age_ensembles"):
            prepare_proxy_records(records)

    def test_member_ids_must_be_unique_complete_finite_and_row_aligned(self):
        for ids in (["same", "same"], ["one"], [None, "two"], [np.nan, "two"], ["", "two"], "ab"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                prepare_proxy_records(dict(age_records()[0], age_member_ids=ids))

    def test_independent_groups_allow_different_member_counts(self):
        records = age_records()
        records[1]["age_ensembles"] = records[1]["age_ensembles"][:1]
        result = multi_proxy_synthesis(records, np.array([0.5]), **dict(self.options, n_members=32))
        np.testing.assert_array_equal(result["chronology_member_indices"]["B:q"], 0)
        self.assertEqual(result["n_members"], 32)


class CoverageWeightingAndLopoTests(unittest.TestCase):
    def test_two_stage_weighted_mean_matches_hand_calculation(self):
        records = [
            record(standardization="none", unit="common", site_weight=2.0),
            record(proxy="q", values=[5.0, 6.0, 7.0], standardization="none", unit="common", weight=3.0, site_weight=2.0),
            record("B", values=[10.0, 20.0, 30.0], standardization="none", unit="common"),
        ]
        result = deterministic(records)
        # Site A=(p+3q)/4=[4,5,6]; regional=(2*A+B)/3.
        np.testing.assert_allclose(result["composite"], [6.0, 10.0, 14.0])

    def test_precision_weighted_mean_matches_inverse_error_variance(self):
        records = [
            record(standardization="none", unit="common", measurement_error=2.0),
            record(proxy="q", values=[5.0, 6.0, 7.0], standardization="none", unit="common", measurement_error=1.0),
        ]
        result = deterministic(records, weighting="precision")
        # Weights 1/4 and 1: (p + 4*q)/5.
        np.testing.assert_allclose(result["composite"], [4.2, 5.2, 6.2])

    def test_precision_does_not_assign_known_precision_to_missing_or_zero_errors(self):
        for errors in ({}, {"measurement_error": 0.0}, {"measurement_error": [0.2, 0.0, 0.2]}):
            with self.subTest(errors=errors), self.assertRaisesRegex(ValueError, "known precision"):
                deterministic([record(**errors)], weighting="precision")
        self.assertTrue(np.all(np.isfinite(deterministic([record()])["composite"])))

    def test_bootstrap_never_draws_zero_weight_sites(self):
        records = [
            record(standardization="none", unit="common"),
            record("B", values=[100.0, 200.0, 300.0], standardization="none", unit="common", site_weight=0.0),
        ]
        result = multi_proxy_synthesis(
            records, np.array([0.0, 1.0, 2.0]), bootstrap_sites=True,
            propagate_age=False, propagate_measurement=False, n_members=128, random_state=29,
        )
        np.testing.assert_array_equal(result["ensembles"], np.tile([1.0, 2.0, 3.0], (128, 1)))
        np.testing.assert_array_equal(result["effective_site_count"], [1, 1, 1])
        np.testing.assert_array_equal(result["proxy_coverage"]["p"], [2, 2, 2])

    def test_proxy_coverage_counts_available_sites_at_each_time_not_bins(self):
        records = [
            record(standardization="none", unit="common"),
            record(proxy="q", ages=[1.0, 2.0, 3.0], values=[4.0, 5.0, 6.0], standardization="none", unit="common"),
            record("B", ages=[2.0, 3.0, 4.0], values=[10.0, 11.0, 12.0], standardization="none", unit="common"),
        ]
        result = deterministic(records, grid=[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(result["proxy_coverage"]["p"], [0, 1, 1, 2, 1, 1, 0])
        np.testing.assert_array_equal(result["proxy_coverage"]["q"], [0, 0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(result["effective_proxy_count"], [0, 1, 2, 2, 2, 1, 0])
        np.testing.assert_array_equal(result["effective_site_count"], [0, 1, 1, 2, 2, 1, 0])
        np.testing.assert_allclose(result["composite"], [np.nan, 1.0, 3.0, 7.0, 8.5, 12.0, np.nan], equal_nan=True)
        self.assertEqual(result["coverage_basis"], "available_sites_at_nominal_ages_before_weighting")

    def test_missing_value_edges_are_not_counted_as_covered(self):
        result = deterministic(
            [record(ages=[0.0, 1.0, 2.0, 3.0], values=[np.nan, 2.0, 3.0, np.nan])],
            grid=[0.0, 1.0, 2.0, 3.0],
        )
        np.testing.assert_array_equal(result["proxy_coverage"]["p"], [0, 1, 1, 0])
        np.testing.assert_array_equal(result["valid_members_per_time"], [0, 1, 1, 0])

    def test_ensemble_intervals_use_available_members_without_extrapolation(self):
        records = age_records()[:1]
        result = multi_proxy_synthesis(
            records, np.array([-2.0, -0.5, 0.5, 1.5, 3.0]),
            bootstrap_sites=False, propagate_measurement=False, n_members=128, random_state=5,
        )
        np.testing.assert_array_equal(result["valid_members_per_time"], np.isfinite(result["ensembles"]).sum(axis=0))
        np.testing.assert_array_equal(result["proxy_coverage"]["p"], [0, 0, 1, 1, 0])
        self.assertTrue(np.all(result["valid_members_per_time"][[1, 3]] > 0))
        self.assertTrue(np.all(result["valid_members_per_time"][[1, 3]] < 128))
        self.assertTrue(np.all(np.isfinite(result["uncertainty_band"]["lower"][[1, 2, 3]])))
        self.assertTrue(np.all(np.isnan(result["uncertainty_band"]["lower"][[0, 4]])))

    def test_default_lopo_preserves_legacy_deterministic_arrays_and_metrics(self):
        records = [record(), record(proxy="q", values=[3.0, 2.0, 1.0])]
        result = leave_one_proxy_out(records, np.array([0.0, 1.0, 2.0]))
        self.assertFalse(result["stochastic"])
        self.assertEqual(result["n_members"], 1)
        np.testing.assert_allclose(result["full"], [0.0, 0.0, 0.0])
        self.assertIsNone(result["full_result"]["ensembles"])
        for summary in result["by_excluded_proxy"].values():
            self.assertEqual(summary["n_members"], 1)
            self.assertIsNone(summary["uncertainty_band"])
            np.testing.assert_array_equal(summary["difference"], summary["composite"] - result["full"])
            self.assertAlmostEqual(summary["rmse"], 1.0)

    def test_stochastic_lopo_preserves_n_members_and_full_and_exclusion_bands(self):
        result = leave_one_proxy_out(
            age_records(shared=True), np.array([0.25, 0.5, 0.75]),
            propagate_age=True, n_members=200, random_state=17,
        )
        self.assertTrue(result["stochastic"])
        self.assertEqual(result["n_members"], 200)
        np.testing.assert_array_equal(result["full"], result["full_result"]["composite"])
        for summary in [result["full_result"], *result["by_excluded_proxy"].values()]:
            self.assertEqual(summary["n_members"], 200)
            self.assertEqual(summary["ensembles"].shape, (200, 3))
            expected = np.percentile(summary["ensembles"], [5, 50, 95], axis=0)
            for key, row in zip(["lower", "median", "upper"], expected):
                np.testing.assert_array_equal(summary["uncertainty_band"][key], row)
            self.assertEqual(summary["uncertainty_band"]["interval_type"], "ensemble_percentile")
            self.assertEqual(summary["uncertainty_layers"], ["chronology"])
            self.assertTrue(summary["chronology_assumptions"])

    def test_measurement_lopo_uses_prepared_errors_once_and_subsets_correlations(self):
        records = [record(measurement_error=0.2), record(proxy="q", measurement_error=0.3)]
        result = leave_one_proxy_out(
            prepare_proxy_records(records), np.array([0.0, 1.0, 2.0]),
            propagate_measurement=True, measurement_correlation=np.array([[1.0, 0.3], [0.3, 1.0]]),
            n_members=40, random_state=21,
        )
        self.assertEqual(result["n_members"], 40)
        for summary in result["by_excluded_proxy"].values():
            self.assertEqual(summary["ensembles"].shape, (40, 3))
            self.assertEqual(summary["uncertainty_layers"], ["measurement"])

    def test_stochastic_lopo_keeps_degenerate_exclusion_distribution(self):
        records = [record(measurement_error=0.2), record(proxy="q")]
        result = leave_one_proxy_out(
            records, np.array([0.0, 1.0, 2.0]), propagate_measurement=True,
            n_members=40, random_state=21,
        )
        certain = result["by_excluded_proxy"]["p"]
        self.assertEqual(certain["n_members"], 40)
        self.assertTrue(certain["distribution_degenerate"])
        self.assertFalse(certain["stochastic"])
        self.assertEqual(certain["uncertainty_layers"], [])
        np.testing.assert_array_equal(certain["ensembles"], np.tile(certain["composite"], (40, 1)))
        np.testing.assert_array_equal(certain["uncertainty_band"]["lower"], certain["uncertainty_band"]["upper"])
        np.testing.assert_array_equal(certain["valid_members_per_time"], [40, 40, 40])

    def test_lopo_does_not_report_inactive_uncertainty_as_stochastic(self):
        result = leave_one_proxy_out(
            [record(), record(proxy="q")], np.array([0.0, 1.0, 2.0]),
            propagate_age=True, n_members=200,
        )
        self.assertFalse(result["stochastic"])
        self.assertEqual(result["n_members"], 1)

    def test_n_members_rejects_non_integer_and_non_positive_inputs(self):
        for value in (0, -1, 2.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                deterministic([record()], n_members=value)


if __name__ == "__main__":
    unittest.main()

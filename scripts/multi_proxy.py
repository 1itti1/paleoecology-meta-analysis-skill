"""Proxy-level integration for heterogeneous paleoecology records.

The module deliberately uses a conservative two-stage estimand:

1. standardize and orient each proxy record independently;
2. combine proxies within a site, then combine sites regionally.

This avoids treating multiple proxies from one archive as independent regional
replicates.  It is a transparent common-target synthesis, not a replacement
for a proxy-specific forward model, REVEALS, or a Bayesian hierarchical model.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from ._utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        weighted_nanmean,
    )
except ImportError:  # pragma: no cover - supports direct script imports
    from _utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        weighted_nanmean,
    )


_STANDARDIZATION_METHODS = {"zscore", "robust", "none"}
_PREPARATION_VERSION = 1
_CHRONOLOGY_ASSUMPTIONS = [
    "Different chronology groups are sampled independently. Records without "
    "chronology_group are independent records, including records at the same site.",
    "Shared chronology_group membership asserts a joint posterior: age_member_ids "
    "identify corresponding members, never their storage row numbers. This "
    "correspondence must be justified by the caller.",
]


class _StandardizedValues(NamedTuple):
    values: np.ndarray
    location: float
    scale: float


def _text(value, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty.")
    return text


def _direction(value) -> float:
    if value is None:
        return 1.0
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"positive", "increase", "increasing", "+", "1"}:
            return 1.0
        if key in {"negative", "decrease", "decreasing", "-", "-1"}:
            return -1.0
        raise ValueError(
            "direction must be positive/increase or negative/decrease."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("direction must be +1 or -1.") from exc
    if numeric not in {-1.0, 1.0}:
        raise ValueError("direction must be +1 or -1.")
    return numeric


def _validate_age_series(ages: np.ndarray, name: str) -> float:
    """Validate monotonic age order and return its direction."""
    if len(ages) < 2:
        raise ValueError(f"{name} needs at least two age points.")
    differences = np.diff(ages)
    if not np.all((differences >= 0) | (differences <= 0)):
        raise ValueError(f"{name} contains non-finite age differences.")
    if not np.all(differences >= 0) and not np.all(differences <= 0):
        raise ValueError(f"{name} must be monotonic; age reversals are not allowed.")
    if not np.any(differences != 0):
        raise ValueError(f"{name} must contain at least two distinct ages.")
    return 1.0 if np.all(differences >= 0) else -1.0


def _standardize_values(
    values: np.ndarray,
    ages: np.ndarray,
    method: str,
    baseline_period: Optional[Tuple[float, float]],
) -> _StandardizedValues:
    """Return transformed values, location, and scale."""
    if method not in _STANDARDIZATION_METHODS:
        raise ValueError(
            f"standardization must be one of {sorted(_STANDARDIZATION_METHODS)}."
        )
    finite = np.isfinite(values)
    if finite.sum() < 2:
        raise ValueError("Each proxy record needs at least two finite values.")

    if baseline_period is None:
        fit_mask = finite
    else:
        if len(baseline_period) != 2:
            raise ValueError("baseline_period must be a (start, end) pair.")
        start, end = map(float, baseline_period)
        if not np.isfinite(start) or not np.isfinite(end) or start > end:
            raise ValueError("baseline_period must be finite and ordered.")
        fit_mask = finite & (ages >= start) & (ages <= end)
        if fit_mask.sum() < 2:
            raise ValueError(
                "baseline_period must contain at least two finite observations."
            )

    if method == "none":
        return _StandardizedValues(values.copy(), 0.0, 1.0)

    fit = values[fit_mask]
    if method == "zscore":
        location = float(np.mean(fit))
        scale = float(np.std(fit, ddof=0))
    else:
        location = float(np.median(fit))
        mad = float(np.median(np.abs(fit - location)))
        scale = 1.4826 * mad
        if scale == 0:
            q25, q75 = np.percentile(fit, [25, 75])
            scale = float((q75 - q25) / 1.349)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"{method} standardization has zero or non-finite scale; "
            "the proxy record is not informative for synthesis."
        )
    transformed = (values - location) / scale
    return _StandardizedValues(transformed, location, scale)


def _measurement_error(
    record: Mapping,
    n: int,
    scale: float,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize a scalar or sample-level measurement error to transformed units."""
    if record.get("_multi_proxy_prepared") == _PREPARATION_VERSION:
        # Prepared measurement_error is already in analysis units. Recompute
        # from the retained raw errors, not from that derived field or aliases.
        if "measurement_error_raw" not in record:
            raise ValueError(f"{name} is prepared but lacks measurement_error_raw.")
        candidates = ["measurement_error_raw"]
    else:
        candidates = [
            key for key in ("measurement_error", "error", "proxy_error") if key in record
        ]
    if len(candidates) > 1:
        raise ValueError(
            f"{name} specifies more than one measurement-error field: {candidates}."
        )
    if not candidates:
        zeros = np.zeros(n, dtype=float)
        return zeros, zeros.copy()

    raw = np.asarray(record[candidates[0]], dtype=float)
    if raw.ndim == 0:
        errors = np.full(n, float(raw), dtype=float)
    elif raw.ndim == 1 and len(raw) == n:
        errors = raw.copy()
    else:
        raise ValueError(
            f"{name}.{candidates[0]} must be a scalar or an array with one value per sample."
        )
    if np.any(~np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError(f"{name}.{candidates[0]} must be finite and non-negative.")
    return errors.copy(), errors / abs(scale)


def _record_iter(proxy_records) -> List[Mapping]:
    if isinstance(proxy_records, Mapping):
        required = {"site_id", "proxy_id", "ages", "values"}
        if required.issubset(proxy_records):
            return [proxy_records]
        if "_multi_proxy_prepared" in proxy_records:
            if proxy_records["_multi_proxy_prepared"] != _PREPARATION_VERSION:
                raise ValueError("Unsupported prepared proxy-record version.")
            return list(proxy_records["records"])
        return list(proxy_records.values())
    try:
        records = list(proxy_records)
    except TypeError as exc:
        raise ValueError("proxy_records must be a record or an iterable of records.") from exc
    return records


def _age_member_ids(value, n_members: int, name: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    ids = np.asarray(value, dtype=object)
    if ids.ndim != 1 or len(ids) != n_members:
        raise ValueError(f"{name} must contain one ID per age_ensembles row.")
    normalized = []
    for member_id in ids:
        if not isinstance(member_id, (str, int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must contain non-empty, finite scalar IDs.")
        if not isinstance(member_id, str) and not np.isfinite(member_id):
            raise ValueError(f"{name} must contain non-empty, finite scalar IDs.")
        normalized.append(_text(member_id, name))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique IDs.")
    return tuple(normalized)


def _chronology_groups(records: Sequence[Mapping]) -> List[Dict]:
    """Validate joint-member identities and map canonical members to row indices.

    A site ID is not evidence of a shared chronology. Unlabelled records get
    separate groups. Canonical ordering makes seeded draws insensitive to row
    permutations, but does not couple draws from independent groups.
    """
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        declared = record["chronology_group"]
        key = ("record", record["record_id"]) if declared is None else ("group", declared)
        grouped[key].append(index)

    groups = []
    for (kind, group_id), indices in sorted(grouped.items()):
        uncertain = [i for i in indices if records[i]["age_ensembles"] is not None]
        shared = kind == "group" and len(indices) > 1
        if shared and uncertain:
            if len(uncertain) != len(indices):
                raise ValueError(
                    f"chronology_group {group_id!r} mixes records with and without "
                    "age_ensembles; shared member correspondence is undefined."
                )
            if any(records[i]["age_member_ids"] is None for i in indices):
                raise ValueError(
                    f"Shared chronology_group {group_id!r} requires explicit "
                    "age_member_ids for every record; row order is not a correspondence."
                )
            expected = set(records[indices[0]]["age_member_ids"])
            if any(set(records[i]["age_member_ids"]) != expected for i in indices):
                raise ValueError(
                    f"Shared chronology_group {group_id!r} must have the same "
                    "age_member_ids (and member count) in every record."
                )

        row_indices = {}
        member_ids = None
        n_available = 0
        for i in uncertain:
            record = records[i]
            ids = record["age_member_ids"]
            n_available = len(record["age_ensembles"])
            if ids is None:
                # Equal-probability members without identity can be ordered by
                # their age trajectories, never matched to another record.
                order = np.lexsort(record["age_ensembles"].T[::-1])
            else:
                member_ids = sorted(ids)
                positions = {member_id: row for row, member_id in enumerate(ids)}
                order = np.asarray([positions[member_id] for member_id in member_ids])
            row_indices[record["record_id"]] = order
        groups.append(
            {
                "chronology_group": group_id if kind == "group" else None,
                "group_type": "declared" if kind == "group" else "independent_record",
                "record_ids": [records[i]["record_id"] for i in indices],
                "n_available_members": n_available,
                "member_ids": member_ids,
                "row_indices": row_indices,
            }
        )
    return groups


def _chronology_metadata(groups: Sequence[Mapping]) -> List[Dict]:
    return [
        {key: deepcopy(value) for key, value in group.items() if key != "row_indices"}
        for group in groups
    ]


def prepare_proxy_records(
    proxy_records: Union[Mapping, Iterable[Mapping]],
    *,
    target: Optional[str] = None,
    default_standardization: str = "zscore",
) -> Dict:
    """Validate and transform a collection of proxy records.

    Each record must contain ``site_id``, ``proxy_id``, ``ages``, and
    ``values``.  Optional fields are ``proxy_type``, ``target``, ``unit``,
    ``direction``, ``standardization`` (``zscore``, ``robust``, or ``none``),
    ``baseline_period``, ``measurement_error``, ``age_ensembles``, ``weight``,
    ``site_weight``, ``chronology_group``, ``age_member_ids``, and ``record_id``.
    Shared uncertain chronologies require explicit, unique ``age_member_ids``
    with the same ID set in every record in that group. Unlabelled records are
    independent even at the same site. Members are assumed equally probable.

    Raw values are retained in ``values``.  The transformed series is stored in
    ``standardized_values`` and is always oriented so larger values support the
    declared common target.  The function does not infer ecological meaning or
    calibrate a proxy to climate. Other input metadata is copied unchanged.
    Both this result and its ``records`` can be passed back to the public
    functions: raw values/errors are re-used, not transformed a second time.
    ``scale`` is a derived numeric field, not an alias for ``standardization``.
    Raw (``none``) records cannot be mixed with standardized records, and a
    collection of raw records requires the same declared unit on every record.
    """
    if default_standardization not in _STANDARDIZATION_METHODS:
        raise ValueError(
            f"default_standardization must be one of {sorted(_STANDARDIZATION_METHODS)}."
        )
    raw_records = _record_iter(proxy_records)
    if not raw_records:
        raise ValueError("proxy_records must contain at least one record.")

    prepared: List[Dict] = []
    seen_record_keys = set()
    seen_site_proxy = set()
    declared_targets = set()

    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"proxy_records[{index}] must be a mapping.")
        name = f"proxy_records[{index}]"
        is_prepared = raw.get("_multi_proxy_prepared") == _PREPARATION_VERSION
        if "_multi_proxy_prepared" in raw and not is_prepared:
            raise ValueError(f"{name} has an unsupported prepared-record version.")
        if "scale" in raw and not is_prepared:
            raise ValueError(
                f"{name}.scale is a derived field, no longer a standardization "
                "alias; set standardization='zscore', 'robust', or 'none' explicitly."
            )
        site_id = _text(raw.get("site_id"), f"{name}.site_id")
        proxy_id = _text(raw.get("proxy_id"), f"{name}.proxy_id")
        site_proxy = (site_id, proxy_id)
        if site_proxy in seen_site_proxy:
            raise ValueError(
                f"Duplicate site/proxy record {site_id!r}/{proxy_id!r}; "
                "aggregate technical replicates before regional synthesis."
            )
        seen_site_proxy.add(site_proxy)

        ages = as_float_array(raw.get("ages"), f"{name}.ages", ndim=1, allow_nan=False)
        values = as_float_array(raw.get("values"), f"{name}.values", ndim=1)
        if np.any(np.isinf(values)) or len(ages) != len(values):
            raise ValueError(f"{name}.values must match ages and contain no infinities.")
        age_direction = _validate_age_series(ages, f"{name}.ages")

        age_ensembles = raw.get("age_ensembles")
        if age_ensembles is not None:
            age_ensembles = as_float_array(
                age_ensembles, f"{name}.age_ensembles", ndim=2, allow_nan=False
            )
            if age_ensembles.shape[1] != len(ages) or age_ensembles.shape[0] < 1:
                raise ValueError(
                    f"{name}.age_ensembles must have shape (members, {len(ages)})."
                )
            for member_index, member_ages in enumerate(age_ensembles):
                member_direction = _validate_age_series(
                    member_ages, f"{name}.age_ensembles[{member_index}]"
                )
                if member_direction != age_direction:
                    raise ValueError(
                        f"{name}.age_ensembles changes age direction relative to ages."
                    )
        if age_ensembles is None and raw.get("age_member_ids") is not None:
            raise ValueError(f"{name}.age_member_ids requires age_ensembles.")
        member_ids = _age_member_ids(
            raw.get("age_member_ids"),
            0 if age_ensembles is None else len(age_ensembles),
            f"{name}.age_member_ids",
        )

        standardization = str(
            raw.get("standardization", default_standardization)
        ).strip().lower()
        baseline_period = deepcopy(raw.get("baseline_period"))
        standardized = _standardize_values(
            values, ages, standardization, baseline_period
        )
        direction = _direction(raw.get("direction", "positive"))
        transformed = direction * standardized.values
        raw_errors, errors = _measurement_error(raw, len(values), standardized.scale, name)

        record_id = _text(
            raw.get("record_id", f"{site_id}:{proxy_id}"), f"{name}.record_id"
        )
        if record_id in seen_record_keys:
            raise ValueError(f"Duplicate record_id: {record_id!r}.")
        seen_record_keys.add(record_id)

        declared_target = raw.get("target")
        if declared_target is not None:
            declared_target = _text(declared_target, f"{name}.target")
            declared_targets.add(declared_target)

        weight = float(raw.get("weight", 1.0))
        site_weight = float(raw.get("site_weight", 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"{name}.weight must be finite and non-negative.")
        if not np.isfinite(site_weight) or site_weight < 0:
            raise ValueError(f"{name}.site_weight must be finite and non-negative.")

        chronology_group = raw.get("chronology_group")
        if chronology_group is not None:
            chronology_group = _text(chronology_group, f"{name}.chronology_group")

        # Preserve source/method/sample_id/age_unit and arbitrary provenance,
        # rather than relying on a whitelist that silently discards lineage.
        record = deepcopy(dict(raw))
        record.update(
            {
                "_multi_proxy_prepared": _PREPARATION_VERSION,
                "record_id": record_id,
                "site_id": site_id,
                "proxy_id": proxy_id,
                "proxy_type": raw.get("proxy_type", "unspecified"),
                "target": declared_target,
                "unit": raw.get("unit"),
                "ages": ages.copy(),
                "values": values.copy(),
                "standardized_values": transformed,
                "measurement_error_raw": raw_errors,
                "measurement_error": errors,
                "age_ensembles": None if age_ensembles is None else age_ensembles.copy(),
                "age_member_ids": member_ids,
                "direction": direction,
                "standardization": standardization,
                "baseline_period": baseline_period,
                "location": standardized.location,
                "scale": standardized.scale,
                "weight": weight,
                "site_weight": site_weight,
                "chronology_group": chronology_group,
                "age_direction": age_direction,
                "n_valid": int(np.isfinite(values).sum()),
            }
        )
        prepared.append(record)

    if len(declared_targets) > 1:
        raise ValueError(
            "All records in one synthesis must target the same construct; "
            f"found {sorted(declared_targets)}."
        )
    inferred_target = next(iter(declared_targets), None)
    if target is not None:
        target = _text(target, "target")
        if inferred_target is not None and inferred_target != target:
            raise ValueError(
                f"target={target!r} conflicts with record target {inferred_target!r}."
            )
    else:
        target = inferred_target
    for record in prepared:
        if record["target"] is not None and record["target"] != target:
            raise ValueError("All record targets must match the requested target.")
        record["target"] = target

    methods = {record["standardization"] for record in prepared}
    if "none" in methods and len(methods) > 1:
        raise ValueError(
            "Cannot mix raw standardization='none' records with standardized "
            "zscore/robust records; choose a common analysis scale first."
        )
    analysis_unit = "dimensionless"
    if methods == {"none"}:
        if any(
            record["unit"] is not None and not isinstance(record["unit"], str)
            for record in prepared
        ):
            raise ValueError(
                "Raw record unit labels must be strings, not numeric/missing markers."
            )
        units = [
            None if record["unit"] is None else record["unit"].strip()
            for record in prepared
        ]
        if len(prepared) > 1 and (not all(units) or len(set(units)) != 1):
            raise ValueError(
                "Records using standardization='none' must all declare the same "
                "unit, or be calibrated to a common unit before synthesis."
            )
        analysis_unit = units[0] or None

    chronology_groups = _chronology_groups(prepared)

    return {
        "_multi_proxy_prepared": _PREPARATION_VERSION,
        "records": prepared,
        "target": target,
        "site_ids": sorted({record["site_id"] for record in prepared}),
        "proxy_ids": sorted({record["proxy_id"] for record in prepared}),
        "analysis_scale": "raw" if methods == {"none"} else "standardized",
        "analysis_unit": analysis_unit,
        "chronology_groups": _chronology_metadata(chronology_groups),
        "chronology_assumptions": list(_CHRONOLOGY_ASSUMPTIONS),
        "independent_chronology_records": [
            record["record_id"] for record in prepared if record["chronology_group"] is None
        ],
        "raw_values_preserved": True,
    }


def _record_weight(record: Mapping, weighting: str) -> float:
    if weighting == "equal":
        return float(record["weight"])
    if weighting != "precision":
        raise ValueError("weighting must be 'equal' or 'precision'.")
    if record["weight"] == 0:
        return 0.0
    errors = np.asarray(record["measurement_error"], dtype=float)
    if np.any(~np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError(
            f"Record {record['record_id']!r}: weighting='precision' requires "
            "explicit, strictly positive measurement errors; missing or zero "
            "errors do not provide known precision. Use weighting='equal' instead."
        )
    precision = 1.0 / float(np.median(errors) ** 2)
    return float(record["weight"]) * precision


def _correlation_matrix(
    proxy_ids: Sequence[str],
    measurement_correlation,
) -> np.ndarray:
    matrix = np.eye(len(proxy_ids), dtype=float)
    if measurement_correlation is None:
        return matrix
    if isinstance(measurement_correlation, np.ndarray):
        supplied = as_float_array(
            measurement_correlation,
            "measurement_correlation",
            ndim=2,
            allow_nan=False,
        )
        if supplied.shape != matrix.shape:
            raise ValueError(
                "measurement_correlation array must have one row/column per proxy."
            )
        matrix = supplied.copy()
    elif isinstance(measurement_correlation, Mapping):
        positions = {proxy_id: i for i, proxy_id in enumerate(proxy_ids)}
        for key, value in measurement_correlation.items():
            if isinstance(key, str) and isinstance(value, Mapping):
                left = key
                pairs = value.items()
            elif isinstance(key, (tuple, list)) and len(key) == 2:
                left, right = key
                pairs = [(right, value)]
            else:
                raise ValueError(
                    "measurement_correlation must map (proxy_a, proxy_b) pairs "
                    "or nested proxy names to correlations."
                )
            if left not in positions:
                raise ValueError(f"Unknown proxy in measurement_correlation: {left!r}.")
            for right, correlation in pairs:
                if right not in positions:
                    raise ValueError(
                        f"Unknown proxy in measurement_correlation: {right!r}."
                    )
                i, j = positions[left], positions[right]
                value_float = float(correlation)
                if not np.isfinite(value_float) or not -1 <= value_float <= 1:
                    raise ValueError("Proxy correlations must lie in [-1, 1].")
                matrix[i, j] = value_float
                matrix[j, i] = value_float
    else:
        raise ValueError("measurement_correlation must be an array or mapping.")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("measurement_correlation must be symmetric.")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise ValueError("measurement_correlation must have a unit diagonal.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(eigenvalues) < -1e-8:
        raise ValueError("measurement_correlation must be positive semi-definite.")
    return matrix


def _apply_measurement_noise(
    curves: np.ndarray,
    error_curves: np.ndarray,
    records: Sequence[Mapping],
    correlation_matrix: np.ndarray,
    correlation_proxy_ids: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    noisy = curves.copy()
    by_site = defaultdict(list)
    positions = {proxy_id: i for i, proxy_id in enumerate(correlation_proxy_ids)}
    for index, record in enumerate(records):
        by_site[record["site_id"]].append(index)

    for indices in by_site.values():
        proxy_ids = [records[index]["proxy_id"] for index in indices]
        try:
            proxy_positions = [positions[proxy_id] for proxy_id in proxy_ids]
        except KeyError as exc:  # pragma: no cover - guarded by preparation
            raise ValueError(f"Unknown proxy in measurement correlation: {exc.args[0]}") from exc
        correlation = correlation_matrix[np.ix_(proxy_positions, proxy_positions)]
        for time_index in range(curves.shape[1]):
            active = [
                position
                for position, index in enumerate(indices)
                if np.isfinite(noisy[index, time_index])
                and np.isfinite(error_curves[index, time_index])
                and error_curves[index, time_index] > 0
            ]
            if not active:
                continue
            errors = np.asarray(
                [error_curves[indices[position], time_index] for position in active],
                dtype=float,
            )
            corr = correlation[np.ix_(active, active)]
            covariance = corr * np.outer(errors, errors)
            draws = rng.multivariate_normal(
                np.zeros(len(active)), covariance, check_valid="raise"
            )
            for draw, position in zip(draws, active):
                noisy[indices[position], time_index] += draw
    return noisy


def _aligned_curves(
    records: Sequence[Mapping],
    time_grid: np.ndarray,
    *,
    age_member_indices: Optional[Mapping[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    curves = np.full((len(records), len(time_grid)), np.nan, dtype=float)
    error_curves = np.full_like(curves, np.nan)
    for index, record in enumerate(records):
        age_values = record["ages"]
        age_ensembles = record["age_ensembles"]
        if age_member_indices is not None and record["record_id"] in age_member_indices:
            age_values = age_ensembles[age_member_indices[record["record_id"]]]
        curves[index] = interpolate_no_extrapolation(
            age_values,
            record["standardized_values"],
            time_grid,
            name=record["record_id"],
        )
        if np.any(record["measurement_error"] > 0):
            error_curves[index] = interpolate_no_extrapolation(
                age_values,
                record["measurement_error"],
                time_grid,
                name=f"{record['record_id']}.measurement_error",
            )
        else:
            error_curves[index] = 0.0
    return curves, error_curves


def _group_indices(records: Sequence[Mapping]) -> Dict[str, List[int]]:
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["site_id"]].append(index)
    return dict(groups)


def _site_composites(
    curves: np.ndarray,
    records: Sequence[Mapping],
    weighting: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    groups = _group_indices(records)
    site_curves = []
    site_weights = []
    site_ids = []
    for site_id, indices in groups.items():
        proxy_weights = np.asarray(
            [_record_weight(records[index], weighting) for index in indices], dtype=float
        )
        if not np.any(proxy_weights > 0):
            raise ValueError(f"Site {site_id!r} has no positive proxy weight.")
        site_curves.append(weighted_nanmean(curves[indices], proxy_weights))
        declared = {float(records[index]["site_weight"]) for index in indices}
        if len(declared) != 1:
            raise ValueError(
                f"All proxy records from site {site_id!r} must share site_weight."
            )
        site_weights.append(next(iter(declared)))
        site_ids.append(site_id)
    site_weights_array = np.asarray(site_weights, dtype=float)
    if np.any(site_weights_array < 0) or not np.any(site_weights_array > 0):
        raise ValueError("At least one site_weight must be positive.")
    return np.asarray(site_curves), site_weights_array, site_ids


def proxy_concordance(
    proxy_curves: Mapping[str, np.ndarray],
    *,
    min_overlap: int = 3,
) -> Dict:
    """Summarize pairwise agreement among already aligned proxy curves.

    Correlations are descriptive and use only overlapping finite bins.  The
    function does not treat agreement as proof that proxies measure the same
    process; metadata and proxy-specific mechanisms remain necessary.
    """
    if min_overlap < 2:
        raise ValueError("min_overlap must be at least 2.")
    names = list(proxy_curves)
    if len(names) < 2:
        return {
            "pairs": {},
            "mean_correlation": np.nan,
            "mean_sign_agreement": np.nan,
            "n_proxies": len(names),
        }
    arrays = {
        name: as_float_array(values, f"proxy_curves[{name}]", ndim=1)
        for name, values in proxy_curves.items()
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("All proxy curves must share the same time-grid length.")

    pairs = {}
    correlations = []
    sign_agreements = []
    for left, right in combinations(names, 2):
        a, b = arrays[left], arrays[right]
        mask = np.isfinite(a) & np.isfinite(b)
        n_overlap = int(mask.sum())
        key = f"{left}__vs__{right}"
        if n_overlap < min_overlap:
            pairs[key] = {
                "proxy_a": left,
                "proxy_b": right,
                "n_overlap": n_overlap,
                "correlation": np.nan,
                "sign_agreement": np.nan,
            }
            continue
        a_valid, b_valid = a[mask], b[mask]
        if np.std(a_valid) == 0 or np.std(b_valid) == 0:
            correlation = np.nan
        else:
            correlation = float(np.corrcoef(a_valid, b_valid)[0, 1])
        sign_mask = (a_valid != 0) & (b_valid != 0)
        sign_agreement = (
            float(np.mean(np.sign(a_valid[sign_mask]) == np.sign(b_valid[sign_mask])))
            if np.any(sign_mask)
            else np.nan
        )
        pairs[key] = {
            "proxy_a": left,
            "proxy_b": right,
            "n_overlap": n_overlap,
            "correlation": correlation,
            "sign_agreement": sign_agreement,
        }
        if np.isfinite(correlation):
            correlations.append(correlation)
        if np.isfinite(sign_agreement):
            sign_agreements.append(sign_agreement)
    return {
        "pairs": pairs,
        "mean_correlation": float(np.mean(correlations)) if correlations else np.nan,
        "mean_sign_agreement": (
            float(np.mean(sign_agreements)) if sign_agreements else np.nan
        ),
        "n_proxies": len(names),
    }


def multi_proxy_synthesis(
    proxy_records: Union[Mapping, Iterable[Mapping]],
    time_grid: np.ndarray,
    *,
    target: Optional[str] = None,
    default_standardization: str = "zscore",
    weighting: str = "equal",
    measurement_correlation=None,
    n_members: int = 500,
    bootstrap_sites: bool = True,
    propagate_age: bool = True,
    propagate_measurement: bool = True,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """Synthesize multiple proxy types without double-counting sites.

    ``proxy_records`` is a list of record dictionaries.  Records are first
    transformed to a common oriented target, interpolated independently, and
    combined within site.  Site composites are then combined regionally.  A
    site bootstrap resamples all proxies from a site together, preserving the
    main within-archive dependence. Sites with zero ``site_weight`` are excluded
    from the bootstrap pool. Precision weighting requires declared positive
    errors; absent or zero errors are not treated as known precision.
    ``measurement_correlation`` optionally
    supplies a proxy-ID correlation matrix or pair mapping for standardized
    measurement errors within a site.

    Chronology groups are sampled independently with replacement. Only an
    explicitly shared ``chronology_group`` couples records, by matching
    ``age_member_ids`` (not row numbers). Returned member indices refer to the
    original rows and allow the joint draws to be audited. Coverage counts are
    available sites per proxy/time bin at nominal ages, before weights and
    resampling; ``valid_members_per_time`` reports regional ensemble support.

    This is a transparent common-target synthesis.  It requires the caller to
    justify why the proxies share a target and does not estimate a latent
    process, proxy productivity, or causal effect.
    """
    grid = as_float_array(time_grid, "time_grid", ndim=1, allow_nan=False)
    if (
        isinstance(n_members, (bool, np.bool_))
        or not isinstance(n_members, (int, np.integer))
        or n_members < 1
    ):
        raise ValueError("n_members must be a positive integer.")
    prepared_info = prepare_proxy_records(
        proxy_records,
        target=target,
        default_standardization=default_standardization,
    )
    records = prepared_info["records"]
    if weighting not in {"equal", "precision"}:
        raise ValueError("weighting must be 'equal' or 'precision'.")
    all_proxy_ids = prepared_info["proxy_ids"]
    if len(all_proxy_ids) > 1 and prepared_info["target"] is None:
        raise ValueError(
            "multi_proxy_synthesis requires an explicit common target when "
            "more than one proxy_id is supplied."
        )
    correlation_matrix = _correlation_matrix(all_proxy_ids, measurement_correlation)

    has_age_uncertainty = propagate_age and any(
        record["age_ensembles"] is not None for record in records
    )
    has_measurement_uncertainty = propagate_measurement and any(
        np.any(record["measurement_error"] > 0) for record in records
    )
    stochastic = bootstrap_sites or has_age_uncertainty or has_measurement_uncertainty
    run_members = n_members if stochastic else 1
    rng = get_rng(random_state)

    chronology_groups = _chronology_groups(records)
    chronology_member_indices = {}
    if propagate_age:
        # One independent draw stream per group; the same canonical member is
        # mapped back by identity for every record in a shared group. Drawing
        # whole trajectories also preserves within-record age covariance.
        for group in chronology_groups:
            if group["n_available_members"]:
                selected = rng.integers(
                    0, group["n_available_members"], size=run_members
                )
                for record_id, rows in group["row_indices"].items():
                    chronology_member_indices[record_id] = rows[selected]

    ensembles = []
    for member_index in range(run_members):
        curves, error_curves = _aligned_curves(
            records,
            grid,
            age_member_indices={
                record_id: int(rows[member_index])
                for record_id, rows in chronology_member_indices.items()
            },
        )
        if not propagate_measurement:
            error_curves = np.zeros_like(error_curves)
        curves = _apply_measurement_noise(
            curves, error_curves, records, correlation_matrix, all_proxy_ids, rng
        )
        site_curves, site_weights, _ = _site_composites(curves, records, weighting)
        if bootstrap_sites:
            eligible = np.flatnonzero(site_weights > 0)
            selected = eligible[rng.integers(0, len(eligible), size=len(eligible))]
            regional = weighted_nanmean(site_curves[selected], site_weights[selected])
        else:
            regional = weighted_nanmean(site_curves, site_weights)
        ensembles.append(regional)
    ensembles_array = np.asarray(ensembles, dtype=float)

    # Build deterministic proxy-level curves for diagnostics.  These are not
    # additional independent evidence; they show coverage and disagreement.
    diagnostic_curves, _ = _aligned_curves(records, grid)
    by_proxy = defaultdict(list)
    by_proxy_weights = defaultdict(list)
    for index, record in enumerate(records):
        by_proxy[record["proxy_id"]].append(diagnostic_curves[index])
        by_proxy_weights[record["proxy_id"]].append(
            record["site_weight"] * _record_weight(record, weighting)
        )
    proxy_curves = {
        proxy_id: (
            weighted_nanmean(
                np.asarray(by_proxy[proxy_id]),
                np.asarray(by_proxy_weights[proxy_id], dtype=float),
            )
            if np.any(np.asarray(by_proxy_weights[proxy_id]) > 0)
            else np.full(len(grid), np.nan)
        )
        for proxy_id in by_proxy
    }
    concordance = proxy_concordance(proxy_curves)
    proxy_coverage = {
        proxy_id: np.sum(np.isfinite(np.asarray(curves)), axis=0).astype(int)
        for proxy_id, curves in by_proxy.items()
    }
    effective_proxy_count = np.sum(
        np.isfinite(np.asarray(list(proxy_curves.values()))), axis=0
    )
    diagnostic_site_curves, diagnostic_site_weights, _ = _site_composites(
        diagnostic_curves, records, weighting
    )
    effective_site_count = np.sum(
        np.isfinite(diagnostic_site_curves) & (diagnostic_site_weights[:, None] > 0),
        axis=0,
    )
    valid_members_per_time = np.sum(np.isfinite(ensembles_array), axis=0)

    if ensembles_array.shape[0] > 1:
        bands = np.full((3, ensembles_array.shape[1]), np.nan, dtype=float)
        for time_index in range(ensembles_array.shape[1]):
            finite = ensembles_array[:, time_index]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                bands[:, time_index] = np.percentile(finite, [5, 50, 95])
        lower, median, upper = bands
        band = {
            "lower": lower,
            "median": median,
            "upper": upper,
            "percentiles": (5, 50, 95),
            "interval_type": "ensemble_percentile",
        }
        composite = median
    else:
        band = None
        composite = ensembles_array[0]

    data_fields = {
        "ages", "values", "standardized_values", "measurement_error_raw",
        "measurement_error", "error", "proxy_error", "age_ensembles",
        "_multi_proxy_prepared",
    }
    metadata = [
        {key: deepcopy(value) for key, value in record.items() if key not in data_fields}
        for record in records
    ]
    uncertainty_layers = []
    if has_age_uncertainty:
        uncertainty_layers.append("chronology")
    if has_measurement_uncertainty:
        uncertainty_layers.append("measurement")
    if bootstrap_sites:
        uncertainty_layers.append("site_bootstrap")

    return {
        "composite": composite,
        "ensembles": ensembles_array if ensembles_array.shape[0] > 1 else None,
        "uncertainty_band": band,
        "time_grid": grid,
        "target": prepared_info["target"],
        "analysis_scale": prepared_info["analysis_scale"],
        "analysis_unit": prepared_info["analysis_unit"],
        "method": "site_clustered_proxy_synthesis",
        "weighting": weighting,
        "n_members": int(ensembles_array.shape[0]),
        "n_records": len(records),
        "n_sites": len(prepared_info["site_ids"]),
        "n_proxies": len(all_proxy_ids),
        "site_ids": prepared_info["site_ids"],
        "proxy_ids": all_proxy_ids,
        "record_metadata": metadata,
        "proxy_curves": proxy_curves,
        "proxy_coverage": proxy_coverage,
        "coverage_basis": "available_sites_at_nominal_ages_before_weighting",
        "effective_proxy_count": effective_proxy_count,
        "effective_site_count": effective_site_count,
        "valid_members_per_time": valid_members_per_time,
        "proxy_concordance": concordance,
        "uncertainty_layers": uncertainty_layers,
        "measurement_error_model": (
            "conditional Gaussian errors on the aligned grid; independent across time bins; "
            "declared cross-proxy correlation within each site"
            if has_measurement_uncertainty else "none propagated"
        ),
        "stochastic": bool(stochastic),
        "chronology_groups": _chronology_metadata(chronology_groups),
        "chronology_member_indices": chronology_member_indices,
        "chronology_assumptions": prepared_info["chronology_assumptions"],
        "independent_chronology_records": prepared_info["independent_chronology_records"],
        "bootstrap_unit": "site_cluster",
        "random_state": random_state if isinstance(random_state, int) else None,
        "raw_values_preserved": True,
        "extrapolation": "nan",
    }


def leave_one_proxy_out(
    proxy_records: Union[Mapping, Iterable[Mapping]],
    time_grid: np.ndarray,
    **kwargs,
) -> Dict:
    """Assess sensitivity to excluding each proxy type.

    The default sensitivity run disables stochastic propagation so differences
    reflect proxy composition rather than different random draws.  Pass
    ``bootstrap_sites=True`` or uncertainty flags explicitly when a stochastic
    leave-one-proxy-out distribution is desired. Such runs honor ``n_members``
    (the synthesis default is 500). ``full`` retains the legacy composite
    array; ``full_result`` and each exclusion also expose ensembles, percentile
    bands, and actual member counts. Stochastic ``difference`` is a difference
    of marginal medians, not a paired difference distribution or its interval.
    If an exclusion removes the last uncertainty source, its distribution is
    deterministic (repeated members, flagged ``distribution_degenerate``), not
    an invented uncertainty layer.
    """
    raw_records = _record_iter(proxy_records)
    prepared = prepare_proxy_records(
        raw_records,
        target=kwargs.get("target"),
        default_standardization=kwargs.get("default_standardization", "zscore"),
    )
    proxy_ids = prepared["proxy_ids"]
    if len(proxy_ids) < 2:
        raise ValueError("leave_one_proxy_out needs at least two proxy types.")
    raw_records = prepared["records"]

    base_kwargs = dict(kwargs)
    base_kwargs.setdefault("bootstrap_sites", False)
    base_kwargs.setdefault("propagate_age", False)
    base_kwargs.setdefault("propagate_measurement", False)
    correlation_spec = base_kwargs.get("measurement_correlation")
    full_correlation = _correlation_matrix(proxy_ids, correlation_spec)
    proxy_positions = {proxy_id: i for i, proxy_id in enumerate(proxy_ids)}
    full = multi_proxy_synthesis(raw_records, time_grid, **base_kwargs)
    exclusions = {}
    for excluded in proxy_ids:
        subset = [record for record in raw_records if record["proxy_id"] != excluded]
        subset_kwargs = dict(base_kwargs)
        if correlation_spec is not None:
            allowed = [proxy_id for proxy_id in proxy_ids if proxy_id != excluded]
            subset_kwargs["measurement_correlation"] = {
                (left, right): full_correlation[
                    proxy_positions[left], proxy_positions[right]
                ]
                for left, right in combinations(allowed, 2)
                if full_correlation[proxy_positions[left], proxy_positions[right]] != 0
            }
        result = multi_proxy_synthesis(subset, time_grid, **subset_kwargs)
        if full["n_members"] > 1 and result["n_members"] == 1:
            # A deterministic exclusion is a point-mass distribution, not a
            # reason to discard the requested stochastic LOPO member count.
            composite = result["composite"]
            result["ensembles"] = np.repeat(
                composite[None, :], full["n_members"], axis=0
            )
            result["n_members"] = full["n_members"]
            result["valid_members_per_time"] = (
                np.isfinite(composite).astype(int) * full["n_members"]
            )
            result["uncertainty_band"] = {
                "lower": composite.copy(),
                "median": composite.copy(),
                "upper": composite.copy(),
                "percentiles": (5, 50, 95),
                "interval_type": "ensemble_percentile",
            }
            result["distribution_degenerate"] = True
        difference = result["composite"] - full["composite"]
        finite = np.isfinite(difference)
        exclusions[excluded] = {
            **result,
            "difference": difference,
            "rmse": (
                float(np.sqrt(np.mean(difference[finite] ** 2)))
                if np.any(finite)
                else np.nan
            ),
            "mean_abs_difference": (
                float(np.mean(np.abs(difference[finite]))) if np.any(finite) else np.nan
            ),
            "n_proxies_remaining": len(proxy_ids) - 1,
        }
    return {
        "full": full["composite"],
        "full_result": full,
        "n_members": full["n_members"],
        "time_grid": as_float_array(time_grid, "time_grid", ndim=1, allow_nan=False),
        "by_excluded_proxy": exclusions,
        "proxy_ids": proxy_ids,
        "method": "leave_one_proxy_out",
        "stochastic": full["stochastic"],
        "difference_definition": "difference_of_marginal_composites_not_paired_draws",
    }

"""Private validation and numerical helpers shared by the skill modules.

These helpers keep the public modules small and, more importantly, make the
handling of missing values, irregular ages, weights, and random state explicit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def as_float_array(
    values,
    name: str,
    *,
    ndim: Optional[int] = None,
    allow_nan: bool = True,
    min_size: int = 1,
) -> np.ndarray:
    """Convert an input to a float array and validate its basic shape."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是可转换为数值的数组。") from exc
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} 必须是 {ndim} 维数组，实际为 {arr.ndim} 维。")
    if arr.size < min_size:
        raise ValueError(f"{name} 至少需要 {min_size} 个元素。")
    if not allow_nan and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含有 NaN 或无穷值。")
    return arr


def validate_same_length(*named_arrays: Tuple[str, np.ndarray]) -> None:
    """Raise a useful error when one-dimensional inputs have different sizes."""
    lengths = {name: np.asarray(value).shape[0] for name, value in named_arrays}
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"输入长度必须一致（{detail}）。")


def prepare_interpolation(
    ages,
    values,
    *,
    name: str = "series",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort an age/value series, remove invalid rows, and average duplicate ages."""
    x = as_float_array(ages, f"{name}.ages", ndim=1, allow_nan=False)
    y = as_float_array(values, f"{name}.values", ndim=1)
    validate_same_length((f"{name}.ages", x), (f"{name}.values", y))

    mask = np.isfinite(y)
    if not np.any(mask):
        raise ValueError(f"{name} 没有可用于插值的有限值。")
    x, y = x[mask], y[mask]
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        unique_y = np.full(len(unique_x), np.nan)
        for i in range(len(unique_x)):
            group = y[inverse == i]
            unique_y[i] = np.nanmean(group)
        y = unique_y
        x = unique_x
    if len(x) < 2:
        raise ValueError(f"{name} 至少需要两个不同年龄点才能插值。")
    return x, y


def interpolate_no_extrapolation(ages, values, time_grid, *, name: str = "series") -> np.ndarray:
    """Interpolate in age space and return NaN outside the observed range.

    The grid may be increasing or decreasing; the returned array preserves its
    original order.  Silent edge-value extrapolation is deliberately avoided.
    """
    x, y = prepare_interpolation(ages, values, name=name)
    grid = as_float_array(time_grid, "time_grid", ndim=1, allow_nan=False)
    order = np.argsort(grid, kind="mergesort")
    sorted_grid = grid[order]
    out_sorted = np.interp(sorted_grid, x, y, left=np.nan, right=np.nan)
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def normalize_weights(weights, n: int) -> np.ndarray:
    """Validate non-negative weights and normalize them to sum to one."""
    if weights is None:
        return np.full(n, 1.0 / n)
    w = as_float_array(weights, "weights", ndim=1, allow_nan=False)
    if len(w) != n:
        raise ValueError(f"weights 长度必须为 {n}，实际为 {len(w)}。")
    if np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("weights 必须非负且至少有一个正值。")
    return w / w.sum()


def weighted_nanmean(values, weights=None, axis: int = 0) -> np.ndarray:
    """Weighted mean that renormalizes weights independently at each position."""
    arr = as_float_array(values, "values", ndim=2)
    w = normalize_weights(weights, arr.shape[axis])
    if axis != 0:
        arr = np.moveaxis(arr, axis, 0)
    valid = np.isfinite(arr)
    numerator = np.nansum(np.where(valid, arr * w[:, None], 0.0), axis=0)
    denominator = np.sum(np.where(valid, w[:, None], 0.0), axis=0)
    return np.divide(numerator, denominator, out=np.full(arr.shape[1], np.nan), where=denominator > 0)


def get_rng(random_state=None) -> np.random.Generator:
    """Return a local generator without mutating NumPy's global RNG state."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def validate_age_ensembles(age_ensembles, n_samples: int, name: str = "age_ensembles") -> np.ndarray:
    """Validate a shared age ensemble with shape ``(members, samples)``."""
    arr = as_float_array(age_ensembles, name, ndim=2, allow_nan=False)
    if arr.shape[1] != n_samples:
        raise ValueError(f"{name} 的第二维必须为 {n_samples}，实际为 {arr.shape[1]}。")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} 至少需要一个成员。")
    return arr


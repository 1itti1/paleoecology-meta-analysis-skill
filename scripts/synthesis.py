"""
古生态学原生综合方法模块。
涵盖 SCC/DCC/CPS/PAI/GAM 五方法、蒙特卡洛集合传播、LOESS 可视化、多方法交叉验证。

文献来源：
- Kaufman 2020 [2]: SCC/DCC/CPS/PAI/GAM 五方法、500 成员集合策略
- Izdebski 2022 [1]: z-score + Bootstrap 范式
- Marlon 2008 [3]: LOESS 平滑
- Cleveland & Devlin 1988 [10]: LOESS 算法
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    from ._utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        normalize_weights,
        weighted_nanmean,
        validate_age_ensembles,
    )
except ImportError:  # pragma: no cover - supports direct script imports
    from _utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        normalize_weights,
        weighted_nanmean,
        validate_age_ensembles,
    )


def scc_composite(
    calibrated_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """Kaufman 2020 Standard Calibrated Composite (SCC)。

    已校准代理（温度、降水等物理量）的面积加权直接合成。

    Parameters
    ----------
    calibrated_values : np.ndarray
        已校准代理值 (n_sites, n_timebins)。
    weights : np.ndarray, optional
        面积权重 (n_sites,)，默认等权。

    Returns
    -------
    Dict
        {'composite': np.ndarray (n_timebins,), 'n_sites': int, 'method': 'SCC'}
    """
    values = as_float_array(calibrated_values, 'calibrated_values', ndim=2)
    weights = normalize_weights(weights, values.shape[0])
    composite = weighted_nanmean(values, weights)
    return {
        'composite': composite,
        'n_sites': values.shape[0],
        'method': 'SCC',
        'weights': weights,
    }


def dcc_composite(
    calibrated_values: np.ndarray,
    time_varying_calib_func: Callable,
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """Kaufman 2020 Dynamic Calibrated Composite (DCC)。

    动态校准后合成，适用于校准关系非平稳的场景。

    Parameters
    ----------
    calibrated_values : np.ndarray
        原始代理值 (n_sites, n_timebins)。
    time_varying_calib_func : callable
        时变校准函数，接受 (value, time) 返回校准值。
    weights : np.ndarray, optional
        面积权重 (n_sites,)。

    Returns
    -------
    Dict
        {'composite': np.ndarray, 'n_sites': int, 'method': 'DCC'}
    """
    values = as_float_array(calibrated_values, 'calibrated_values', ndim=2)
    n_sites, n_bins = values.shape
    weights = normalize_weights(weights, n_sites)
    calibrated = np.full_like(values, np.nan)
    for t in range(n_bins):
        for s in range(n_sites):
            calibrated[s, t] = time_varying_calib_func(values[s, t], t)

    composite = weighted_nanmean(calibrated, weights)
    return {
        'composite': composite,
        'n_sites': n_sites,
        'method': 'DCC',
        'weights': weights,
    }


def cps_composite(
    z_scores: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
) -> Dict:
    """Kaufman 2020 Composite Plus Scale (CPS)。

    未校准代理（原始 z-score）合成后以研究时段均值和标准差标定。

    Parameters
    ----------
    z_scores : np.ndarray
        z-score 标准化值 (n_sites, n_timebins)。
    baseline_mean : float
        研究时段均值。
    baseline_std : float
        研究时段标准差。

    Returns
    -------
    Dict
        {'composite': np.ndarray, 'baseline_mean': float, 'baseline_std': float, 'method': 'CPS'}
    """
    values = as_float_array(z_scores, 'z_scores', ndim=2)
    if not np.isfinite(baseline_std) or baseline_std < 0:
        raise ValueError('baseline_std 必须为非负有限值。')
    composite_z = np.nanmean(values, axis=0)
    composite = composite_z * baseline_std + baseline_mean
    return {
        'composite': composite,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'method': 'CPS',
    }


def pai_composite(
    z_scores: np.ndarray,
    direction: str = 'positive',
) -> Dict:
    """Kaufman 2020 Pairwise Comparison Index (PAI)。

    两两比较各点位变化方向，合成方向一致性指数。
    适用于未校准代理，关注变化方向而非幅度。

    Parameters
    ----------
    z_scores : np.ndarray
        z-score 标准化值 (n_sites, n_timebins)。
    direction : str, optional
        关注方向：'positive' (增加) 或 'negative' (减少)。

    Returns
    -------
    Dict
        {'composite': np.ndarray (n_timebins,), 'agreement_ratio': np.ndarray, 'method': 'PAI'}
    """
    values = as_float_array(z_scores, 'z_scores', ndim=2)
    if direction not in {'positive', 'negative'}:
        raise ValueError("direction 须为 'positive' 或 'negative'。")
    n_sites, n_bins = values.shape
    diff = np.full_like(values, np.nan)
    diff[:, 0] = 0.0
    if n_bins > 1:
        diff[:, 1:] = np.diff(values, axis=1)
    valid = np.isfinite(diff)
    signs = np.where(direction == 'positive', diff > 0, diff < 0)
    agreement_ratio = np.divide(
        np.sum(np.where(valid, signs, False), axis=0),
        np.sum(valid, axis=0),
        out=np.full(n_bins, np.nan, dtype=float),
        where=np.sum(valid, axis=0) > 0,
    )
    # PAI 指数 = 2 * agreement - 1；无有效样本的位置保持 NaN
    composite = 2 * agreement_ratio - 1

    return {
        'composite': composite,
        'agreement_ratio': agreement_ratio,
        'method': 'PAI',
    }


def gam_composite(
    time: np.ndarray,
    values: np.ndarray,
    n_splines: int = 20,
) -> Dict:
    """Kaufman 2020 PyGAM 合成：拟合合成时序的非线性趋势。

    Parameters
    ----------
    time : np.ndarray
        时间轴 (n_points,)。
    values : np.ndarray
        合成值 (n_points,)。
    n_splines : int, optional
        样条节点数，默认 20 (Kaufman 2020 拟合 6000 年)。
        准则：每 200-300 年一个节点。

    Returns
    -------
    Dict
        {'gam': LinearGAM, 'predicted': np.ndarray, 'n_splines': int, 'method': 'GAM'}
    """
    time_arr = as_float_array(time, 'time', ndim=1, allow_nan=False, min_size=4)
    values_arr = as_float_array(values, 'values', ndim=1)
    if len(time_arr) != len(values_arr):
        raise ValueError('time 和 values 长度必须一致。')
    valid = np.isfinite(values_arr)
    if valid.sum() < max(4, n_splines):
        raise ValueError('GAM 有效样本数不足以支持当前 n_splines。')
    if n_splines < 4:
        raise ValueError('n_splines 至少为 4。')
    try:
        from pygam import LinearGAM, s
    except ImportError as exc:
        raise ImportError(
            'gam_composite 需要可选依赖 PyGAM；请安装 requirements-optional.txt，'
            '或选择 scc/cps/weighted_mean。'
        ) from exc

    gam = LinearGAM(s(0, n_splines=n_splines)).fit(
        time_arr[valid, None], values_arr[valid]
    )
    predicted = gam.predict(time_arr[:, None])

    return {
        'gam': gam,
        'predicted': predicted,
        'n_splines': n_splines,
        'method': 'GAM',
        'n_valid': int(valid.sum()),
    }


def monte_carlo_ensemble(
    records: np.ndarray,
    age_ensembles: np.ndarray,
    proxy_errors: np.ndarray,
    composite_func: Callable,
    n_members: int = 500,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """Kaufman 2020 500 成员集合策略：传播年龄+校准+采样三层不确定性。

    每个集合成员独立执行完整合成流程，整体采样年龄成员保持地层单调性。

    Parameters
    ----------
    records : np.ndarray
        各点位代理值 (n_sites, n_depths)。
    age_ensembles : np.ndarray
        外部年龄模型产生的年龄集合 (n_ensemble_members, n_depths)。
    proxy_errors : np.ndarray
        各点位代理校准误差 (n_sites,)。
    composite_func : callable
        合成函数，接受 (ages, values) 返回合成结果。
    n_members : int, optional
        集合成员数，默认 500 (Kaufman 2020)。

    Returns
    -------
    Dict
        {'ensembles': np.ndarray (n_members, n_timebins), 'n_members': int, 'method': 'Monte Carlo'}
    """
    values = as_float_array(records, 'records', ndim=2)
    errors = as_float_array(proxy_errors, 'proxy_errors', ndim=1, allow_nan=False)
    if len(errors) != values.shape[0]:
        raise ValueError('proxy_errors 必须按站点提供。')
    if np.any(errors < 0):
        raise ValueError('proxy_errors 必须非负。')
    age_arr = as_float_array(age_ensembles, 'age_ensembles', ndim=None, allow_nan=False)
    if age_arr.ndim not in {2, 3}:
        raise ValueError('age_ensembles 必须为 (members, samples) 或 (sites, members, samples)。')
    if age_arr.ndim == 2:
        validate_age_ensembles(age_arr, values.shape[1])
        n_pool = age_arr.shape[0]
    else:
        if age_arr.shape[0] != values.shape[0] or age_arr.shape[2] != values.shape[1]:
            raise ValueError('逐站点 age_ensembles 必须为 (sites, members, samples)。')
        n_pool = age_arr.shape[1]
    if n_members < 1:
        raise ValueError('n_members 至少为 1。')
    rng = get_rng(random_state)
    ensembles = []

    for i in range(n_members):
        if age_arr.ndim == 2:
            ages = age_arr[i % n_pool]
        else:
            ages = age_arr[:, i % n_pool, :]
        # 采样代理校准不确定性
        noise = rng.normal(0, errors[:, np.newaxis], size=values.shape)
        noisy_values = values + noise
        # 执行合成
        composite = composite_func(ages, noisy_values)
        ensembles.append(composite)

    return {
        'ensembles': np.array(ensembles),
        'n_members': n_members,
        'method': 'Monte Carlo',
        'random_state': random_state if isinstance(random_state, int) else None,
    }


def uncertainty_band(
    ensembles: np.ndarray,
    percentiles: Tuple[int, ...] = (5, 50, 95),
) -> Dict:
    """从集合中提取不确定性带。

    Parameters
    ----------
    ensembles : np.ndarray
        集合结果 (n_members, n_timebins)。
    percentiles : tuple, optional
        百分位数，默认 (5, 50, 95) 即 90% 不确定性带。

    Returns
    -------
    Dict
        {'lower': float, 'median': np.ndarray, 'upper': np.ndarray, 'percentiles': tuple}
    """
    values = as_float_array(ensembles, 'ensembles', ndim=2, allow_nan=True, min_size=2)
    if len(percentiles) != 3 or tuple(sorted(percentiles)) != tuple(percentiles):
        raise ValueError('percentiles 必须是三个按升序排列的数值。')
    if any(p < 0 or p > 100 for p in percentiles):
        raise ValueError('percentiles 必须位于 0 到 100 之间。')
    bands = np.full((len(percentiles), values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        column = values[:, j]
        finite = column[np.isfinite(column)]
        if len(finite):
            bands[:, j] = np.percentile(finite, percentiles)
    return {
        'lower': bands[0],
        'median': bands[1],
        'upper': bands[2],
        'percentiles': percentiles,
    }


def loess_trend(
    time: np.ndarray,
    values: np.ndarray,
    frac: float = 0.2,
) -> Dict:
    """Marlon 2008, Cleveland & Devlin 1988：LOESS 平滑可视化趋势。

    LOESS 在本方案中定位为可视化工具而非推断工具——用于展示合成曲线
    的趋势形态，不用于显著性判断。

    Parameters
    ----------
    time : np.ndarray
        时间轴 (n_points,)。
    values : np.ndarray
        合成值 (n_points,)。
    frac : float, optional
        平滑参数（窗口宽度占比），默认 0.2。
        较小保留更多细节但噪声大，较大更平滑但可能过度平滑。

    Returns
    -------
    Dict
        {'smoothed': np.ndarray (n_points, 2), 'frac': float, 'method': 'LOESS'}
    """
    time_arr = as_float_array(time, 'time', ndim=1, allow_nan=False)
    values_arr = as_float_array(values, 'values', ndim=1)
    if len(time_arr) != len(values_arr):
        raise ValueError('time 和 values 长度必须一致。')
    valid = np.isfinite(values_arr)
    if valid.sum() < 3:
        raise ValueError('LOESS 至少需要三个有限观测。')
    if not 0 < frac <= 1:
        raise ValueError('frac 必须位于 (0, 1]。')
    smoothed = lowess(values_arr[valid], time_arr[valid], frac=frac, return_sorted=True)
    return {
        'smoothed': smoothed,
        'frac': frac,
        'method': 'LOESS',
    }


def resample_and_composite(
    ages: np.ndarray,
    values: np.ndarray,
    time_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Kaufman 2020 辅助函数：重采样到统一网格并加权合成。

    Parameters
    ----------
    ages : np.ndarray
        年龄数组 (n_depths,)。
    values : np.ndarray
        代理值 (n_sites, n_depths)。
    time_grid : np.ndarray
        统一时间网格 (n_bins,)。
    weights : np.ndarray, optional
        面积权重 (n_sites,)。

    Returns
    -------
    np.ndarray
        合成结果 (n_bins,)。
    """
    values_arr = as_float_array(values, 'values', ndim=2)
    ages_arr = as_float_array(ages, 'ages', ndim=None, allow_nan=False)
    grid = as_float_array(time_grid, 'time_grid', ndim=1, allow_nan=False)
    n_sites = values_arr.shape[0]
    weights = normalize_weights(weights, n_sites)
    if ages_arr.ndim == 1:
        ages_arr = np.broadcast_to(ages_arr, (n_sites, ages_arr.shape[0]))
    if ages_arr.shape != values_arr.shape:
        raise ValueError('逐站点 ages 必须与 values 形状一致。')

    resampled = np.full((n_sites, len(grid)), np.nan)
    for s in range(n_sites):
        resampled[s] = interpolate_no_extrapolation(
            ages_arr[s], values_arr[s], grid, name=f'site_{s}'
        )
    return weighted_nanmean(resampled, weights)


def multi_method_cross_validation(
    records: np.ndarray,
    age_ensembles: np.ndarray,
    proxy_errors: np.ndarray,
    methods: Optional[List[str]] = None,
    n_members: int = 500,
    time_grid: Optional[np.ndarray] = None,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """Kaufman 2020 多方法集合策略：同时运行多种合成方法比较结果一致性。

    Parameters
    ----------
    records : np.ndarray
        各点位代理值 (n_sites, n_depths)。
    age_ensembles : np.ndarray
        年龄集合 (n_ensemble, n_depths)。
    proxy_errors : np.ndarray
        代理校准误差 (n_sites,)。
    methods : list, optional
        合成方法列表，默认 ['scc', 'gam', 'cps']。
    n_members : int, optional
        集合成员数，默认 500。
    time_grid : np.ndarray, optional
        统一时间网格。
    baseline_mean, baseline_std : float
        CPS 方法的基准均值和标准差。

    Returns
    -------
    Dict
        {'results': dict (method -> ensemble array), 'consistency': float, 'methods': list}
    """
    records_arr = as_float_array(records, 'records', ndim=2)
    age_arr = as_float_array(age_ensembles, 'age_ensembles', ndim=2, allow_nan=False)
    if age_arr.shape[1] != records_arr.shape[1]:
        raise ValueError('age_ensembles 与 records 的样本维度必须一致。')
    errors = as_float_array(proxy_errors, 'proxy_errors', ndim=1, allow_nan=False)
    if len(errors) != records_arr.shape[0]:
        raise ValueError('proxy_errors 必须按站点提供。')
    if methods is None:
        methods = ['scc', 'gam', 'cps']
    allowed = {'scc', 'gam', 'cps'}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f'不支持的 synthesis method: {sorted(unknown)}')
    if time_grid is None:
        time_grid = np.linspace(age_arr.min(), age_arr.max(), 200)
    time_grid = as_float_array(time_grid, 'time_grid', ndim=1, allow_nan=False)

    results = {}

    for method in methods:
        def comp_func(ages, vals, _method=method):
            ages_for_sites = ages if np.asarray(ages).ndim == 2 else np.asarray(ages)
            resampled = np.full((vals.shape[0], len(time_grid)), np.nan)
            for s in range(vals.shape[0]):
                age_s = ages_for_sites[s] if np.asarray(ages_for_sites).ndim == 2 else ages_for_sites
                resampled[s] = interpolate_no_extrapolation(
                    age_s, vals[s], time_grid, name=f'site_{s}'
                )
            if _method == 'scc':
                return scc_composite(resampled)['composite']
            if _method == 'cps':
                return cps_composite(resampled, baseline_mean, baseline_std)['composite']
            mean_curve = np.nanmean(resampled, axis=0)
            return gam_composite(time_grid, mean_curve)['predicted']

        ens = monte_carlo_ensemble(
            records_arr, age_arr, errors, comp_func, n_members,
            random_state=random_state,
        )
        results[method] = ens['ensembles']

    # 一致性：各方法中位数曲线的成对相关系数均值
    medians = {m: np.median(v, axis=0) for m, v in results.items()}
    method_list = list(medians.keys())
    corrs = []
    for i in range(len(method_list)):
        for j in range(i + 1, len(method_list)):
            r = np.corrcoef(medians[method_list[i]], medians[method_list[j]])[0, 1]
            corrs.append(r)
    consistency = np.mean(corrs) if corrs else np.nan

    return {
        'results': results,
        'consistency': consistency,
        'methods': list(methods),
        'n_members': n_members,
        'random_state': random_state if isinstance(random_state, int) else None,
    }

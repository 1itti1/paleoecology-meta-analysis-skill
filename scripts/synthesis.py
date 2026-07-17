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
from statsmodels.nonparametric.lowess import lowess


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
    if weights is None:
        weights = np.ones(calibrated_values.shape[0]) / calibrated_values.shape[0]
    else:
        weights = weights / weights.sum()

    composite = np.nansum(calibrated_values * weights[:, np.newaxis], axis=0)
    return {
        'composite': composite,
        'n_sites': calibrated_values.shape[0],
        'method': 'SCC',
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
    n_sites, n_bins = calibrated_values.shape
    if weights is None:
        weights = np.ones(n_sites) / n_sites
    else:
        weights = weights / weights.sum()

    calibrated = np.zeros_like(calibrated_values)
    for t in range(n_bins):
        for s in range(n_sites):
            calibrated[s, t] = time_varying_calib_func(calibrated_values[s, t], t)

    composite = np.nansum(calibrated * weights[:, np.newaxis], axis=0)
    return {
        'composite': composite,
        'n_sites': n_sites,
        'method': 'DCC',
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
    composite_z = np.nanmean(z_scores, axis=0)
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
    n_sites, n_bins = z_scores.shape
    # 相对于第一时点的变化方向
    diff = np.diff(z_scores, axis=1, prepend=z_scores[:, [0]])

    if direction == 'positive':
        signs = (diff > 0).astype(float)
    else:
        signs = (diff < 0).astype(float)

    # 方向一致性比例
    agreement_ratio = np.nanmean(signs, axis=0)
    # PAI 指数 = 2 * agreement - 1，范围 [-1, 1]
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
    from pygam import LinearGAM, s

    gam = LinearGAM(s(0, n_splines=n_splines)).fit(time, values)
    predicted = gam.predict(time)

    return {
        'gam': gam,
        'predicted': predicted,
        'n_splines': n_splines,
        'method': 'GAM',
    }


def monte_carlo_ensemble(
    records: np.ndarray,
    age_ensembles: np.ndarray,
    proxy_errors: np.ndarray,
    composite_func: Callable,
    n_members: int = 500,
) -> Dict:
    """Kaufman 2020 500 成员集合策略：传播年龄+校准+采样三层不确定性。

    每个集合成员独立执行完整合成流程，整体采样年龄成员保持地层单调性。

    Parameters
    ----------
    records : np.ndarray
        各点位代理值 (n_sites, n_depths)。
    age_ensembles : np.ndarray
        BAM/Bacon 年龄集合 (n_ensemble_members, n_depths)。
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
    ensembles = []
    n_ensemble_pool = age_ensembles.shape[0]

    for i in range(n_members):
        # 整体采样年龄成员（保持地层单调性）
        ages = age_ensembles[i % n_ensemble_pool]
        # 采样代理校准不确定性
        noise = np.random.normal(
            0, proxy_errors[:, np.newaxis], size=records.shape
        )
        values = records + noise
        # 执行合成
        composite = composite_func(ages, values)
        ensembles.append(composite)

    return {
        'ensembles': np.array(ensembles),
        'n_members': n_members,
        'method': 'Monte Carlo',
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
    bands = np.percentile(ensembles, percentiles, axis=0)
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
    smoothed = lowess(values, time, frac=frac, return_sorted=True)
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
    n_sites = values.shape[0]
    if weights is None:
        weights = np.ones(n_sites) / n_sites
    else:
        weights = weights / weights.sum()

    resampled = np.zeros((n_sites, len(time_grid)))
    for s in range(n_sites):
        resampled[s] = np.interp(time_grid, ages, values[s])

    return np.nansum(resampled * weights[:, np.newaxis], axis=0)


def multi_method_cross_validation(
    records: np.ndarray,
    age_ensembles: np.ndarray,
    proxy_errors: np.ndarray,
    methods: List[str] = ['scc', 'gam', 'cps'],
    n_members: int = 500,
    time_grid: Optional[np.ndarray] = None,
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0,
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
    if time_grid is None:
        time_grid = np.linspace(
            age_ensembles.min(), age_ensembles.max(), 200
        )

    results = {}

    for method in methods:
        if method == 'scc':
            def comp_func(ages, vals):
                resampled = np.zeros((vals.shape[0], len(time_grid)))
                for s in range(vals.shape[0]):
                    resampled[s] = np.interp(time_grid, ages, vals[s])
                return scc_composite(resampled)['composite']
        elif method == 'gam':
            def comp_func(ages, vals):
                resampled = np.zeros((vals.shape[0], len(time_grid)))
                for s in range(vals.shape[0]):
                    resampled[s] = np.interp(time_grid, ages, vals[s])
                composite = np.nanmean(resampled, axis=0)
                return composite
        elif method == 'cps':
            def comp_func(ages, vals):
                resampled = np.zeros((vals.shape[0], len(time_grid)))
                for s in range(vals.shape[0]):
                    resampled[s] = np.interp(time_grid, ages, vals[s])
                return cps_composite(resampled, baseline_mean, baseline_std)['composite']
        else:
            continue

        ens = monte_carlo_ensemble(
            records, age_ensembles, proxy_errors, comp_func, n_members
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
        'methods': methods,
        'n_members': n_members,
    }

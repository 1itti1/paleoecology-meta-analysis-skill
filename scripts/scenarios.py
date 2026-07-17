"""
三场景编排模块（多代理、多区域通用）。

将 preprocessing / synthesis / effect_size / validation 模块组装为完整工作流。
场景选择以数据结构为判据，不预设特定区域或代理类型。

支持两套并行通道：
- 分类群通道（花粉/硅藻/有孔虫等百分比数据）
- 连续值通道（δDwax/brGDGTs/粒度等数值序列）

文献来源：
- Kaufman 2020 [2]: 多方法集合框架、场景二方法链
- Izdebski 2022 [1]: z-score + Bootstrap BCa、场景三准实验框架
- Hedges 1999 [6]: 场景一效应量
- Marlon 2008 [3]: LOESS 趋势
"""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import (
    zscore_standardize, resample_to_grid, spatial_clustering,
    bam_age_ensemble, harmonize_names,
)
from synthesis import (
    scc_composite, gam_composite, monte_carlo_ensemble,
    uncertainty_band, loess_trend, multi_method_cross_validation,
)
from effect_size import (
    log_response_ratio, effect_size_bca, rmsep, loocv,
    quasi_experiment_effect_size,
)
from continuous_proxy import (
    standardize_continuous_proxy, calibrate_continuous_proxy,
    composite_continuous_proxy, propagate_continuous_uncertainty,
    cross_validate_calibration, proxy_comparison,
)


def select_scenario(
    data_structure: str,
    proxy_type: str = 'auto',
) -> Dict:
    """场景选择器：以数据结构为判据。

    Parameters
    ----------
    data_structure : str
        数据结构类型：
        - 'paired_comparison'：配对比较（代理推断值 vs 真值）→ 场景一
        - 'time_series_stacking'：时序叠加（多站点时间序列）→ 场景二
        - 'before_after'：事件前后准实验 → 场景三
    proxy_type : str, optional
        代理类型：'taxa'（分类群百分比）、'continuous'（连续值）、'auto'（自动判断）。

    Returns
    -------
    Dict
        {'scenario': int, 'name': str, 'proxy_channel': str, 'data_structure': str}
    """
    scenarios = {
        'paired_comparison': {
            'scenario': 1, 'name': '代用指标有效性评估',
            'description': '配对比较结构：代理推断值 vs 已知真值',
        },
        'time_series_stacking': {
            'scenario': 2, 'name': '多站点变化综合',
            'description': '时序叠加结构：多站点时间序列合成',
        },
        'before_after': {
            'scenario': 3, 'name': '事件归因分析',
            'description': '准实验结构：事件前后指标变化',
        },
    }

    info = scenarios.get(data_structure)
    if info is None:
        raise ValueError(
            f"data_structure 须为 'paired_comparison'/'time_series_stacking'/'before_after'，"
            f"得到 '{data_structure}'"
        )

    return {
        'scenario': info['scenario'],
        'name': info['name'],
        'proxy_channel': proxy_type,
        'data_structure': data_structure,
        'description': info['description'],
    }


# ---------------------------------------------------------------------------
# 场景一：代用指标有效性评估
# ---------------------------------------------------------------------------

def scenario1_proxy_validation(
    proxy_values: np.ndarray,
    observed_values: np.ndarray,
    proxy_type: str = 'continuous',
    calibration_x: Optional[np.ndarray] = None,
    calibration_y: Optional[np.ndarray] = None,
    n_boot: int = 10000,
) -> Dict:
    """场景一：代用指标有效性评估（分类群+连续值双通道）。

    数据结构特征：存在配对比较——代理推断值 vs 已知真值。
    这是唯一天然适配经典效应量的场景。

    分类群通道：花粉/硅藻等百分比数据，直接用 log response ratio
    连续值通道：δDwax/brGDGTs 等，先校准再评估

    Parameters
    ----------
    proxy_values : np.ndarray
        代理推断值 (n,)。
    observed_values : np.ndarray
        观测真值 (n,)。
    proxy_type : str, optional
        'continuous' (默认) 或 'taxa'。
    calibration_x, calibration_y : np.ndarray, optional
        校准集（连续值通道需要）。
    n_boot : int, optional
        Bootstrap 次数，默认 10000 (Izdebski 2022)。

    Returns
    -------
    Dict
        {'effect_size': dict, 'rmsep': float, 'ci': tuple, 'calibration': dict or None}
    """
    # 效应量计算（通用）
    ratios = log_response_ratio(proxy_values, observed_values)
    ci = effect_size_bca(proxy_values, observed_values, n_boot=n_boot)
    precision = rmsep(proxy_values, observed_values)

    result = {
        'effect_size': {'ratios': ratios, 'mean_ratio': float(np.mean(ratios))},
        'rmsep': precision,
        'ci': ci,
        'calibration': None,
    }

    # 连续值通道：额外做校准验证
    if proxy_type == 'continuous' and calibration_x is not None and calibration_y is not None:
        calib = calibrate_continuous_proxy(
            proxy_values, calibration_x, calibration_y
        )
        cv = cross_validate_calibration(calibration_x, calibration_y)
        result['calibration'] = {
            'slope': calib['slope'],
            'intercept': calib['intercept'],
            'r2': calib['r2'],
            'rmsep_calibration': calib['rmsep'],
            'cv_rmsep': cv['rmsep'],
            'cv_r2': cv['r2'],
        }

    return result


# ---------------------------------------------------------------------------
# 场景二：多站点变化综合
# ---------------------------------------------------------------------------

def scenario2_multi_site_synthesis(
    site_data: Dict[str, Dict],
    time_grid: np.ndarray,
    proxy_type: str = 'taxa',
    synthesis_method: str = 'gam',
    site_coords: Optional[pd.DataFrame] = None,
    spatial_method: str = 'auto',
    n_members: int = 500,
) -> Dict:
    """场景二：多站点变化综合（分类群+连续值双通道）。

    数据结构特征：多站点时间序列，无配对结构，存在年龄不确定性和自相关。
    经典效应量不适用。

    分类群通道：z-score → 时空对齐 → SCC/GAM/CPS 合成
    连续值通道：标准化 → composite_continuous_proxy 合成

    Parameters
    ----------
    site_data : dict
        站点数据字典 {site_name: {'ages': np.ndarray, 'values': np.ndarray,
        'age_ensembles': np.ndarray (optional), 'proxy_error': float (optional)}}。
    time_grid : np.ndarray
        统一时间网格 (n_bins,)。
    proxy_type : str, optional
        'taxa' (默认) 或 'continuous'。
    synthesis_method : str, optional
        合成方法：'gam'/'scc'/'cps' (分类群通道) 或 'weighted_mean' (连续值通道)。
    site_coords : pd.DataFrame, optional
        站点坐标（含 'lat', 'lon' 列），用于空间聚类。
    spatial_method : str, optional
        空间聚类方法，默认 'auto'（自动选择）。
    n_members : int, optional
        蒙特卡洛集合成员数，默认 500。

    Returns
    -------
    Dict
        {'composite': np.ndarray, 'uncertainty_band': dict,
         'spatial_clusters': dict or None, 'method': str, 'n_sites': int}
    """
    site_names = list(site_data.keys())
    n_sites = len(site_data)

    # 空间聚类（可选）
    clusters = None
    if site_coords is not None:
        clusters = spatial_clustering(site_coords, method=spatial_method)

    if proxy_type == 'continuous':
        # 连续值通道
        site_values_list = [site_data[s]['values'] for s in site_names]
        # 确保等长（重采样到 time_grid）
        max_len = max(len(v) for v in site_values_list)
        site_values = np.full((n_sites, max_len), np.nan)
        for i, v in enumerate(site_values_list):
            site_values[i, :len(v)] = v

        site_ages = np.array([site_data[s]['ages'] for s in site_names])

        age_ens = None
        if 'age_ensembles' in site_data[site_names[0]]:
            age_ens = site_data[site_names[0]]['age_ensembles']

        proxy_errors = None
        if 'proxy_error' in site_data[site_names[0]]:
            proxy_errors = np.array([site_data[s].get('proxy_error', 0) for s in site_names])

        result = composite_continuous_proxy(
            site_values, site_ages, time_grid,
            age_ensembles=age_ens, proxy_errors=proxy_errors,
            n_members=n_members,
        )
        return {
            'composite': result['composite'],
            'uncertainty_band': result['uncertainty_band'],
            'spatial_clusters': clusters,
            'method': f'continuous_{synthesis_method}',
            'n_sites': n_sites,
        }

    # 分类群通道
    ensembles_list = []
    for s in site_names:
        ages = site_data[s]['ages']
        values = site_data[s]['values']
        age_ens = site_data[s].get('age_ensembles')

        z = zscore_standardize(values)
        resampled = resample_to_grid(ages, z['z_scores'], time_grid, age_ensembles=age_ens)
        ensembles_list.append(resampled['resampled'])

    # 合成
    if synthesis_method == 'gam':
        # 用第一个成员拟合 GAM
        first = ensembles_list[0]
        if first.ndim == 2:
            first = first[0]  # 取第一个年龄成员
        gam_result = gam_composite(time_grid, first)
        composite = gam_result['fitted_values'] if 'fitted_values' in gam_result else gam_result.get('composite', first)
    elif synthesis_method == 'scc':
        stacked = np.mean([e if e.ndim == 1 else e[0] for e in ensembles_list], axis=0)
        composite = scc_composite(stacked, time_grid)['composite']
    else:
        stacked = np.mean([e if e.ndim == 1 else e[0] for e in ensembles_list], axis=0)
        composite = stacked

    # 蒙特卡洛集合
    all_ensembles = []
    for e in ensembles_list:
        if e.ndim == 2:
            all_ensembles.append(e)
    if all_ensembles:
        stacked_ens = np.mean(all_ensembles, axis=0)
        band = uncertainty_band(stacked_ens)
    else:
        band = None

    return {
        'composite': composite,
        'uncertainty_band': band,
        'spatial_clusters': clusters,
        'method': f'taxa_{synthesis_method}',
        'n_sites': n_sites,
    }


# ---------------------------------------------------------------------------
# 场景三：事件归因分析
# ---------------------------------------------------------------------------

def scenario3_human_attribution(
    before_data: np.ndarray,
    after_data: np.ndarray,
    event_year: float,
    indicators: Optional[List[str]] = None,
    n_boot: int = 10000,
) -> Dict:
    """场景三：事件归因分析（准实验框架）。

    数据结构特征：事件前后准实验比较。用 Bootstrap BCa 检验差异显著性，
    多时间窗口稳健性验证，双指标系统交叉验证。

    适用于任意代理类型（分类群百分比或连续值）和任意研究区域。
    常见应用：政策实施、战乱、气候事件、土地利用变化前后的指标变化。

    Parameters
    ----------
    before_data : np.ndarray
        事件前数据 (n_before,) 或 (n_before, n_indicators)。
    after_data : np.ndarray
        事件后数据 (n_after,) 或 (n_after, n_indicators)。
    event_year : float
        事件发生年份（用于多窗口分析）。
    indicators : list, optional
        指标名称列表。
    n_boot : int, optional
        Bootstrap 次数，默认 10000。

    Returns
    -------
    Dict
        {'difference': float, 'ci': tuple, 'p_value': float,
         'effect_size': float, 'indicators': list, 'method': str}
    """
    before = np.asarray(before_data, dtype=float)
    after = np.asarray(after_data, dtype=float)

    if before.ndim == 1:
        before = before[:, np.newaxis]
    if after.ndim == 1:
        after = after[:, np.newaxis]

    n_indicators = before.shape[1]
    if indicators is None:
        indicators = [f'indicator_{i+1}' for i in range(n_indicators)]

    results = []
    for i in range(n_indicators):
        diff = np.mean(after[:, i]) - np.mean(before[:, i])
        # Bootstrap BCa 差异检验
        from scipy.stats import bootstrap

        def mean_diff(data, axis=None):
            a, b = data[0], data[1]
            if axis is None:
                return np.mean(a) - np.mean(b)
            return np.mean(a, axis=axis) - np.mean(b, axis=axis)

        try:
            boot_result = bootstrap(
                (after[:, i], before[:, i]),
                statistic=mean_diff,
                n_resamples=n_boot, method='BCa',
                confidence_level=0.95,
            )
            ci = (boot_result.confidence_interval.low,
                  boot_result.confidence_interval.high)
        except Exception:
            ci = (np.nan, np.nan)

        # 效应量
        es = quasi_experiment_effect_size(before[:, i], after[:, i])

        # Mann-Whitney U 检验（非参数）
        from scipy.stats import mannwhitneyu
        try:
            _, p_val = mannwhitneyu(before[:, i], after[:, i], alternative='two-sided')
        except Exception:
            p_val = np.nan

        results.append({
            'indicator': indicators[i],
            'difference': float(diff),
            'ci': ci,
            'p_value': float(p_val),
            'effect_size': es,
        })

    return {
        'results': results,
        'n_indicators': n_indicators,
        'event_year': event_year,
        'method': 'Bootstrap BCa + Mann-Whitney U',
        'n_boot': n_boot,
    }


def build_indicators(
    data: pd.DataFrame,
    taxon_groups: Dict[str, List[str]],
    agg_func: str = 'sum',
) -> Dict:
    """指标构建器（完全用户自定义）。

    用户通过 taxon_groups 字典定义指标体系，不预设任何特定模式。
    适用于任意研究区域和分类群体系。

    使用示例：
    # 东亚农业区指标
    indicators = build_indicators(pollen_df, {
        'crop': ['Oryza', 'Triticum', 'Hordeum'],
        'pasture': ['Poaceae', 'Cyperaceae'],
        'forest': ['Quercus', 'Pinus', 'Castanopsis'],
        'disturbance': ['Artemisia', 'Chenopodiaceae'],
    })

    # 喀斯特石漠化指标
    indicators = build_indicators(pollen_df, {
        'karst_forest': ['Quercus', 'Carpinus'],
        'degraded': ['Pinus', 'Poaceae'],
        'rock_desert': ['Artemisia', 'Chenopodiaceae', 'Ephedra'],
    })

    Parameters
    ----------
    data : pd.DataFrame
        分类群百分比数据，列为分类群名。
    taxon_groups : dict
        指标定义 {indicator_name: [taxon1, taxon2, ...]}。
        用户根据研究区域和科学问题自行设计。
    agg_func : str, optional
        聚合方式：'sum' (默认, 求和) 或 'mean' (均值)。

    Returns
    -------
    Dict
        {'indicators': pd.DataFrame, 'indicator_names': list, 'n_groups': int}
    """
    indicator_df = pd.DataFrame(index=data.index)

    for name, taxa in taxon_groups.items():
        present = [t for t in taxa if t in data.columns]
        if not present:
            indicator_df[name] = 0.0
            continue
        if agg_func == 'sum':
            indicator_df[name] = data[present].sum(axis=1)
        elif agg_func == 'mean':
            indicator_df[name] = data[present].mean(axis=1)
        else:
            raise ValueError(f"agg_func 须为 'sum'/'mean'")

    return {
        'indicators': indicator_df,
        'indicator_names': list(taxon_groups.keys()),
        'n_groups': len(taxon_groups),
        'agg_func': agg_func,
    }


def before_after_test(
    time_series: np.ndarray,
    ages: np.ndarray,
    event_year: float,
    window: int = 50,
    n_boot: int = 10000,
) -> Dict:
    """事件前后差异检验。

    在事件年份前后各取 window 年的数据，用 Bootstrap BCa 检验差异。

    Parameters
    ----------
    time_series : np.ndarray
        时间序列值 (n,)。
    ages : np.ndarray
        年龄数组 (n,)。
    event_year : float
        事件年份。
    window : int, optional
        前后窗口大小（年），默认 50。
    n_boot : int, optional
        Bootstrap 次数。

    Returns
    -------
    Dict
        {'before_mean': float, 'after_mean': float, 'difference': float,
         'ci': tuple, 'p_value': float, 'window': int}
    """
    ages = np.asarray(ages)
    values = np.asarray(time_series, dtype=float)

    before_mask = (ages >= event_year - window) & (ages < event_year)
    after_mask = (ages >= event_year) & (ages <= event_year + window)

    before = values[before_mask]
    after = values[after_mask]

    if len(before) == 0 or len(after) == 0:
        return {'error': f'窗口 {window} 年内数据不足（before={len(before)}, after={len(after)}）'}

    diff = np.mean(after) - np.mean(before)

    from scipy.stats import bootstrap, mannwhitneyu

    try:
        result = bootstrap(
            (after, before),
            statistic=lambda a, b: np.mean(a) - np.mean(b),
            n_resamples=n_boot, method='BCa',
            confidence_level=0.95,
        )
        ci = (result.confidence_interval.low, result.confidence_interval.high)
    except Exception:
        ci = (np.nan, np.nan)

    try:
        _, p_val = mannwhitneyu(before, after, alternative='two-sided')
    except Exception:
        p_val = np.nan

    return {
        'before_mean': float(np.mean(before)),
        'after_mean': float(np.mean(after)),
        'difference': float(diff),
        'ci': ci,
        'p_value': float(p_val),
        'window': window,
        'n_before': len(before),
        'n_after': len(after),
    }


def multi_window_robustness(
    time_series: np.ndarray,
    ages: np.ndarray,
    event_year: float,
    windows: List[int] = None,
    n_boot: int = 10000,
) -> Dict:
    """多时间窗口稳健性检验。

    用不同窗口大小（如 100/50/25 年）重复 before_after_test，
    评估结论是否对窗口选择敏感。

    Parameters
    ----------
    time_series : np.ndarray
        时间序列值 (n,)。
    ages : np.ndarray
        年龄数组 (n,)。
    event_year : float
        事件年份。
    windows : list, optional
        窗口列表，默认 [100, 50, 25]。
    n_boot : int, optional
        Bootstrap 次数。

    Returns
    -------
    Dict
        {'results': dict (window -> test_result), 'robust': bool, 'windows': list}
    """
    if windows is None:
        windows = [100, 50, 25]

    results = {}
    for w in windows:
        results[w] = before_after_test(time_series, ages, event_year, window=w, n_boot=n_boot)

    # 稳健性判断：所有窗口的差异方向一致且 p<0.05
    diffs = [r.get('difference', np.nan) for r in results.values() if 'difference' in r]
    p_vals = [r.get('p_value', np.nan) for r in results.values() if 'p_value' in r]

    if len(diffs) >= 2 and len(p_vals) >= 2:
        same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
        all_significant = all(p < 0.05 for p in p_vals if not np.isnan(p))
        robust = same_sign and all_significant
    else:
        robust = None

    return {
        'results': results,
        'robust': robust,
        'windows': windows,
        'n_windows': len(windows),
    }

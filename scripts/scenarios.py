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

try:
    from .preprocessing import (
        zscore_standardize, resample_to_grid, spatial_clustering,
        age_ensemble_from_errors, bam_age_ensemble, harmonize_names,
    )
    from .synthesis import (
        scc_composite, cps_composite, gam_composite, monte_carlo_ensemble,
        uncertainty_band, loess_trend, multi_method_cross_validation,
    )
    from .effect_size import (
        log_response_ratio, hedges_d, effect_size_bca, rmsep, loocv,
        quasi_experiment_effect_size,
    )
    from .continuous_proxy import (
        standardize_continuous_proxy, calibrate_continuous_proxy,
        composite_continuous_proxy, propagate_continuous_uncertainty,
        cross_validate_calibration, proxy_comparison,
    )
except ImportError:  # pragma: no cover - supports direct script imports
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import (
        zscore_standardize, resample_to_grid, spatial_clustering,
        age_ensemble_from_errors, bam_age_ensemble, harmonize_names,
    )
    from synthesis import (
        scc_composite, cps_composite, gam_composite, monte_carlo_ensemble,
        uncertainty_band, loess_trend, multi_method_cross_validation,
    )
    from effect_size import (
        log_response_ratio, hedges_d, effect_size_bca, rmsep, loocv,
        quasi_experiment_effect_size,
    )
    from continuous_proxy import (
        standardize_continuous_proxy, calibrate_continuous_proxy,
        composite_continuous_proxy, propagate_continuous_uncertainty,
        cross_validate_calibration, proxy_comparison,
    )

try:
    from ._utils import as_float_array, interpolate_no_extrapolation, weighted_nanmean
except ImportError:  # pragma: no cover
    from _utils import as_float_array, interpolate_no_extrapolation, weighted_nanmean


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
            'scenario': 3, 'name': '事件前后变化比较',
            'description': '准实验结构：事件前后指标变化；默认解释为关联而非因果',
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
    effect_type: str = 'auto',
    random_state: Optional[Union[int, np.random.Generator]] = None,
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
    proxy_arr = as_float_array(proxy_values, 'proxy_values', ndim=1)
    observed_arr = as_float_array(observed_values, 'observed_values', ndim=1)
    if len(proxy_arr) != len(observed_arr):
        raise ValueError('proxy_values 与 observed_values 长度必须一致。')
    valid = np.isfinite(proxy_arr) & np.isfinite(observed_arr)
    proxy_arr, observed_arr = proxy_arr[valid], observed_arr[valid]
    if len(proxy_arr) < 3:
        raise ValueError('有效配对样本至少需要三个。')
    if effect_type == 'auto':
        effect_type = 'lnrr' if np.all(proxy_arr > 0) and np.all(observed_arr > 0) else 'd'
    if effect_type == 'lnrr':
        ratios = log_response_ratio(proxy_arr, observed_arr)
        effect_summary = {'ratios': ratios, 'mean_ratio': float(np.mean(ratios))}
    elif effect_type == 'd':
        ratios = None
        effect_summary = hedges_d(proxy_arr, observed_arr)
    else:
        raise ValueError("effect_type 须为 'auto'/'lnrr'/'d'。")
    ci = effect_size_bca(
        proxy_arr, observed_arr, effect_type=effect_type,
        n_boot=n_boot, random_state=random_state,
    )
    precision = rmsep(proxy_values, observed_values)

    result = {
        'effect_size': effect_summary,
        'rmsep': precision,
        'ci': ci,
        'calibration': None,
        'n_valid': int(len(proxy_arr)),
        'effect_type': effect_type,
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
    baseline_mean: float = 0.0,
    baseline_std: float = 1.0,
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
    baseline_mean, baseline_std : float, optional
        CPS 的外部校准基准。未提供真实校准时不要把默认值解释为物理量。

    Returns
    -------
    Dict
        {'composite': np.ndarray, 'uncertainty_band': dict,
         'spatial_clusters': dict or None, 'method': str, 'n_sites': int}
    """
    if not site_data:
        raise ValueError('site_data 至少需要一个站点。')
    if proxy_type not in {'taxa', 'continuous'}:
        raise ValueError("proxy_type 须为 'taxa' 或 'continuous'。")
    allowed_methods = {'scc', 'cps', 'gam', 'weighted_mean'}
    if synthesis_method not in allowed_methods:
        raise ValueError(f'不支持的 synthesis_method: {synthesis_method}')
    grid = as_float_array(time_grid, 'time_grid', ndim=1, allow_nan=False)
    site_names = list(site_data.keys())
    n_sites = len(site_names)

    clusters = None
    if site_coords is not None:
        clusters = spatial_clustering(site_coords, method=spatial_method)

    site_curves = []
    site_weights = []
    for site_name in site_names:
        record = site_data[site_name]
        values = as_float_array(record['values'], f'{site_name}.values', ndim=1)
        ages = as_float_array(record['ages'], f'{site_name}.ages', ndim=1, allow_nan=False)
        if len(values) != len(ages):
            raise ValueError(f'{site_name} 的 values 与 ages 长度不一致。')
        standardized = zscore_standardize(values)['z_scores'] if proxy_type == 'taxa' else values
        result = resample_to_grid(
            ages, standardized, grid,
            age_ensembles=record.get('age_ensembles'),
        )
        curves = result['resampled']
        if curves.ndim == 1:
            curves = curves[None, :]
        site_curves.append(curves)
        site_weights.append(record.get('weight', 1.0))

    site_weights = np.asarray(site_weights, dtype=float)
    if np.any(~np.isfinite(site_weights)) or np.any(site_weights < 0) or site_weights.sum() <= 0:
        raise ValueError('站点 weight 必须为非负有限值且至少有一个正值。')
    n_effective = max(curves.shape[0] for curves in site_curves)
    if n_effective > 1:
        n_effective = min(n_effective, n_members)
    members = []
    for member_index in range(n_effective):
        stacked = np.vstack([
            curves[member_index % curves.shape[0]] for curves in site_curves
        ])
        if synthesis_method == 'scc':
            composite = scc_composite(stacked, site_weights)['composite']
        elif synthesis_method == 'cps':
            composite = cps_composite(
                stacked, baseline_mean=baseline_mean, baseline_std=baseline_std
            )['composite']
        elif synthesis_method == 'gam':
            mean_curve = weighted_nanmean(stacked, site_weights)
            composite = gam_composite(grid, mean_curve)['predicted']
        else:
            composite = weighted_nanmean(stacked, site_weights)
        members.append(composite)

    members = np.asarray(members)
    band = uncertainty_band(members) if len(members) > 1 else None
    composite = band['median'] if band is not None else members[0]
    effective_site_count = np.sum(
        np.isfinite(np.vstack([curves[0] for curves in site_curves])), axis=0
    )
    return {
        'composite': composite,
        'ensembles': members if len(members) > 1 else None,
        'uncertainty_band': band,
        'spatial_clusters': clusters,
        'method': f'{proxy_type}_{synthesis_method}',
        'n_sites': n_sites,
        'n_members': int(len(members)),
        'time_grid': grid,
        'effective_site_count': effective_site_count,
    }


# ---------------------------------------------------------------------------
# 场景三：事件前后关联分析
# ---------------------------------------------------------------------------

def scenario3_human_attribution(
    before_data: np.ndarray,
    after_data: np.ndarray,
    event_year: float,
    indicators: Optional[List[str]] = None,
    n_boot: int = 10000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """场景三：事件前后关联分析（准实验框架）。

    数据结构特征：事件前后准实验比较。用 Bootstrap BCa 检验差异区间，
    多时间窗口稳健性验证，双指标系统交叉验证。默认不作因果归因。

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

        def mean_diff(a, b, axis=-1):
            return np.mean(a, axis=axis) - np.mean(b, axis=axis)

        try:
            boot_result = bootstrap(
                (after[:, i], before[:, i]),
                statistic=mean_diff,
                n_resamples=n_boot, method='BCa',
                confidence_level=0.95,
                random_state=random_state,
            )
            ci = (boot_result.confidence_interval.low,
                  boot_result.confidence_interval.high)
        except Exception as exc:
            ci = (np.nan, np.nan)
            bootstrap_error = str(exc)
        else:
            bootstrap_error = None

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
            'bootstrap_error': bootstrap_error,
        })

    return {
        'results': results,
        'n_indicators': n_indicators,
        'event_year': event_year,
        'method': 'Bootstrap BCa + Mann-Whitney U (association check)',
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
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data 必须是 pandas DataFrame。')
    indicator_df = pd.DataFrame(index=data.index)
    coverage = {}

    for name, taxa in taxon_groups.items():
        present = [t for t in taxa if t in data.columns]
        coverage[name] = {
            'requested': list(taxa),
            'present': present,
            'missing': [t for t in taxa if t not in data.columns],
        }
        if not present:
            indicator_df[name] = np.nan
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
        'coverage': coverage,
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
    ages = as_float_array(ages, 'ages', ndim=1, allow_nan=False)
    values = as_float_array(time_series, 'time_series', ndim=1)
    if len(ages) != len(values):
        raise ValueError('time_series 与 ages 长度必须一致。')

    before_mask = (ages >= event_year - window) & (ages < event_year)
    after_mask = (ages >= event_year) & (ages <= event_year + window)

    before = values[before_mask]
    after = values[after_mask]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]

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
        finite_p = [p for p in p_vals if np.isfinite(p)]
        all_significant = bool(finite_p) and all(p < 0.05 for p in finite_p)
        robust = same_sign and all_significant and len(finite_p) == len(diffs)
    else:
        robust = None

    return {
        'results': results,
        'robust': robust,
        'windows': windows,
        'n_windows': len(windows),
        'valid_windows': len(diffs),
        'note': '窗口重叠且共享时间序列时，稳健性结果不等同于独立重复检验。',
    }

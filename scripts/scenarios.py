"""
三场景配置编排模块。
场景一：代用指标有效性评估；场景二：多站点植被变化综合；场景三：跨区域人地归因。

文献来源：
- Izdebski 2022 [1]: 场景三范式、四指标模式、BCa 差异检验、多窗口验证
- Kaufman 2020 [2]: 场景二五方法框架、500 成员集合
- Hedges 1999 [6]: 场景一 log response ratio

依赖：preprocessing.py, synthesis.py, effect_size.py, validation.py
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import zscore_standardize, resample_to_grid, spatial_clustering
from synthesis import (
    scc_composite, gam_composite, monte_carlo_ensemble,
    uncertainty_band, loess_trend, multi_method_cross_validation,
)
from effect_size import (
    log_response_ratio, effect_size_bca, rmsep, loocv,
    quasi_experiment_effect_size,
)


def select_scenario(data_structure: str) -> Dict:
    """11.1 决策流程图：根据数据结构特征自动选择场景。

    Parameters
    ----------
    data_structure : str
        数据结构类型：
        - 'paired_proxy': 推断值 vs 真值配对 → 场景一
        - 'multi_site_timeseries': 多点时序叠加 → 场景二
        - 'before_after_event': 事件前后比较 → 场景三

    Returns
    -------
    Dict
        {'scenario': int, 'name': str, 'methods': list, 'effect_size_active': bool}
    """
    mapping = {
        'paired_proxy': {
            'scenario': 1,
            'name': '代用指标有效性评估',
            'methods': ['z-score', 'LOOCV', 'log response ratio', 'BCa', 'RMSEP'],
            'effect_size_active': True,
        },
        'multi_site_timeseries': {
            'scenario': 2,
            'name': '多站点植被变化综合',
            'methods': ['BAM年龄集合', 'z-score', '时空对齐', 'SCC/GAM/CPS', '500成员集合', 'LOESS'],
            'effect_size_active': False,
        },
        'before_after_event': {
            'scenario': 3,
            'name': '跨区域人地归因',
            'methods': ['z-score+多指标', 'BCa差异检验', '多窗口', '双指标', '可选效应量(准实验)'],
            'effect_size_active': True,
        },
    }

    if data_structure not in mapping:
        raise ValueError(
            f"data_structure 须为 {list(mapping.keys())} 之一"
        )

    return mapping[data_structure]


def scenario1_proxy_validation(
    proxy_values: np.ndarray,
    observed_values: np.ndarray,
    calibration_func: Optional[Callable] = None,
    n_boot: int = 10000,
) -> Dict:
    """场景一：代用指标有效性评估完整流水线。

    数据结构：配对比较（推断值 vs 真值），满足 Hedges 1999 效应量前提。
    方法链：z-score → LOOCV → log response ratio → BCa → RMSEP

    Parameters
    ----------
    proxy_values : np.ndarray
        代用指标推断值 (n,)。
    observed_values : np.ndarray
        观测真值 (n,)。
    calibration_func : callable, optional
        校准函数，提供时执行 LOOCV。
    n_boot : int, optional
        Bootstrap 重采样次数，默认 10000 (Izdebski 2022)。

    Returns
    -------
    Dict
        {'z_scores': dict, 'effect_size': dict, 'rmsep': dict,
         'loocv': dict or None, 'n': int}
    """
    # 1. z-score 标准化
    z_proxy = zscore_standardize(proxy_values)
    z_observed = zscore_standardize(observed_values)

    # 2. log response ratio + BCa 置信区间
    try:
        es_result = effect_size_bca(
            proxy_values, observed_values,
            effect_type='lnrr', n_boot=n_boot
        )
    except ValueError:
        # 含零值时改用 Hedges' d
        es_result = effect_size_bca(
            proxy_values, observed_values,
            effect_type='d', n_boot=n_boot
        )

    # 3. RMSEP
    rmsep_result = rmsep(proxy_values, observed_values)

    # 4. LOOCV（可选）
    loocv_result = None
    if calibration_func is not None:
        loocv_result = loocv(
            calibration_func,
            proxy_values.reshape(-1, 1) if proxy_values.ndim == 1 else proxy_values,
            observed_values,
        )

    return {
        'z_scores': {'proxy': z_proxy, 'observed': z_observed},
        'effect_size': es_result,
        'rmsep': rmsep_result,
        'loocv': loocv_result,
        'n': len(proxy_values),
        'scenario': 1,
    }


def scenario2_multi_site_synthesis(
    records: np.ndarray,
    age_ensembles: np.ndarray,
    proxy_errors: np.ndarray,
    site_coords: Optional[pd.DataFrame] = None,
    methods: List[str] = ['scc', 'gam', 'cps'],
    n_members: int = 500,
    time_grid: Optional[np.ndarray] = None,
) -> Dict:
    """场景二：多站点植被变化综合完整流水线。

    数据结构：多点时序叠加，无配对，有年龄不确定性和自相关。
    效应量不适用（时序数据违反独立性假设）。
    方法链：BAM年龄集合 → z-score → 时空对齐 → SCC/GAM/CPS合成
            → 500成员集合 → LOESS → 多方法交叉验证

    Parameters
    ----------
    records : np.ndarray
        各点位代理值 (n_sites, n_depths)。
    age_ensembles : np.ndarray
        BAM/Bacon 年龄集合 (n_ensemble, n_depths)。
    proxy_errors : np.ndarray
        各点位代理校准误差 (n_sites,)。
    site_coords : pd.DataFrame, optional
        点位坐标，提供时执行空间聚类。
    methods : list, optional
        合成方法列表，默认 ['scc', 'gam', 'cps']。
    n_members : int, optional
        集合成员数，默认 500 (Kaufman 2020)。
    time_grid : np.ndarray, optional
        统一时间网格。

    Returns
    -------
    Dict
        {'multi_method': dict, 'uncertainty_band': dict, 'loess': dict,
         'spatial_clusters': dict or None, 'n_members': int}
    """
    # 空间聚类（可选）
    spatial_clusters = None
    if site_coords is not None:
        spatial_clusters = spatial_clustering(site_coords, method='karst')

    # 多方法交叉验证 + 集合传播
    multi_result = multi_method_cross_validation(
        records, age_ensembles, proxy_errors,
        methods=methods, n_members=n_members, time_grid=time_grid,
    )

    # 取第一个方法的结果提取不确定性带和 LOESS
    first_method = methods[0]
    ensembles = multi_result['results'][first_method]
    ub = uncertainty_band(ensembles)

    # LOESS 趋势可视化
    median_curve = ub['median']
    if time_grid is not None:
        loess_result = loess_trend(time_grid, median_curve)
    else:
        time_axis = np.arange(len(median_curve))
        loess_result = loess_trend(time_axis, median_curve)

    return {
        'multi_method': multi_result,
        'uncertainty_band': ub,
        'loess': loess_result,
        'spatial_clusters': spatial_clusters,
        'n_members': n_members,
        'scenario': 2,
    }


def scenario3_human_attribution(
    data: pd.DataFrame,
    event_year: float,
    taxon_groups: Dict[str, list],
    windows: List[int] = [100, 50, 25],
    n_boot: int = 10000,
) -> Dict:
    """场景三：跨区域人地归因完整流水线。

    数据结构：离散事件前后多点比较，准实验结构 (before-after)。
    方法链：z-score+多指标 → BCa差异检验 → 多窗口 → 双指标 → 可选效应量(准实验标注)

    事件边界年份须基于独立证据（历史文献/政策记录）确定，
    不可循环使用花粉数据自身确定事件年份。

    Parameters
    ----------
    data : pd.DataFrame
        含 'year' 列和花粉百分比列的数据。
    event_year : float
        事件年份（须从历史文献独立确定）。
    taxon_groups : dict
        分类群分组字典，如 {'cereal': ['Cerealia', 'Triticum'], ...}。
    windows : list, optional
        时间窗口列表（年），默认 [100, 50, 25] (Izdebski 2022)。
    n_boot : int, optional
        Bootstrap 重采样次数，默认 10000。

    Returns
    -------
    Dict
        {'indicators': pd.DataFrame, 'multi_window': dict,
         'effect_size': dict or None, 'event_year': float, 'scenario': 3}
    """
    # 1. 构建多指标
    indicators = build_indicators(data, taxon_groups)

    # 2. z-score 标准化指标
    z_indicators = {}
    for col in indicators.columns:
        z_result = zscore_standardize(indicators[col].values)
        z_indicators[col] = z_result['z_scores']
    z_df = pd.DataFrame(z_indicators, index=indicators.index)
    z_df['year'] = data['year']

    # 3. 多时间窗口稳健性检验
    mw_result = multi_window_robustness(z_df, event_year, windows, n_boot)

    # 4. 可选效应量（准实验标注）
    # 对主窗口（最大窗口）的显著变化计算效应量
    main_window = windows[0]
    before = z_df[
        (z_df.year >= event_year - main_window) & (z_df.year < event_year)
    ].drop(columns=['year'])
    after = z_df[
        (z_df.year > event_year) & (z_df.year <= event_year + main_window)
    ].drop(columns=['year'])

    effect_sizes = {}
    for col in before.columns:
        before_vals = before[col].dropna().values
        after_vals = after[col].dropna().values
        if len(before_vals) > 0 and len(after_vals) > 0:
            effect_sizes[col] = quasi_experiment_effect_size(
                before_vals, after_vals
            )

    return {
        'indicators': indicators,
        'z_indicators': z_df,
        'multi_window': mw_result,
        'effect_size': effect_sizes,
        'event_year': event_year,
        'scenario': 3,
    }


def build_indicators(
    pollen_data: pd.DataFrame,
    taxon_groups: Dict[str, list],
) -> pd.DataFrame:
    """Izdebski 2022 四指标模式：构建谷物/牧业/快速演替/慢速演替指标。

    Parameters
    ----------
    pollen_data : pd.DataFrame
        花粉百分比数据，列为分类群名。
    taxon_groups : dict
        分类群分组字典，如：
        {'cereal': ['Cerealia', 'Triticum', 'Hordeum'],
         'pastoral': ['Plantago', 'Rumex', 'Urtica'],
         'rapid_succession': ['Betula', 'Pinus', 'Corylus'],
         'slow_succession': ['Quercus', 'Fagus', 'Carpinus']}

    Returns
    -------
    pd.DataFrame
        各指标列的 DataFrame。
    """
    indicators = {}
    for name, taxa in taxon_groups.items():
        present_taxa = [t for t in taxa if t in pollen_data.columns]
        if present_taxa:
            indicators[name] = pollen_data[present_taxa].sum(axis=1)
        else:
            indicators[name] = 0.0

    return pd.DataFrame(indicators, index=pollen_data.index)


def before_after_test(
    before_values: np.ndarray,
    after_values: np.ndarray,
    n_boot: int = 10000,
) -> Dict:
    """Izdebski 2022 BCa 方法：检验事件前后均值差异。

    Parameters
    ----------
    before_values : np.ndarray
        事件前子时段指标值 (n_before,) 或 (n_before, n_indicators)。
    after_values : np.ndarray
        事件后子时段指标值 (n_after,) 或 (n_after, n_indicators)。
    n_boot : int, optional
        Bootstrap 重采样次数，默认 10000。

    Returns
    -------
    Dict
        {'diff': float or np.ndarray, 'ci_lower': float, 'ci_upper': float,
         'significant': bool, 'n_before': int, 'n_after': int}
    """
    before_flat = np.asarray(before_values).ravel()
    after_flat = np.asarray(after_values).ravel()
    diff = np.mean(after_flat) - np.mean(before_flat)

    def mean_diff(data, axis=None):
        n_after = len(after_flat)
        if axis is None:
            return np.mean(data[n_after:]) - np.mean(data[:n_after])
        return np.mean(data[n_after:], axis=axis) - np.mean(data[:n_after], axis=axis)

    combined = np.concatenate([before_flat, after_flat])
    result = bootstrap(
        (combined,),
        statistic=mean_diff,
        n_resamples=n_boot,
        method='BCa',
        confidence_level=0.95,
    )

    ci_lower = result.confidence_interval.low
    ci_upper = result.confidence_interval.high
    significant = not (ci_lower <= 0 <= ci_upper)

    return {
        'diff': diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': significant,
        'n_before': len(before_flat),
        'n_after': len(after_flat),
        'method': 'BCa',
    }


def multi_window_robustness(
    data: pd.DataFrame,
    event_year: float,
    windows: List[int] = [100, 50, 25],
    n_boot: int = 10000,
) -> Dict:
    """Izdebski 2022 验证策略：多时间窗口稳健性检验。

    100/50/25 年三期分析，检验结论对窗口选择的敏感性。

    Parameters
    ----------
    data : pd.DataFrame
        含 'year' 列和指标列的 DataFrame。
    event_year : float
        事件年份。
    windows : list, optional
        时间窗口列表（年），默认 [100, 50, 25]。
    n_boot : int, optional
        Bootstrap 重采样次数。

    Returns
    -------
    Dict
        {f'{w}yr': before_after_test result, ...} for each window
    """
    results = {}
    for w in windows:
        before = data[
            (data.year >= event_year - w) & (data.year < event_year)
        ].drop(columns=['year'])
        after = data[
            (data.year > event_year) & (data.year <= event_year + w)
        ].drop(columns=['year'])

        if len(before) > 0 and len(after) > 0:
            # 对所有指标列合并检验
            before_vals = before.values.ravel()
            after_vals = after.values.ravel()
            # 移除 NaN
            before_vals = before_vals[~np.isnan(before_vals)]
            after_vals = after_vals[~np.isnan(after_vals)]

            if len(before_vals) >= 20 and len(after_vals) >= 20:
                results[f'{w}yr'] = before_after_test(
                    before_vals, after_vals, n_boot
                )
            else:
                results[f'{w}yr'] = {
                    'diff': np.mean(after_vals) - np.mean(before_vals),
                    'note': f'样本量不足 (n_before={len(before_vals)}, '
                            f'n_after={len(after_vals)})，BCa 不稳定',
                    'significant': None,
                }
        else:
            results[f'{w}yr'] = {
                'diff': None,
                'note': '该窗口内无数据',
                'significant': None,
            }

    return results

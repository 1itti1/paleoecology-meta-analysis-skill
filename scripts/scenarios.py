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
    from ._utils import (as_float_array, interpolate_no_extrapolation, weighted_nanmean,
                         get_rng, finite_uncertainty_band, validate_age_ensembles,
                         draw_chronology_indices)
except ImportError:  # pragma: no cover
    from _utils import (as_float_array, interpolate_no_extrapolation, weighted_nanmean,
                        get_rng, finite_uncertainty_band, validate_age_ensembles,
                        draw_chronology_indices)


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
        'calibration': {
            'scenario': 4, 'name': '代理校准',
            'description': '训练代理到环境变量的映射并单独进行样本外验证',
        },
        'multi_proxy': {
            'scenario': 5, 'name': '共同目标多代理综合',
            'description': '逐代理标准化、年代对齐、站点内再站点间综合',
        },
        'study_meta': {
            'scenario': 6, 'name': '研究级效应量综合',
            'description': '使用 analysis_contracts.run_analysis 核验尺度与独立簇',
        },
        'narrative': {
            'scenario': 0, 'name': '结构化叙述综合',
            'description': '目标或数据不支持定量合并时保留证据与不合并理由',
        },
    }

    info = scenarios.get(data_structure)
    if info is None:
        raise ValueError(
            f"data_structure 须为 {list(scenarios)}，"
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

    按完整配对验证预测值与参考值。有校准数据时先转换到观测单位。
    默认报告配对预测偏差和 RMSEP。只有显式选择 lnrr 或
    paired_log_ratio 才计算对应比值目标；配对 SMD 不在本接口实现。

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
    if (calibration_x is None) != (calibration_y is None):
        raise ValueError('Provide both calibration_x and calibration_y.')
    calibration = None
    if calibration_x is not None:
        if proxy_type != 'continuous':
            raise ValueError('Calibration is available for the continuous channel.')
        calib = calibrate_continuous_proxy(proxy_arr, calibration_x, calibration_y)
        proxy_arr = calib['calibrated']
        cv = cross_validate_calibration(calibration_x, calibration_y)
        calibration = {
            'slope': calib['slope'], 'intercept': calib['intercept'], 'r2': calib['r2'],
            'rmsep_calibration': calib['rmsep'], 'cv_rmsep': cv['rmsep'], 'cv_r2': cv['r2'],
        }
    # Numeric positivity does not establish that a variable has a meaningful
    # ratio scale. Prediction bias in the observed unit is the default target.
    if effect_type == 'auto':
        effect_type = 'mean_difference'
    if effect_type == 'mean_difference':
        residuals = proxy_arr - observed_arr
        ci = _difference_bootstrap(residuals, np.zeros_like(residuals), n_boot=n_boot,
                                   random_state=random_state, method='iid')
        effect_summary = {'estimate': float(np.mean(residuals)), 'estimand': 'paired_prediction_bias'}
    elif effect_type in {'lnrr', 'paired_log_ratio', 'd'}:
        ci = effect_size_bca(proxy_arr, observed_arr, effect_type=effect_type,
                             design='paired', n_boot=n_boot, random_state=random_state)
        effect_summary = {'estimate': ci['effect_size'], 'estimand': effect_type, 'design': 'paired'}
    else:
        raise ValueError("effect_type must be auto/mean_difference/lnrr/paired_log_ratio/d.")
    precision = rmsep(proxy_arr, observed_arr)
    result = {
        'effect_size': effect_summary,
        'rmsep': precision,
        'ci': ci,
        'calibration': calibration,
        'evaluated_values': proxy_arr,
        'evaluation_scale': 'calibrated_observation_unit' if calibration is not None else 'common_input_unit',
        'n_valid': int(len(proxy_arr)),
        'effect_type': effect_type,
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
    random_state: Optional[Union[int, np.random.Generator]] = None,
    bootstrap_sites: bool = False,
) -> Dict:
    """场景二：多站点变化综合（分类群+连续值双通道）。

    数据结构特征：多站点时间序列，无配对结构，存在年龄不确定性和自相关。
    经典效应量不适用。

    分类群通道：z-score → 时空对齐 → SCC/GAM/CPS 合成
    连续值通道：用户先校准到共同单位，再对齐与合成；不静默改变尺度。
    两通道均传播各站点年代与测量误差，可显式启用站点 bootstrap。

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

    prepared = []
    site_weights = []
    for site_name in site_names:
        record = site_data[site_name]
        values = as_float_array(record['values'], f'{site_name}.values', ndim=1)
        ages = as_float_array(record['ages'], f'{site_name}.ages', ndim=1, allow_nan=False)
        if len(values) != len(ages):
            raise ValueError(f'{site_name} 的 values 与 ages 长度不一致。')
        if proxy_type == 'taxa':
            standardized_info = zscore_standardize(values)
            standardized = standardized_info['z_scores']
            scale = float(standardized_info['std'])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'{site_name} has no informative standardization scale.')
        else:
            standardized, scale = values, 1.0
        error_keys = [key for key in ('proxy_error', 'measurement_error') if key in record]
        if len(error_keys) > 1:
            raise ValueError('Specify one measurement error field per site.')
        errors = np.broadcast_to(np.asarray(record[error_keys[0]] if error_keys else 0.0, dtype=float), values.shape).copy()
        if np.any(~np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError('Measurement errors must be non-negative and finite.')
        pool = record.get('age_ensembles')
        if pool is not None:
            pool = validate_age_ensembles(pool, len(ages), f'{site_name}.age_ensembles')
        prepared.append({'ages': ages, 'values': standardized, 'errors': errors / scale,
                         'pool': pool, 'group': record.get('chronology_group')})
        site_weights.append(record.get('weight', 1.0))

    site_weights = np.asarray(site_weights, dtype=float)
    if np.any(~np.isfinite(site_weights)) or np.any(site_weights < 0) or site_weights.sum() <= 0:
        raise ValueError('站点 weight 必须为非负有限值且至少有一个正值。')
    layers = []
    if any(record['pool'] is not None for record in prepared):
        layers.append('chronology')
    if any(np.any(record['errors'] > 0) for record in prepared):
        layers.append('measurement')
    if bootstrap_sites:
        layers.append('site_bootstrap')
    n_effective = n_members if layers else 1
    if not isinstance(n_members, int) or n_members < 1 or (layers and n_members < 2):
        raise ValueError('Specify an integer n_members >= 2 for uncertainty propagation.')
    rng = get_rng(random_state)
    members = []
    for member_index in range(n_effective):
        selected = draw_chronology_indices([record['pool'] for record in prepared], rng,
                                           [record['group'] for record in prepared])
        curves = []
        for index, record in enumerate(prepared):
            age_values = record['ages'] if record['pool'] is None else record['pool'][selected[index]]
            values = record['values'] + rng.normal(0, record['errors'])
            curves.append(interpolate_no_extrapolation(age_values, values, grid, name=site_names[index]))
        stacked = np.asarray(curves)
        weights = site_weights
        if bootstrap_sites:
            active_sites = np.flatnonzero(site_weights > 0)
            selected_sites = rng.choice(active_sites, size=len(active_sites), replace=True)
            stacked = stacked[selected_sites]
            weights = site_weights[selected_sites]
        if synthesis_method == 'scc':
            composite = scc_composite(stacked, weights)['composite']
        elif synthesis_method == 'cps':
            composite = cps_composite(
                stacked, baseline_mean=baseline_mean, baseline_std=baseline_std, weights=weights
            )['composite']
        elif synthesis_method == 'gam':
            mean_curve = weighted_nanmean(stacked, weights)
            composite = gam_composite(grid, mean_curve)['predicted']
        else:
            composite = weighted_nanmean(stacked, weights)
        members.append(composite)

    members = np.asarray(members)
    band = finite_uncertainty_band(members) if len(members) > 1 else None
    composite = band['median'] if band is not None else members[0]
    effective_site_count = np.sum(
        np.isfinite(np.vstack([interpolate_no_extrapolation(record['ages'], record['values'], grid)
                               for record in prepared])) & (site_weights[:, None] > 0), axis=0
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
        'uncertainty_layers': layers,
        'random_state': random_state if isinstance(random_state, int) else None,
        'spatial_adjustment': 'none; clusters are descriptive metadata',
    }


# ---------------------------------------------------------------------------
# 场景三：事件前后关联分析
# ---------------------------------------------------------------------------

def _difference_bootstrap(after, before, *, n_boot=10000, random_state=None,
                          method='block', block_length=None, confidence_level=0.95):
    """Mean-difference intervals with an explicit sampling assumption."""
    after = as_float_array(after, 'after', ndim=1)
    before = as_float_array(before, 'before', ndim=1)
    after, before = after[np.isfinite(after)], before[np.isfinite(before)]
    if min(len(after), len(before)) < 2:
        raise ValueError('Each comparison window needs at least two finite observations.')
    if method not in {'block', 'iid'}:
        raise ValueError("bootstrap_method must be 'block' or 'iid'.")
    if not isinstance(n_boot, int) or n_boot < 100 or not 0 < confidence_level < 1:
        raise ValueError('Specify n_boot >= 100 and confidence_level in (0, 1).')
    if block_length is None:
        block_length = max(1, int(np.ceil(min(len(after), len(before)) ** (1 / 3))))
    if not isinstance(block_length, int) or not 1 <= block_length <= min(len(after), len(before)):
        raise ValueError('block_length must fit both comparison windows.')
    rng = get_rng(random_state)

    def draw(values):
        n = len(values)
        if method == 'iid':
            return values[rng.integers(n, size=n)]
        starts = rng.integers(n, size=int(np.ceil(n / block_length)))
        indices = (starts[:, None] + np.arange(block_length)) % n
        return values[indices.ravel()[:n]]

    replicates = np.array([np.mean(draw(after)) - np.mean(draw(before)) for _ in range(n_boot)])
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(replicates, [alpha, 1 - alpha])
    return {
        'ci': (float(lower), float(upper)), 'ci_lower': float(lower), 'ci_upper': float(upper),
        'effect_size': float(np.mean(after) - np.mean(before)), 'effect_type': 'mean_difference',
        'confidence_level': confidence_level, 'n_boot': n_boot,
        'method': 'circular_block_percentile' if method == 'block' else 'iid_percentile',
        'block_length': block_length if method == 'block' else 1,
        'p_value': None,
    }


def scenario3_human_attribution(
    before_data: np.ndarray,
    after_data: np.ndarray,
    event_year: float,
    indicators: Optional[List[str]] = None,
    n_boot: int = 10000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    bootstrap_method: str = 'block',
    block_length: Optional[int] = None,
    confidence_level: float = 0.95,
    multiple_testing: str = 'bonferroni',
) -> Dict:
    """场景三：事件前后关联分析（准实验框架）。

    用声明的 block 或 iid 重抽产生均值差区间，默认 block。
    多指标默认 Bonferroni 区间。结果是事件前后关联，不自动提供因果归因。

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

    if before.ndim != 2 or after.ndim != 2 or before.shape[1] != after.shape[1]:
        raise ValueError('Before and after must have the same indicator columns.')
    n_indicators = before.shape[1]
    if indicators is None:
        indicators = [f'indicator_{i+1}' for i in range(n_indicators)]
    if len(indicators) != n_indicators or n_indicators < 1:
        raise ValueError('Provide one name per indicator.')
    if multiple_testing not in {'bonferroni', 'none'}:
        raise ValueError("multiple_testing must be 'bonferroni' or 'none'.")
    adjusted_confidence = (1 - (1 - confidence_level) / n_indicators
                           if multiple_testing == 'bonferroni' else confidence_level)
    rng = get_rng(random_state)

    results = []
    for i in range(n_indicators):
        estimate = _difference_bootstrap(after[:, i], before[:, i], n_boot=n_boot,
                                         random_state=rng, method=bootstrap_method,
                                         block_length=block_length, confidence_level=adjusted_confidence)
        results.append(dict(estimate, indicator=indicators[i], difference=estimate['effect_size'],
                            interpretation='association', bootstrap_error=None))

    return {
        'results': results,
        'n_indicators': n_indicators,
        'event_year': event_year,
        'method': f'{bootstrap_method} bootstrap mean difference (association)',
        'n_boot': n_boot,
        'multiple_testing': multiple_testing,
        'family_confidence_level': confidence_level,
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
    if agg_func not in {'sum', 'mean'}:
        raise ValueError("agg_func must be sum or mean.")
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
            indicator_df[name] = data[present].sum(axis=1, min_count=1)
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
    random_state: Optional[Union[int, np.random.Generator]] = None,
    bootstrap_method: str = 'block',
    block_length: Optional[int] = None,
    confidence_level: float = 0.95,
    time_direction: str = 'increasing',
) -> Dict:
    """事件前后差异检验。

    在事件年份前后各取 window 年的数据，按时间排序做均值差的块重抽。
    time_direction='BP' 表示较大的年龄数值位于事件前。

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
    if not np.isfinite(event_year) or not np.isfinite(window) or window <= 0:
        raise ValueError('event_year must be finite and window must be positive.')
    if time_direction not in {'increasing', 'decreasing', 'CE', 'BP'}:
        raise ValueError('time_direction must be increasing/CE or decreasing/BP.')
    direction = 1 if time_direction in {'increasing', 'CE'} else -1
    order = np.argsort(direction * ages, kind='stable')
    ages, values = ages[order], values[order]
    offset = direction * (ages - event_year)
    before_mask = (offset >= -window) & (offset < 0)
    after_mask = (offset >= 0) & (offset <= window)

    before = values[before_mask]
    after = values[after_mask]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]

    if min(len(before), len(after)) < 2:
        return {'error': f'窗口 {window} 年内数据不足（before={len(before)}, after={len(after)}）'}

    diff = np.mean(after) - np.mean(before)

    estimate = _difference_bootstrap(after, before, n_boot=n_boot, random_state=random_state,
                                     method=bootstrap_method, block_length=block_length,
                                     confidence_level=confidence_level)

    return {
        'before_mean': float(np.mean(before)),
        'after_mean': float(np.mean(after)),
        'difference': float(diff),
        'ci': estimate['ci'],
        'p_value': None,
        'window': window,
        'n_before': len(before),
        'n_after': len(after),
        'method': estimate['method'],
        'confidence_level': confidence_level,
        'block_length': estimate['block_length'],
        'time_direction': time_direction,
        'interpretation': 'association',
    }


def multi_window_robustness(
    time_series: np.ndarray,
    ages: np.ndarray,
    event_year: float,
    windows: List[int] = None,
    n_boot: int = 10000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    **bootstrap_options,
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
        {'results': dict, 'direction_consistent': bool or None,
         'robust': None, 'multiple_testing': 'bonferroni_intervals'}
    """
    if windows is None:
        windows = [100, 50, 25]
    if not windows or len(set(windows)) != len(windows):
        raise ValueError('Specify distinct non-empty windows.')
    family_level = bootstrap_options.pop('confidence_level', 0.95)
    per_window_level = 1 - (1 - family_level) / len(windows)
    rng = get_rng(random_state)

    results = {}
    for w in windows:
        results[w] = before_after_test(time_series, ages, event_year, window=w, n_boot=n_boot,
                                       random_state=rng, confidence_level=per_window_level,
                                       **bootstrap_options)

    diffs = [r.get('difference', np.nan) for r in results.values() if 'difference' in r]
    same_sign = (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)) if diffs else None

    return {
        'results': results,
        'robust': None,
        'direction_consistent': same_sign,
        'multiple_testing': 'bonferroni_intervals',
        'family_confidence_level': family_level,
        'windows': windows,
        'n_windows': len(windows),
        'valid_windows': len(diffs),
        'note': '窗口重叠且共享时间序列时，稳健性结果不等同于独立重复检验。',
    }

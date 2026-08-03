"""
统计严谨性验证模块。
涵盖假设检验、块 Bootstrap、不确定性传播、交叉验证策略。

文献来源：
- Izdebski 2022 [1]: 双指标系统验证、多时间窗口验证
- Kaufman 2020 [2]: 逐一剔除检验、三层不确定性传播
- Hall 1988 [11]: BCa 置信区间理论
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap

try:
    from ._utils import as_float_array, get_rng, normalize_weights
except ImportError:  # pragma: no cover - supports direct script imports
    from _utils import as_float_array, get_rng, normalize_weights


def check_normality_bootstrap(
    data: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """第七章 7.1 正态性假设检验。

    Shapiro-Wilk 检验 + Q-Q 图数据。
    n>30 时 Bootstrap 渐近稳健，严重偏态时增加重采样次数至 20000。

    Parameters
    ----------
    data : np.ndarray
        待检验数据 (n,)。
    alpha : float, optional
        显著性水平，默认 0.05。

    Returns
    -------
    Dict
        {'statistic': float, 'p_value': float, 'is_normal': bool, 'recommendation': str}
    """
    values = as_float_array(data, 'data', ndim=1)
    values = values[np.isfinite(values)]
    n = len(values)

    if n < 3:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'is_normal': None,
            'recommendation': '样本量 n<3，无法检验正态性',
        }

    # Shapiro-Wilk 检验（n<=5000 时有效）
    if n <= 5000:
        stat, p = stats.shapiro(data)
        is_normal = p > alpha
        if is_normal:
            recommendation = '该诊断未发现明显偏离；Bootstrap 是否合适仍取决于依赖结构和统计量。'
        elif n > 30:
            recommendation = '偏离正态；不要因此自动否定 Bootstrap，应检查统计量、依赖结构和尾部。'
        else:
            recommendation = '样本量较小且偏离正态；增加重采样次数不能替代合适的统计模型。'
    else:
        # 大样本用 D'Agostino-Pearson 检验
        stat, p = stats.normaltest(data)
        is_normal = p > alpha
        recommendation = '大样本使用 D\'Agostino-Pearson 诊断；重点检查依赖结构和异常值。'

    return {
        'statistic': stat,
        'p_value': p,
        'is_normal': is_normal,
        'n': n,
        'recommendation': recommendation,
        'test': 'Shapiro-Wilk' if n <= 5000 else 'D\'Agostino-Pearson',
    }


def check_temporal_independence(
    data: Union[np.ndarray, pd.DataFrame],
    time_col: Optional[str] = None,
    detrend: bool = True,
) -> Dict:
    """第七章 7.1 时间独立性假设检验。

    AR1 自相关系数 + Durbin-Watson 统计量。
    违反时使用块 Bootstrap (block_bootstrap)。

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        时序数据。若 DataFrame 且 time_col 指定，按 time_col 排序后取值列。
    time_col : str, optional
        时间列名。

    Returns
    -------
    Dict
        {'ar1': float, 'dw_statistic': float, 'is_independent': bool, 'recommendation': str}
    """
    if isinstance(data, pd.DataFrame) and time_col is not None:
        if time_col not in data.columns:
            raise ValueError(f'time_col 不存在：{time_col}')
        data = data.sort_values(time_col)
        time = as_float_array(data[time_col].values, 'time', ndim=1, allow_nan=False)
        numeric = data.drop(columns=[time_col]).select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError('DataFrame 没有可检验的数值列。')
        values = as_float_array(numeric.iloc[:, 0].values, 'data', ndim=1)
    else:
        values = as_float_array(data, 'data', ndim=1)
        time = np.arange(len(values), dtype=float)

    valid = np.isfinite(values) & np.isfinite(time)
    values, time = values[valid], time[valid]

    n = len(values)

    if n < 3:
        return {
            'ar1': np.nan,
            'dw_statistic': np.nan,
            'is_independent': None,
            'recommendation': '样本量不足，无法检验时间独立性',
        }

    analysis_values = values
    if detrend and n >= 4 and np.unique(time).size >= 2:
        coeffs = np.polyfit(time, values, 1)
        analysis_values = values - np.polyval(coeffs, time)

    if np.std(analysis_values) == 0:
        return {
            'ar1': np.nan,
            'dw_statistic': np.nan,
            'is_independent': None,
            'recommendation': '序列方差为零，无法判断时间独立性。',
            'detrended': detrend,
        }

    # AR1 自相关系数
    ar1 = np.corrcoef(analysis_values[:-1], analysis_values[1:])[0, 1]

    # Durbin-Watson 统计量 (2=独立, 0=正自相关, 4=负自相关)
    residuals = analysis_values - np.mean(analysis_values)
    denominator = np.sum(residuals ** 2)
    dw = np.sum(np.diff(residuals) ** 2) / denominator if denominator > 0 else np.nan

    # 判断：DW 接近 2 且 |AR1| < 0.3 为独立
    is_independent = (abs(dw - 2) < 0.5) and (abs(ar1) < 0.3)

    block_length = 1
    if is_independent:
        recommendation = '时间独立性满足，可使用普通 Bootstrap'
    else:
        block_length = int(np.ceil((1 + ar1) / (1 - ar1))) if 0 < ar1 < 1 else 1
        block_length = max(1, min(block_length, max(1, n // 2)))
        recommendation = (
            f'时间独立性不满足 (AR1={ar1:.3f}, DW={dw:.3f})。'
            f'建议使用块 Bootstrap，块长 ≈ {block_length}'
        )

    return {
        'ar1': ar1,
        'dw_statistic': dw,
        'is_independent': is_independent,
        'block_length_suggestion': block_length if np.isfinite(ar1) else None,
        'recommendation': recommendation,
        'detrended': detrend,
    }


def check_spatial_independence(
    values: np.ndarray,
    coords: np.ndarray,
    radius_km: Optional[float] = 200.0,
) -> Dict:
    """第七章 7.1 空间独立性假设检验。

    使用 PySAL esda.Moran 检验空间自相关。
    违反时空间聚类后再合成，或使用空间加权（距离倒数加权）。

    Parameters
    ----------
    values : np.ndarray
        各点位值 (n_sites,)。
    coords : np.ndarray
        各点位坐标 (n_sites, 2)，经纬度。

    Returns
    -------
    Dict
        {'moran_i': float, 'p_value': float, 'is_independent': bool, 'recommendation': str}
    """
    values = as_float_array(values, 'values', ndim=1, allow_nan=False)
    coords = as_float_array(coords, 'coords', ndim=2, allow_nan=False)
    if coords.shape[0] != len(values) or coords.shape[1] != 2:
        raise ValueError('coords 必须为 (n_sites, 2)，且与 values 行数一致。')
    if len(values) < 3:
        return {'moran_i': np.nan, 'p_value': np.nan, 'is_independent': None,
                'recommendation': '空间站点少于 3 个，无法稳定检验。'}
    if radius_km is not None and (radius_km <= 0 or not np.isfinite(radius_km)):
        raise ValueError('radius_km 必须为正的有限值或 None。')

    # Approximate lon/lat as local kilometres for neighborhood construction.
    lat0 = np.deg2rad(np.mean(coords[:, 0]))
    metric_coords = np.column_stack([
        coords[:, 1] * 111.32 * np.cos(lat0),
        coords[:, 0] * 110.57,
    ])

    try:
        from libpysal.weights import DistanceBand, KNN
        from esda.moran import Moran

        if radius_km is None:
            w = KNN.from_array(metric_coords, k=min(4, len(values) - 1))
        else:
            w = DistanceBand(metric_coords, threshold=radius_km, binary=True, silence_warnings=True)
            if getattr(w, 'islands', []):
                w = KNN.from_array(metric_coords, k=min(4, len(values) - 1))
        w.transform = 'r'
        moran = Moran(values, w)

        is_independent = moran.p_sim > 0.05

        if is_independent:
            recommendation = '空间独立性满足，可使用普通加权合成'
        else:
            recommendation = (
                f'空间自相关显著 (Moran\'s I={moran.I:.3f}, p={moran.p_sim:.3f})。'
                '建议空间聚类后再合成，或使用距离倒数加权'
            )

        return {
            'moran_i': moran.I,
            'p_value': moran.p_sim,
            'is_independent': is_independent,
            'recommendation': recommendation,
            'test': 'Moran\'s I',
            'radius_km': radius_km,
        }

    except ImportError:
        # PySAL 不可用时使用明确标注的简化距离权重。
        n = len(values)
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(metric_coords[i] - metric_coords[j])
                    if radius_km is None or dist <= radius_km:
                        w[i, j] = 1 / dist if dist > 0 else 0

        w_sum = w.sum()
        if w_sum == 0:
            return {'moran_i': np.nan, 'p_value': np.nan, 'is_independent': None,
                    'recommendation': '无法计算空间自相关（坐标退化）'}

        w_row = w.sum(axis=1)
        z = values - np.mean(values)
        z_sum = np.sum(z ** 2)
        if z_sum == 0:
            return {'moran_i': np.nan, 'p_value': np.nan, 'is_independent': None,
                    'recommendation': 'values 方差为零，无法计算 Moran\'s I。'}
        moran_i = (n / w_sum) * np.sum(w * np.outer(z, z)) / z_sum

        is_independent = abs(moran_i) < 0.3

        return {
            'moran_i': moran_i,
            'p_value': np.nan,
            'is_independent': is_independent,
            'recommendation': 'PySAL 不可用，使用简化 Moran\'s I。'
                              + ('空间独立性满足' if is_independent else '建议空间聚类后再合成'),
            'test': 'simplified Moran\'s I',
            'radius_km': radius_km,
        }


def check_sample_size(n: int, method: str = 'bca') -> Dict:
    """第七章 7.1 样本量充分性检查。

    BCa 要求 n>20（最低），n>30（渐近正态）。

    Parameters
    ----------
    n : int
        样本量。
    method : str, optional
        方法：'bca' (默认) 或 'percentile'。

    Returns
    -------
    Dict
        {'sufficient': bool, 'n': int, 'recommendation': str, 'alternative': str or None}
    """
    if n < 1:
        raise ValueError('n 必须为正整数。')
    if method == 'bca':
        if n >= 30:
            return {
                'sufficient': True,
                'n': n,
                'recommendation': '样本量相对有利，但 BCa 稳定性仍取决于统计量和分布。',
                'alternative': None,
            }
        elif n >= 20:
            return {
                'sufficient': True,
                'n': n,
                'recommendation': '样本量较小；BCa 加速因子可能不稳定，应报告敏感性分析。',
                'alternative': None,
            }
        else:
            return {
                'sufficient': False,
                'n': n,
                'recommendation': '样本量很小；不要把 BCa 结果当作稳定推断，应考虑 percentile 或模型方法。',
                'alternative': '改用百分位法 (percentile) 并明确报告样本量限制',
            }
    else:
        if n >= 10:
            return {
                'sufficient': True,
                'n': n,
                'recommendation': '百分位法样本量充分 (n≥10)',
                'alternative': None,
            }
        else:
            return {
                'sufficient': False,
                'n': n,
                'recommendation': '样本量严重不足 (n<10)',
                'alternative': '增加样本量或使用参数化方法',
            }


def block_bootstrap(
    data: np.ndarray,
    statistic: Callable,
    block_length: int,
    n_resamples: int = 10000,
    method: str = 'percentile',
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """第七章 7.1 块 Bootstrap：应对时间自相关。

    当数据存在时间自相关时，普通 Bootstrap 会低估不确定性。
    块 Bootstrap 将数据分为连续块，在块级别重采样。
    块长 = AR1 衰减尺度 (check_temporal_independence 的输出)。

    Parameters
    ----------
    data : np.ndarray
        时序数据 (n,)。
    statistic : callable
        统计量函数，接受 (data, axis) 返回统计量。
    block_length : int
        块长度（建议 = 1/(1-AR1)）。
    n_resamples : int, optional
        重采样次数，默认 10000。
    method : str, optional
        置信区间方法，默认 'BCa'。

    Returns
    -------
    Dict
        {'statistic': float, 'ci_lower': float, 'ci_upper': float, 'block_length': int}
    """
    values = as_float_array(data, 'data', ndim=1, allow_nan=False, min_size=4)
    n = len(values)
    if block_length < 1 or block_length > n:
        raise ValueError('block_length 必须位于 1 和 n 之间。')
    if n_resamples < 100:
        raise ValueError('n_resamples 至少为 100。')
    method_requested = method.lower()
    if method_requested not in {'percentile', 'basic', 'bca'}:
        raise ValueError("method 须为 'percentile'/'basic'/'bca'。")
    if method_requested == 'bca':
        warnings.warn(
            '当前实现对相关数据不宣称精确 BCa；改用 moving-block percentile 区间。',
            RuntimeWarning,
        )
    rng = get_rng(random_state)
    n_blocks = int(np.ceil(n / block_length))
    boot_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        resampled = np.concatenate([
            values[(start + np.arange(block_length)) % n] for start in starts
        ])[:n]
        boot_stats[i] = statistic(resampled)

    if not np.all(np.isfinite(boot_stats)):
        raise ValueError('block bootstrap 统计量产生了非有限值。')
    point_estimate = float(statistic(values))
    q_low, q_high = np.percentile(boot_stats, [2.5, 97.5])
    if method_requested == 'basic':
        ci_lower, ci_upper = 2 * point_estimate - q_high, 2 * point_estimate - q_low
    else:
        ci_lower, ci_upper = q_low, q_high

    return {
        'statistic': point_estimate,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'block_length': block_length,
        'n_resamples': n_resamples,
        'method': f'moving_block_{"percentile" if method_requested == "bca" else method_requested}',
        'requested_method': method_requested,
        'random_state': random_state if isinstance(random_state, int) else None,
    }


def leave_one_out_validation(
    records: np.ndarray,
    age_ensembles: np.ndarray,
    composite_func: Callable,
    n_members: int = 500,
) -> Dict:
    """Kaufman 2020 逐一剔除检验：评估单点位影响。

    每次移除一个点位重新合成，评估结论对单一点位的依赖性。

    Parameters
    ----------
    records : np.ndarray
        各点位代理值 (n_sites, n_depths)。
    age_ensembles : np.ndarray
        年龄集合 (n_ensemble, n_depths)。
    composite_func : callable
        合成函数，接受 (ages, values) 返回合成结果。
    n_members : int, optional
        集合成员数，默认 500。

    Returns
    -------
    Dict
        {'full': np.ndarray, 'leave_one_out': dict, 'max_diff': float, 'stable': bool}
    """
    records = as_float_array(records, 'records', ndim=2)
    age_ensembles = as_float_array(age_ensembles, 'age_ensembles', ndim=2, allow_nan=False)
    if age_ensembles.shape[1] != records.shape[1]:
        raise ValueError('age_ensembles 与 records 的样本维度必须一致。')
    n_sites = records.shape[0]
    n_eval = min(max(1, n_members), age_ensembles.shape[0])

    full_members = [composite_func(age_ensembles[m], records) for m in range(n_eval)]
    full_result = np.nanmedian(np.asarray(full_members), axis=0)

    loo_results = {}
    diffs = []
    for i in range(n_sites):
        mask = np.ones(n_sites, dtype=bool)
        mask[i] = False
        loo_members = [composite_func(age_ensembles[m], records[mask]) for m in range(n_eval)]
        loo_result = np.nanmedian(np.asarray(loo_members), axis=0)
        loo_results[f'site_{i}_removed'] = loo_result
        if len(loo_result) == len(full_result):
            diffs.append(np.nanmax(np.abs(loo_result - full_result)))
    max_diff = max(diffs) if diffs else np.nan

    full_std = np.nanstd(full_result)
    stable = bool(max_diff < 0.5 * full_std) if np.isfinite(max_diff) and full_std > 0 else None

    return {
        'full': full_result,
        'leave_one_out': loo_results,
        'max_diff': max_diff,
        'stable': stable,
        'n_sites': n_sites,
        'n_members_evaluated': n_eval,
    }


def sensitivity_analysis(
    data: np.ndarray,
    param_name: str,
    param_values: list,
    analysis_func: Callable,
) -> Dict:
    """第七章 7.3 敏感性分析：系统变化关键参数评估结论稳健性。

    Parameters
    ----------
    data : np.ndarray
        输入数据。
    param_name : str
        参数名（如 'frac', 'n_splines'）。
    param_values : list
        参数值列表。
    analysis_func : callable
        分析函数，接受 (data, **{param_name: value}) 返回结果。

    Returns
    -------
    Dict
        {'results': dict, 'param_name': str, 'stable': bool, 'conclusion': str}
    """
    results = {}
    for val in param_values:
        kwargs = {param_name: val}
        results[str(val)] = analysis_func(data, **kwargs)

    # 稳定性判断：结果的变异系数
    if all(isinstance(v, (int, float, np.floating)) for v in results.values()):
        values = np.array(list(results.values()))
        cv = np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else np.nan
        stable = cv < 0.2 if not np.isnan(cv) else None
        conclusion = f'变异系数 CV={cv:.3f}' + (
            '，结论稳健' if stable else '，结论对参数敏感'
        )
    else:
        stable = None
        conclusion = '结果为非数值类型，需人工检查一致性'

    return {
        'results': results,
        'param_name': param_name,
        'stable': stable,
        'conclusion': conclusion,
    }


def propagate_three_layer_uncertainty(
    age_ensembles: np.ndarray,
    calibration_errors: np.ndarray,
    sample_data: np.ndarray,
    composite_func: Callable,
    n_members: int = 500,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """第七章 7.2 三层不确定性传播：年龄+校准+采样联合传播。

    - 年龄不确定性：从外部年代模型后验整体采样完整年龄-深度曲线（保持地层单调性）
    - 校准不确定性：代理-气候校准残差作为独立正态噪声，标准差来自 RMSEP
    - 采样不确定性：Bootstrap 重采样自然传播

    Parameters
    ----------
    age_ensembles : np.ndarray
        年龄集合 (n_ensemble, n_depths)。
    calibration_errors : np.ndarray
        各点位校准误差 (n_sites,)，来自 RMSEP 或 CV-RMSE。
    sample_data : np.ndarray
        各点位代理值 (n_sites, n_depths)。
    composite_func : callable
        合成函数。
    n_members : int, optional
        集合成员数，默认 500 (Kaufman 2020)。

    Returns
    -------
    Dict
        {'ensembles': np.ndarray (n_members, n_timebins),
         'uncertainty_band': dict, 'n_members': int, 'layers': list}
    """
    sample_data = as_float_array(sample_data, 'sample_data', ndim=2)
    age_ensembles = as_float_array(age_ensembles, 'age_ensembles', ndim=2, allow_nan=False)
    calibration_errors = as_float_array(calibration_errors, 'calibration_errors', ndim=1, allow_nan=False)
    if age_ensembles.shape[1] != sample_data.shape[1]:
        raise ValueError('age_ensembles 与 sample_data 的样本维度必须一致。')
    if len(calibration_errors) != sample_data.shape[0] or np.any(calibration_errors < 0):
        raise ValueError('calibration_errors 必须按站点提供且非负。')
    if n_members < 2:
        raise ValueError('n_members 至少为 2。')
    n_sites = sample_data.shape[0]
    n_ensemble_pool = age_ensembles.shape[0]
    rng = get_rng(random_state)
    ensembles = []

    for i in range(n_members):
        # 1. 年龄层：整体采样一个年龄成员（保持地层单调性）
        ages = age_ensembles[i % n_ensemble_pool]

        # 2. 校准层：添加校准残差正态噪声
        calib_noise = rng.normal(
            0, calibration_errors[:, np.newaxis], size=sample_data.shape
        )
        calibrated = sample_data + calib_noise

        # 3. 采样层：Bootstrap 重采样（对点位重采样）
        boot_indices = rng.choice(n_sites, size=n_sites, replace=True)
        boot_data = calibrated[boot_indices]

        # 执行合成
        composite = composite_func(ages, boot_data)
        ensembles.append(composite)

    ensembles = np.array(ensembles)

    # 提取不确定性带
    band = {
        'lower': np.percentile(ensembles, 5, axis=0),
        'median': np.percentile(ensembles, 50, axis=0),
        'upper': np.percentile(ensembles, 95, axis=0),
    }

    return {
        'ensembles': ensembles,
        'uncertainty_band': band,
        'n_members': n_members,
        'layers': ['age', 'calibration', 'sampling'],
        'method': 'three-layer Monte Carlo',
        'random_state': random_state if isinstance(random_state, int) else None,
    }


def dual_indicator_check(
    data: pd.DataFrame,
    indicator_set_a: Dict[str, list],
    indicator_set_b: Dict[str, list],
    analysis_func: Callable,
) -> Dict:
    """Izdebski 2022 双指标系统验证：两套独立生态指标分组对比。

    使用两套独立生态指标（如 Ellenberg vs Niinemets 光指数）分别分析，
    比较结论一致性。

    Parameters
    ----------
    data : pd.DataFrame
        分类群百分比数据（花粉、硅藻、有孔虫等）。
    indicator_set_a : dict
        第一套指标分组（如基于 Ellenberg 光指数）。
    indicator_set_b : dict
        第二套指标分组（如基于 Niinemets 光指数）。
    analysis_func : callable
        分析函数，接受 (indicators_df) 返回结果。

    Returns
    -------
    Dict
        {'result_a': any, 'result_b': any, 'consistent': bool or None, 'note': str}
    """
    # 构建两套指标
    try:
        from .scenarios import build_indicators
    except ImportError:  # pragma: no cover
        from scenarios import build_indicators

    indicators_a = build_indicators(data, indicator_set_a)
    indicators_b = build_indicators(data, indicator_set_b)

    # 分别分析
    result_a = analysis_func(indicators_a['indicators'])
    result_b = analysis_func(indicators_b['indicators'])

    # 一致性判断（简化：若结果为数值则比较方向）
    consistent = None
    if isinstance(result_a, (int, float, np.floating)) and \
       isinstance(result_b, (int, float, np.floating)):
        consistent = np.sign(result_a) == np.sign(result_b)

    return {
        'result_a': result_a,
        'result_b': result_b,
        'consistent': consistent,
        'note': 'Izdebski 2022 验证策略：双指标系统稳健性检验。'
                '两套独立指标结论一致只能作为一致性证据，不能单独证明因果关系。',
        'coverage_a': indicators_a.get('coverage'),
        'coverage_b': indicators_b.get('coverage'),
    }

"""
统计严谨性验证模块。
涵盖假设检验、块 Bootstrap、不确定性传播、交叉验证策略。

文献来源：
- Izdebski 2022 [1]: 双指标系统验证、多时间窗口验证
- Kaufman 2020 [2]: 逐一剔除检验、三层不确定性传播
- Hall 1988 [11]: BCa 置信区间理论
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap


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
    n = len(data)

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
            recommendation = '正态性假设满足，可使用 Bootstrap BCa'
        elif n > 30:
            recommendation = '正态性不满足但 n>30，Bootstrap 渐近稳健，可继续分析'
        else:
            recommendation = '正态性不满足且 n≤30，建议增加重采样次数至 20000 或改用非参数方法'
    else:
        # 大样本用 D'Agostino-Pearson 检验
        stat, p = stats.normaltest(data)
        is_normal = p > alpha
        recommendation = '大样本(n>5000)使用 D\'Agostino-Pearson 检验，Bootstrap 渐近稳健'

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
        data = data.sort_values(time_col)
        values = data.drop(columns=[time_col]).select_dtypes(include=[np.number])
        # 取第一数值列
        values = values.iloc[:, 0].values
    else:
        values = np.asarray(data).ravel()

    n = len(values)

    if n < 3:
        return {
            'ar1': np.nan,
            'dw_statistic': np.nan,
            'is_independent': None,
            'recommendation': '样本量不足，无法检验时间独立性',
        }

    # AR1 自相关系数
    ar1 = np.corrcoef(values[:-1], values[1:])[0, 1]

    # Durbin-Watson 统计量 (2=独立, 0=正自相关, 4=负自相关)
    residuals = values - np.mean(values)
    dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)

    # 判断：DW 接近 2 且 |AR1| < 0.3 为独立
    is_independent = (abs(dw - 2) < 0.5) and (abs(ar1) < 0.3)

    if is_independent:
        recommendation = '时间独立性满足，可使用普通 Bootstrap'
    else:
        block_length = int(1 / (1 - ar1)) if ar1 > 0 else 1
        recommendation = (
            f'时间独立性不满足 (AR1={ar1:.3f}, DW={dw:.3f})。'
            f'建议使用块 Bootstrap，块长 ≈ {block_length}'
        )

    return {
        'ar1': ar1,
        'dw_statistic': dw,
        'is_independent': is_independent,
        'block_length_suggestion': int(1 / (1 - ar1)) if ar1 > 0 else 1,
        'recommendation': recommendation,
    }


def check_spatial_independence(
    values: np.ndarray,
    coords: np.ndarray,
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
    try:
        from libpysal.weights import DistanceBand
        from esda.moran import Moran

        w = DistanceBand(coords, threshold=np.inf)
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
        }

    except ImportError:
        # PySAL 不可用时用简化版本
        n = len(values)
        # 简化的 Moran's I 计算
        w = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt(
                        (coords[i, 0] - coords[j, 0]) ** 2
                        + (coords[i, 1] - coords[j, 1]) ** 2
                    )
                    w[i, j] = 1 / dist if dist > 0 else 0

        w_sum = w.sum()
        if w_sum == 0:
            return {'moran_i': np.nan, 'p_value': np.nan, 'is_independent': None,
                    'recommendation': '无法计算空间自相关（坐标退化）'}

        w_row = w.sum(axis=1)
        z = values - np.mean(values)
        moran_i = (n / w_sum) * np.sum(w * np.outer(z, z)) / np.sum(z ** 2)

        is_independent = abs(moran_i) < 0.3

        return {
            'moran_i': moran_i,
            'p_value': np.nan,
            'is_independent': is_independent,
            'recommendation': 'PySAL 不可用，使用简化 Moran\'s I。'
                              + ('空间独立性满足' if is_independent else '建议空间聚类后再合成'),
            'test': 'simplified Moran\'s I',
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
    if method == 'bca':
        if n >= 30:
            return {
                'sufficient': True,
                'n': n,
                'recommendation': '样本量充分 (n≥30)，BCa 渐近正态',
                'alternative': None,
            }
        elif n >= 20:
            return {
                'sufficient': True,
                'n': n,
                'recommendation': '样本量达到 BCa 最低要求 (n≥20)，但建议报告样本量限制',
                'alternative': None,
            }
        else:
            return {
                'sufficient': False,
                'n': n,
                'recommendation': '样本量不足 (n<20)，BCa 加速因子可能不稳定',
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
    method: str = 'BCa',
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
    n = len(data)
    n_blocks = n // block_length

    # 生成块 Bootstrap 样本
    boot_stats = np.zeros(n_resamples)
    for i in range(n_resamples):
        # 随机选择块起点
        block_starts = np.random.randint(0, n - block_length + 1, size=n_blocks)
        # 拼接块
        resampled = np.concatenate([
            data[s:s + block_length] for s in block_starts
        ])
        boot_stats[i] = statistic(resampled)

    point_estimate = statistic(data)
    ci_lower = np.percentile(boot_stats, 2.5)
    ci_upper = np.percentile(boot_stats, 97.5)

    return {
        'statistic': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'block_length': block_length,
        'n_resamples': n_resamples,
        'method': f'block {method}',
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
    n_sites = records.shape[0]

    # 完整合成
    full_result = composite_func(
        age_ensembles[0], records
    )

    # 逐一剔除
    loo_results = {}
    for i in range(n_sites):
        mask = np.ones(n_sites, dtype=bool)
        mask[i] = False
        loo_records = records[mask]
        loo_results[f'site_{i}_removed'] = composite_func(
            age_ensembles[0], loo_records
        )

    # 计算最大差异
    diffs = []
    for key, val in loo_results.items():
        if len(val) == len(full_result):
            diffs.append(np.max(np.abs(val - full_result)))
    max_diff = max(diffs) if diffs else np.nan

    # 稳定性判断：最大差异 < 完整合成标准差的 50%
    full_std = np.std(full_result)
    stable = max_diff < 0.5 * full_std if not np.isnan(max_diff) else None

    return {
        'full': full_result,
        'leave_one_out': loo_results,
        'max_diff': max_diff,
        'stable': stable,
        'n_sites': n_sites,
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
) -> Dict:
    """第七章 7.2 三层不确定性传播：年龄+校准+采样联合传播。

    - 年龄不确定性：从 BAM/Bacon 后验分布整体采样完整年龄-深度曲线（保持地层单调性）
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
    n_sites = sample_data.shape[0]
    n_ensemble_pool = age_ensembles.shape[0]
    ensembles = []

    for i in range(n_members):
        # 1. 年龄层：整体采样一个年龄成员（保持地层单调性）
        ages = age_ensembles[i % n_ensemble_pool]

        # 2. 校准层：添加校准残差正态噪声
        calib_noise = np.random.normal(
            0, calibration_errors[:, np.newaxis], size=sample_data.shape
        )
        calibrated = sample_data + calib_noise

        # 3. 采样层：Bootstrap 重采样（对点位重采样）
        boot_indices = np.random.choice(n_sites, size=n_sites, replace=True)
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
    from scenarios import build_indicators

    indicators_a = build_indicators(data, indicator_set_a)
    indicators_b = build_indicators(data, indicator_set_b)

    # 分别分析
    result_a = analysis_func(indicators_a)
    result_b = analysis_func(indicators_b)

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
                '两套独立指标结论一致则增强结论可信度。',
    }

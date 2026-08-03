"""
连续值代理专用模块（δDwax / brGDGTs / 粒度 / 有机碳 / Mg-Ca 等）。

与分类群百分比型代理（花粉、硅藻、有孔虫）不同，连续值代理是单一数值序列，
不需要分类群求和/百分比约束，但需要特定的校准和不确定性传播方法。

本模块提供连续值代理的标准化、校准、多站点合成、不确定性传播专用函数，
与 preprocessing.py / synthesis.py 中的通用函数形成双通道并行架构。

文献来源：
- Kaufman 2020 [2]: brGDGTs 温度校准、多站点合成框架
- Roberts 2018 [5]: 代理-气候校准回归方法
- Izdebski 2022 [1]: z-score 标准化、Bootstrap BCa
- Marlon 2008 [3]: LOESS 趋势可视化
- Hedges 1999 [6]: 校准模型验证中的效应量
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    from ._utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        normalize_weights,
        weighted_nanmean,
    )
except ImportError:  # pragma: no cover - supports direct script imports
    from _utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        normalize_weights,
        weighted_nanmean,
    )


# ---------------------------------------------------------------------------
# 常见连续值代理元数据（用于文档提示，不强制使用）
# ---------------------------------------------------------------------------
CONTINUOUS_PROXY_REGISTRY: Dict[str, Dict] = {
    'dDwax': {
        'description': '正构烷烃氢同位素比率',
        'climate_variable': '降水 δD / 蒸发',
        'common_units': '‰ (VSMOW)',
        'calibration_note': '通常用全球或区域校准线 δDwax = a × δDp + b',
    },
    'brGDGTs': {
        'description': '支链甘油二烷基甘油四醚',
        'climate_variable': '温度 / pH',
        'common_units': '℃ (MBT\'5me) /无量纲',
        'calibration_note': 'MBT\'5me 或 CBT 校准，注意土壤 vs 湖泊校准集差异',
    },
    'grain_size': {
        'description': '沉积物粒度分布',
        'climate_variable': '风强 / 水动力',
        'common_units': 'μm / φ',
        'calibration_note': '通常不直接校准为气候变量，用 z-score 标准化后比较',
    },
    'toc': {
        'description': '总有机碳',
        'climate_variable': '生产力 / 保存',
        'common_units': 'wt%',
        'calibration_note': '需校正矿化作用随深度的系统性变化',
    },
    'mgca': {
        'description': '有孔虫 Mg/Ca 比值',
        'climate_variable': '海水温度',
        'common_units': 'mmol/mol',
        'calibration_note': '需校正 ΔCO3²⁻ 和 salinity 影响',
    },
    'uk37': {
        'description': '长链烯酮不饱和度',
        'climate_variable': '海表温度',
        'common_units': '℃',
        'calibration_note': 'Uk37 = C37:2/(C37:2+C37:3)',
    },
}


def standardize_continuous_proxy(
    values: np.ndarray,
    method: str = 'zscore',
    baseline_period: Optional[Tuple[float, float]] = None,
    ages: Optional[np.ndarray] = None,
) -> Dict:
    """连续值代理标准化（多方法可选）。

    连续值代理的量纲和尺度差异极大（‰、℃、μm、wt%），标准化是合成的必要前提。
    本函数提供三种标准化方法，以数据分布特征为判据选择。

    Parameters
    ----------
    values : np.ndarray
        代理值数组 (n,)。
    method : str, optional
        标准化方法：
        - 'zscore' (默认)：z = (x - μ) / σ (Izdebski 2022)。适用于近似正态分布。
        - 'minmax'：缩放到 [0, 1]。适用于已知边界的代理（如百分比）。
        - 'robust'：(x - median) / IQR。适用于含离群点的代理。
    baseline_period : tuple, optional
        基准时段 (start_age, end_age)，需同时提供 ages。
    ages : np.ndarray, optional
        年龄数组，与 values 等长。baseline_period 指定时用于选取基准子集。

    Returns
    -------
    Dict
        {'standardized': np.ndarray, 'method': str, 'params': dict}
    """
    arr = as_float_array(values, 'values', ndim=1)

    # 提取基准子集
    if baseline_period is not None:
        if ages is None:
            raise ValueError('使用 baseline_period 时必须提供 ages。')
        ages = as_float_array(ages, 'ages', ndim=1, allow_nan=False)
        if len(ages) != len(arr):
            raise ValueError('ages 必须与 values 长度一致。')
        mask = (ages >= baseline_period[0]) & (ages <= baseline_period[1])
        baseline_data = arr[mask]
        if not np.any(np.isfinite(baseline_data)):
            raise ValueError('baseline_period 内没有有效观测。')
    else:
        baseline_data = arr

    if method == 'zscore':
        mu = np.nanmean(baseline_data)
        sigma = np.nanstd(baseline_data)
        if not np.isfinite(sigma) or sigma == 0:
            raise ValueError('z-score 基准标准差为零或非有限值。')
        standardized = (arr - mu) / sigma
        params = {'mean': mu, 'std': sigma}

    elif method == 'minmax':
        lo = np.nanmin(baseline_data)
        hi = np.nanmax(baseline_data)
        if hi == lo:
            raise ValueError('minmax 基准范围为零。')
        rng = hi - lo
        standardized = (arr - lo) / rng
        params = {'min': lo, 'max': hi}

    elif method == 'robust':
        med = np.nanmedian(baseline_data)
        q1, q3 = np.nanpercentile(baseline_data, [25, 75])
        if q3 == q1:
            raise ValueError('robust 标准化的 IQR 为零。')
        iqr = q3 - q1
        standardized = (arr - med) / iqr
        params = {'median': med, 'iqr': iqr, 'q1': q1, 'q3': q3}

    else:
        raise ValueError(f"method 须为 'zscore'/'minmax'/'robust'，得到 '{method}'")

    return {'standardized': standardized, 'method': method, 'params': params}


def calibrate_continuous_proxy(
    proxy_values: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    method: str = 'linear',
    regression: str = 'ols',
) -> Dict:
    """连续值代理-气候变量校准。

    将代理原始值转换为气候变量（温度、降水等）。
    支持线性回归和标准化主轴回归（SMA），后者适用于 X 和 Y 均含误差的情况
    (Roberts 2018)。

    Parameters
    ----------
    proxy_values : np.ndarray
        待校准的代理值 (n,)。
    calibration_x : np.ndarray
        校准集代理值 (m,)。
    calibration_y : np.ndarray
        校准集气候变量值 (m,)。
    method : str, optional
        校准方法：'linear' (默认) 或 'polynomial'。
    regression : str, optional
        回归类型：'ols' (普通最小二乘, 默认) 或 'sma' (标准化主轴回归)。

    Returns
    -------
    Dict
        {'calibrated': np.ndarray, 'slope': float, 'intercept': float,
         'r2': float, 'rmsep': float, 'method': str, 'regression': str}
    """
    x = as_float_array(calibration_x, 'calibration_x', ndim=1, allow_nan=False, min_size=3)
    y = as_float_array(calibration_y, 'calibration_y', ndim=1, allow_nan=False, min_size=3)
    proxy = as_float_array(proxy_values, 'proxy_values', ndim=1)
    if len(x) != len(y):
        raise ValueError('calibration_x 与 calibration_y 长度必须一致。')
    if np.unique(x).size < 2:
        raise ValueError('calibration_x 至少需要两个不同值。')
    coeffs = None

    if regression == 'ols':
        if method == 'linear':
            slope, intercept, r, p, se = sp_stats.linregress(x, y)
        elif method == 'polynomial':
            coeffs = np.polyfit(x, y, 2)
            slope = np.nan
            intercept = np.nan
            r = np.nan
        else:
            raise ValueError(f"method 须为 'linear'/'polynomial'")
    elif regression == 'sma':
        # 标准化主轴回归 (Standardized Major Axis)
        # slope_sma = sign(r) * (sy/sx)
        sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
        if sx == 0 or sy == 0:
            raise ValueError('SMA 回归要求 x 和 y 都有非零方差。')
        r = np.corrcoef(x, y)[0, 1]
        slope = np.sign(r) * (sy / sx)
        intercept = np.mean(y) - slope * np.mean(x)
    else:
        raise ValueError(f"regression 须为 'ols'/'sma'")

    # 校准
    if method == 'linear' or regression == 'sma':
        calibrated = slope * proxy + intercept
    else:
        calibrated = np.polyval(coeffs, proxy)

    # 校准集 RMSEP
    predicted = slope * x + intercept if method == 'linear' or regression == 'sma' else np.polyval(coeffs, x)
    rmsep_val = np.sqrt(np.mean((predicted - y) ** 2))
    ss_res = np.sum((predicted - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'calibrated': calibrated,
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'rmsep': float(rmsep_val),
        'method': method,
        'regression': regression,
        'coefficients': coeffs.tolist() if coeffs is not None else None,
    }


def composite_continuous_proxy(
    site_values: Union[np.ndarray, Sequence[np.ndarray]],
    site_ages: Union[np.ndarray, Sequence[np.ndarray]],
    time_grid: np.ndarray,
    weights: Optional[np.ndarray] = None,
    age_ensembles: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    proxy_errors: Optional[np.ndarray] = None,
    n_members: int = 500,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """连续值代理多站点合成（含年龄不确定性传播）。

    与分类群合成不同，连续值代理不需要分类群求和/百分比约束，
    直接在统一时间网格上插值后加权平均即可。

    Parameters
    ----------
    site_values : np.ndarray
        各站点代理值 (n_sites, n_depths)。
    site_ages : np.ndarray
        各站点年龄 (n_sites, n_depths) 或共享年龄 (n_depths,)。
    time_grid : np.ndarray
        统一时间网格 (n_bins,)。
    weights : np.ndarray, optional
        各站点权重 (n_sites,)，默认等权。
    age_ensembles : np.ndarray, optional
        年龄集合 (n_ensemble, n_depths)，用于传播年龄不确定性。
        若各站点有独立年龄集合，取首个（共享年表）或预处理为统一集合。
    proxy_errors : np.ndarray, optional
        各站点代理校准误差 (n_sites,)，用于传播校准不确定性。
    n_members : int, optional
        蒙特卡洛集合成员数，默认 500 (Kaufman 2020)。

    Returns
    -------
    Dict
        {'composite': np.ndarray (n_bins,), 'ensembles': np.ndarray or None,
         'uncertainty_band': dict or None, 'n_sites': int, 'n_members': int}
    """
    grid = as_float_array(time_grid, 'time_grid', ndim=1, allow_nan=False)

    if isinstance(site_values, np.ndarray) and site_values.ndim == 2:
        values_list = [as_float_array(row, f'site_values[{i}]', ndim=1) for i, row in enumerate(site_values)]
    else:
        values_list = [as_float_array(row, f'site_values[{i}]', ndim=1) for i, row in enumerate(site_values)]
    n_sites = len(values_list)
    if n_sites == 0:
        raise ValueError('至少需要一个站点。')

    if isinstance(site_ages, np.ndarray) and site_ages.ndim == 1:
        ages_list = [as_float_array(site_ages, 'site_ages', ndim=1, allow_nan=False)] * n_sites
    elif isinstance(site_ages, np.ndarray) and site_ages.ndim == 2:
        if site_ages.shape[0] != n_sites:
            raise ValueError('site_ages 的站点维度不匹配。')
        ages_list = [as_float_array(row, f'site_ages[{i}]', ndim=1, allow_nan=False) for i, row in enumerate(site_ages)]
    else:
        ages_list = [as_float_array(row, f'site_ages[{i}]', ndim=1, allow_nan=False) for i, row in enumerate(site_ages)]
    if len(ages_list) != n_sites:
        raise ValueError('site_values 与 site_ages 的站点数量必须一致。')
    for i, (values, ages) in enumerate(zip(values_list, ages_list)):
        if len(values) != len(ages):
            raise ValueError(f'站点 {i} 的 values 与 ages 长度不一致。')

    weights_arr = normalize_weights(weights, n_sites)
    if proxy_errors is None:
        errors = np.zeros(n_sites)
    else:
        errors = as_float_array(proxy_errors, 'proxy_errors', ndim=1, allow_nan=False)
        if len(errors) != n_sites or np.any(errors < 0):
            raise ValueError('proxy_errors 必须按站点提供且非负。')

    # Normalize shared or per-site age ensembles without padding ragged records.
    age_members = None
    if age_ensembles is not None:
        if isinstance(age_ensembles, np.ndarray) and age_ensembles.ndim == 2:
            shared = as_float_array(age_ensembles, 'age_ensembles', ndim=2, allow_nan=False)
            if any(len(ages) != shared.shape[1] for ages in ages_list):
                raise ValueError('共享 age_ensembles 的样本维度必须匹配每个站点。')
            age_members = [shared] * n_sites
        elif isinstance(age_ensembles, np.ndarray) and age_ensembles.ndim == 3:
            if age_ensembles.shape[0] != n_sites:
                raise ValueError('逐站点 age_ensembles 的站点维度不匹配。')
            age_members = [as_float_array(age_ensembles[i], f'age_ensembles[{i}]', ndim=2, allow_nan=False) for i in range(n_sites)]
        else:
            if len(age_ensembles) != n_sites:
                raise ValueError('逐站点 age_ensembles 的站点数量不匹配。')
            age_members = [as_float_array(v, f'age_ensembles[{i}]', ndim=2, allow_nan=False) for i, v in enumerate(age_ensembles)]
            for i, (members, ages) in enumerate(zip(age_members, ages_list)):
                if members.shape[1] != len(ages):
                    raise ValueError(f'站点 {i} 的 age_ensembles 样本维度不匹配。')

    rng = get_rng(random_state)
    has_uncertainty = age_members is not None or np.any(errors > 0)

    def make_member(member_index: int) -> np.ndarray:
        resampled = np.full((n_sites, len(grid)), np.nan)
        for s in range(n_sites):
            ages_s = ages_list[s]
            if age_members is not None:
                pool = age_members[s]
                ages_s = pool[member_index % pool.shape[0]]
            values_s = values_list[s]
            if errors[s] > 0:
                values_s = values_s + rng.normal(0, errors[s], size=len(values_s))
            resampled[s] = interpolate_no_extrapolation(
                ages_s, values_s, grid, name=f'site_{s}'
            )
        return weighted_nanmean(resampled, weights_arr)

    if not has_uncertainty:
        composite = make_member(0)
        return {
            'composite': composite,
            'ensembles': None,
            'uncertainty_band': None,
            'n_sites': n_sites,
            'n_members': 1,
            'weights': weights_arr,
            'extrapolation': 'nan',
        }

    if n_members < 2:
        raise ValueError('有不确定性传播时 n_members 至少为 2。')
    ensembles = np.array([make_member(m) for m in range(n_members)])
    band = {
        'lower': np.nanpercentile(ensembles, 5, axis=0),
        'median': np.nanpercentile(ensembles, 50, axis=0),
        'upper': np.nanpercentile(ensembles, 95, axis=0),
        'percentiles': (5, 50, 95),
    }
    return {
        'composite': band['median'],
        'ensembles': ensembles,
        'uncertainty_band': band,
        'n_sites': n_sites,
        'n_members': n_members,
        'weights': weights_arr,
        'random_state': random_state if isinstance(random_state, int) else None,
        'extrapolation': 'nan',
    }


def propagate_continuous_uncertainty(
    proxy_values: np.ndarray,
    age_ensembles: np.ndarray,
    calibration_error: float,
    time_grid: np.ndarray,
    composite_func: Optional[Callable] = None,
    n_members: int = 500,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """连续值代理三层不确定性传播（年龄+校准+采样）。

    专为单一连续值代理序列设计的不确定性传播函数。
    与 validation.propagate_three_layer_uncertainty 的区别：
    后者处理多站点矩阵，本函数处理单站点序列。

    Parameters
    ----------
    proxy_values : np.ndarray
        单站点代理值 (n_depths,)。
    age_ensembles : np.ndarray
        年龄集合 (n_ensemble, n_depths)。
    calibration_error : float
        校准误差（来自 RMSEP），作为正态噪声标准差。
    time_grid : np.ndarray
        统一时间网格 (n_bins,)。
    composite_func : callable, optional
        自定义合成函数。默认为线性插值到 time_grid。
    n_members : int, optional
        集合成员数，默认 500。

    Returns
    -------
    Dict
        {'ensembles': np.ndarray (n_members, n_bins),
         'uncertainty_band': dict, 'n_members': int, 'layers': list}
    """
    proxy_arr = as_float_array(proxy_values, 'proxy_values', ndim=1, allow_nan=False, min_size=2)
    age_arr = as_float_array(age_ensembles, 'age_ensembles', ndim=2, allow_nan=False)
    if age_arr.shape[1] != len(proxy_arr):
        raise ValueError('age_ensembles 的样本维度必须与 proxy_values 一致。')
    if calibration_error < 0 or not np.isfinite(calibration_error):
        raise ValueError('calibration_error 必须为非负有限值。')
    if n_members < 2:
        raise ValueError('n_members 至少为 2。')
    grid = as_float_array(time_grid, 'time_grid', ndim=1, allow_nan=False)
    n_ens_pool = age_arr.shape[0]
    rng = get_rng(random_state)
    ensembles = np.full((n_members, len(grid)), np.nan)

    for i in range(n_members):
        # 1. 年龄层：采样一个年龄成员（保持地层单调性）
        ages = age_arr[i % n_ens_pool]

        # 2. 校准层：添加校准误差正态噪声
        noise = rng.normal(0, calibration_error, size=len(proxy_arr))
        noisy_values = proxy_arr + noise

        # 3. 采样层：Bootstrap 重采样（对深度点重采样）
        n_depths = len(proxy_arr)
        boot_idx = rng.choice(n_depths, size=n_depths, replace=True)
        boot_ages = ages[boot_idx]
        boot_values = noisy_values[boot_idx]
        # 保持单调
        sort_idx = np.argsort(boot_ages)
        boot_ages = boot_ages[sort_idx]
        boot_values = boot_values[sort_idx]

        # 插值到统一网格
        if composite_func is not None:
            ensembles[i] = composite_func(boot_ages, boot_values, grid)
        else:
            ensembles[i] = interpolate_no_extrapolation(
                boot_ages, boot_values, grid, name=f'member_{i}'
            )

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
        'method': 'three-layer Monte Carlo (continuous proxy)',
        'random_state': random_state if isinstance(random_state, int) else None,
    }


def cross_validate_calibration(
    proxy_x: np.ndarray,
    climate_y: np.ndarray,
    calibration_func: Optional[Callable] = None,
    method: str = 'loocv',
    n_splits: int = 5,
) -> Dict:
    """连续值代理校准模型交叉验证。

    校准模型（如 δDwax-δDp 线性回归、brGDGTs-MBT 温度校准）须通过
    留一交叉验证（LOOCV）或 k-fold 交叉验证评估预测能力。

    Parameters
    ----------
    proxy_x : np.ndarray
        校准集代理值 (n,)。
    climate_y : np.ndarray
        校准集气候变量值 (n,)。
    calibration_func : callable, optional
        校准函数，接受 (X_train, y_train) 返回模型对象。
        默认使用线性回归。
    method : str, optional
        验证方法：'loocv' (留一, 默认)、'kfold' 或按时间顺序的
        'timeseries'。时间序列优先使用 'timeseries'。
    n_splits : int, optional
        kfold/timeseries 的折数，默认 5。

    Returns
    -------
    Dict
        {'predictions': np.ndarray, 'residuals': np.ndarray,
         'rmsep': float, 'r2': float, 'method': str}
    """
    proxy_x = as_float_array(proxy_x, 'proxy_x', ndim=1, allow_nan=False)
    climate_y = as_float_array(climate_y, 'climate_y', ndim=1, allow_nan=False)
    if len(proxy_x) != len(climate_y):
        raise ValueError('proxy_x 与 climate_y 长度必须一致。')
    n = len(proxy_x)
    if n < 3:
        raise ValueError('交叉验证至少需要三个样本。')
    predictions = np.zeros(n)

    if calibration_func is None:
        # 默认线性回归
        def calibration_func(X_train, y_train):
            s, i, _, _, _ = sp_stats.linregress(X_train, y_train)
            class _M:
                def predict(self, X): return s * X + i
            return _M()

    if method == 'loocv':
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            model = calibration_func(proxy_x[mask], climate_y[mask])
            predictions[i] = model.predict(proxy_x[~mask])[0]

    elif method == 'kfold':
        try:
            from sklearn.model_selection import KFold
        except ImportError as exc:
            raise ImportError(
                'kfold 需要可选依赖 scikit-learn；请安装 requirements-optional.txt，'
                '或使用 loocv/timeseries。'
            ) from exc
        kf = KFold(n_splits=min(max(2, n_splits), n), shuffle=False)
        for train_idx, test_idx in kf.split(proxy_x):
            model = calibration_func(proxy_x[train_idx], climate_y[train_idx])
            predictions[test_idx] = model.predict(proxy_x[test_idx])

    elif method == 'timeseries':
        try:
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError as exc:
            raise ImportError(
                'timeseries 需要可选依赖 scikit-learn；请安装 requirements-optional.txt。'
            ) from exc
        n_split = min(max(2, n_splits), n - 1)
        splitter = TimeSeriesSplit(n_splits=n_split)
        for train_idx, test_idx in splitter.split(proxy_x):
            model = calibration_func(proxy_x[train_idx], climate_y[train_idx])
            predictions[test_idx] = model.predict(proxy_x[test_idx])
        # The earliest observations cannot be scored without a training history.
        first_scored = min(test_idx[0] for _, test_idx in splitter.split(proxy_x))
        predictions[:first_scored] = np.nan

    else:
        raise ValueError("method 须为 'loocv'/'kfold'/'timeseries'")

    valid = np.isfinite(predictions)
    residuals = climate_y - predictions
    rmsep_val = np.sqrt(np.mean(residuals[valid] ** 2))
    ss_res = np.sum(residuals[valid] ** 2)
    ss_tot = np.sum((climate_y[valid] - np.mean(climate_y[valid])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'predictions': predictions,
        'residuals': residuals,
        'rmsep': rmsep_val,
        'r2': r2,
        'method': method,
        'n': n,
        'n_scored': int(valid.sum()),
    }


def proxy_comparison(
    proxy_a_values: np.ndarray,
    proxy_b_values: np.ndarray,
    ages: Optional[np.ndarray] = None,
    n_boot: int = 10000,
    agreement_threshold: Optional[float] = None,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """双代理交叉验证：评估两个独立代理的一致性。

    当两个独立代理（如 δDwax 和 brGDGTs）重建同一气候变量时，
    其一致性是结论可信度的重要证据。

    Parameters
    ----------
    proxy_a_values : np.ndarray
        代理 A 值 (n,)。
    proxy_b_values : np.ndarray
        代理 B 值 (n,)。
    ages : np.ndarray, optional
        年龄数组，用于时序图。
    n_boot : int, optional
        Bootstrap 次数，默认 10000。

    Returns
    -------
    Dict
        {'correlation': float, 'ci_lower': float, 'ci_upper': float,
         'agreement': bool, 'mean_diff': float, 'method': str}
    """
    from scipy.stats import bootstrap

    a = as_float_array(proxy_a_values, 'proxy_a_values', ndim=1)
    b = as_float_array(proxy_b_values, 'proxy_b_values', ndim=1)
    if len(a) != len(b):
        raise ValueError('两个代理序列长度必须一致。')
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3:
        raise ValueError('代理比较至少需要三个共同有限观测。')

    # Pearson 相关
    r, p = sp_stats.pearsonr(a, b)

    # Bootstrap 置信区间
    def corr_stat(x, y, axis=-1):
        mx = np.mean(x, axis=axis)
        my = np.mean(y, axis=axis)
        mx2 = np.mean(x * x, axis=axis)
        my2 = np.mean(y * y, axis=axis)
        mxy = np.mean(x * y, axis=axis)
        cov = mxy - mx * my
        sx = np.sqrt(mx2 - mx ** 2)
        sy = np.sqrt(my2 - my ** 2)
        return np.divide(cov, sx * sy, out=np.full_like(cov, np.nan, dtype=float), where=(sx > 0) & (sy > 0))

    try:
        result = bootstrap(
            (a, b), statistic=corr_stat,
            n_resamples=n_boot, method='BCa',
            confidence_level=0.95, paired=True,
            random_state=get_rng(random_state),
        )
        ci_lower = result.confidence_interval.low
        ci_upper = result.confidence_interval.high
    except Exception as exc:
        raise ValueError(f'代理相关系数 bootstrap 失败：{exc}') from exc

    mean_diff = np.mean(a - b)
    rmse = np.sqrt(np.mean((a - b) ** 2))
    agreement = None if agreement_threshold is None else abs(mean_diff) <= agreement_threshold

    return {
        'correlation': float(r),
        'p_value': float(p),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'agreement': agreement,
        'mean_diff': float(mean_diff),
        'rmse': float(rmse),
        'method': 'paired Pearson association + BCa',
        'n_boot': n_boot,
        'agreement_threshold': agreement_threshold,
    }

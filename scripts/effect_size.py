"""
经典效应量模块（条件模块）。
涵盖 log response ratio、Hedges' d、Bootstrap BCa 置信区间、LOOCV、RMSEP。

文献来源：
- Hedges 1999 [6]: log response ratio, Hedges' d, 方差公式
- Izdebski 2022 [1]: Bootstrap BCa 方法
- Kaufman 2020 [2]: RMSEP 预测精度
- Hall 1988 [11]: BCa 置信区间理论

适用边界：仅在数据存在配对比较结构时激活（推断值vs真值 或 事件前vs事件后）。
纯时序叠加数据不适用（违反独立性假设），详见 references/methodology_gaps.md。
"""

from typing import Callable, Dict, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.stats import bootstrap, norm

try:
    from ._utils import as_float_array, get_rng, validate_same_length
except ImportError:  # pragma: no cover - supports direct script imports
    from _utils import as_float_array, get_rng, validate_same_length


def log_response_ratio(
    x_treatment: np.ndarray,
    x_control: np.ndarray,
) -> np.ndarray:
    """Hedges 1999 log response ratio: L = ln(X_T / X_C)。

    取对数的原因是比值型数据在原始尺度上偏态，对数变换后近似正态。
    X_T 为处理组均值，X_C 为对照组均值。
    在古生态学场景中：
    - 场景一：X_T = 代用指标推断值，X_C = 观测真值
    - 场景三：X_T = 事件后均值，X_C = 事件前均值

    Parameters
    ----------
    x_treatment : np.ndarray
        处理组值 (n,)。
    x_control : np.ndarray
        对照组值 (n,)。

    Returns
    -------
    np.ndarray
        log response ratio 数组 (n,)。

    Raises
    ------
    ValueError
        若数据含零值或负值（log response ratio 无定义）。
    """
    treatment = as_float_array(x_treatment, 'x_treatment', ndim=1, allow_nan=False)
    control = as_float_array(x_control, 'x_control', ndim=1, allow_nan=False)
    validate_same_length(('x_treatment', treatment), ('x_control', control))
    if np.any(treatment <= 0) or np.any(control <= 0):
        raise ValueError(
            "log response ratio 要求 X_T 和 X_C 均为正值。"
            "含零值或负值时请改用 hedges_d()。"
        )
    return np.log(treatment / control)


def log_response_ratio_variance(
    x_t: np.ndarray,
    x_c: np.ndarray,
    s_t: float,
    s_c: float,
    n_t: int,
    n_c: int,
) -> float:
    """Hedges 1999 log response ratio 方差。

    Var(L) = (s_T² / (n_T · X_T²)) + (s_C² / (n_C · X_C²))

    Parameters
    ----------
    x_t, x_c : np.ndarray
        处理组和对照组均值。
    s_t, s_c : float
        处理组和对照组标准差。
    n_t, n_c : int
        处理组和对照组样本量。

    Returns
    -------
    float
        log response ratio 的方差。
    """
    x_t = as_float_array(x_t, 'x_t', ndim=1, allow_nan=False)
    x_c = as_float_array(x_c, 'x_c', ndim=1, allow_nan=False)
    if n_t < 1 or n_c < 1 or s_t < 0 or s_c < 0:
        raise ValueError('样本量必须为正，标准差必须非负。')
    mean_t = float(np.mean(x_t))
    mean_c = float(np.mean(x_c))
    if mean_t <= 0 or mean_c <= 0:
        raise ValueError('log response ratio 的均值必须为正。')
    return float((s_t ** 2 / (n_t * mean_t ** 2)) + (s_c ** 2 / (n_c * mean_c ** 2)))


def hedges_d(
    x_treatment: np.ndarray,
    x_control: np.ndarray,
) -> Dict:
    """Hedges 1999 Hedges' d: 标准化均值差异 + 小样本校正。

    当数据包含零值或负值时，log response ratio 无定义，改用 Hedges' d。
    Hedges' d 不受零值限制，但丢失比值含义。

    d = (X_T - X_C) / s_pooled × J(n_T + n_C - 3)

    Parameters
    ----------
    x_treatment : np.ndarray
        处理组值 (n,)。
    x_control : np.ndarray
        对照组值 (n,)。

    Returns
    -------
    Dict
        {'d': float, 'variance': float, 'J_correction': float, 'n_t': int, 'n_c': int}
    """
    treatment = as_float_array(x_treatment, 'x_treatment', ndim=1, allow_nan=False, min_size=2)
    control = as_float_array(x_control, 'x_control', ndim=1, allow_nan=False, min_size=2)
    n_t = len(treatment)
    n_c = len(control)
    mean_t = np.mean(treatment)
    mean_c = np.mean(control)
    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)

    # 合并标准差
    s_pooled = np.sqrt(
        ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2)
    )

    if not np.isfinite(s_pooled):
        raise ValueError('合并标准差不是有限值。')
    if s_pooled == 0:
        if mean_t == mean_c:
            d_raw = 0.0
        else:
            raise ValueError('两组方差均为零但均值不同，Hedges\' g 无有限定义。')
    else:
        d_raw = (mean_t - mean_c) / s_pooled

    # 小样本校正因子 J(df)
    df = n_t + n_c - 2
    J = 1 - (3 / (4 * df - 1))
    d = d_raw * J

    # 方差近似
    variance = (n_t + n_c) / (n_t * n_c) + d ** 2 / (2 * (n_t + n_c))

    return {
        'd': d,
        'variance': variance,
        'J_correction': J,
        'n_t': n_t,
        'n_c': n_c,
    }


def effect_size_bca(
    x_treatment: np.ndarray,
    x_control: np.ndarray,
    effect_type: str = 'lnrr',
    n_boot: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """Izdebski 2022 BCa 方法估计效应量置信区间。

    使用 scipy.stats.bootstrap (method='BCa') 估计效应量置信区间。
    BCa (bias-corrected and accelerated) 修正了偏度和加速因子 (Hall 1988)。
    要求 n > 20 以保证 BCa 置信区间的稳定性。

    Parameters
    ----------
    x_treatment : np.ndarray
        处理组值 (n,)。
    x_control : np.ndarray
        对照组值 (n,)。
    effect_type : str, optional
        效应量类型：'lnrr' (log response ratio, 默认) 或 'd' (Hedges' d)。
    n_boot : int, optional
        Bootstrap 重采样次数，默认 10000 (Izdebski 2022)。
    confidence_level : float, optional
        置信水平，默认 0.95。

    Returns
    -------
    Dict
        {'effect_size': float, 'ci_lower': float, 'ci_upper': float,
         'n_t': int, 'n_c': int, 'significant': bool, 'method': 'BCa'}
    """
    treatment = as_float_array(x_treatment, 'x_treatment', ndim=1, allow_nan=False)
    control = as_float_array(x_control, 'x_control', ndim=1, allow_nan=False)
    if n_boot < 100:
        raise ValueError('n_boot 至少为 100。')
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level 必须位于 (0, 1)。')
    n_t, n_c = len(treatment), len(control)

    if min(n_t, n_c) < 20:
        warnings.warn(
            f"样本量 n_t={n_t}, n_c={n_c}，BCa 加速因子可能不稳定。",
            RuntimeWarning,
        )

    if effect_type == 'lnrr':
        validate_same_length(('x_treatment', treatment), ('x_control', control))
        ratios = log_response_ratio(treatment, control)
        point_estimate = np.mean(ratios)
        data = (ratios,)

        def statistic(values, axis=-1):
            return np.nanmean(values, axis=axis)

    elif effect_type == 'd':
        d_result = hedges_d(treatment, control)
        point_estimate = d_result['d']

        def statistic(treatment_values, control_values, axis=-1):
            mean_t = np.mean(treatment_values, axis=axis)
            mean_c = np.mean(control_values, axis=axis)
            var_t = np.var(treatment_values, axis=axis, ddof=1)
            var_c = np.var(control_values, axis=axis, ddof=1)
            pooled = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2))
            df = n_t + n_c - 2
            correction = 1 - (3 / (4 * df - 1))
            return correction * np.divide(
                mean_t - mean_c,
                pooled,
                out=np.zeros_like(mean_t, dtype=float),
                where=pooled > 0,
            )
        data = (treatment, control)

    else:
        raise ValueError("effect_type 须为 'lnrr' 或 'd'")

    rng = get_rng(random_state)
    try:
        result = bootstrap(
            data,
            statistic=statistic,
            n_resamples=n_boot,
            method='BCa',
            confidence_level=confidence_level,
            random_state=rng,
        )
        ci_method = 'BCa'
    except Exception as exc:
        warnings.warn(
            f'BCa 计算失败，改用 percentile 区间：{exc}',
            RuntimeWarning,
        )
        result = bootstrap(
            data,
            statistic=statistic,
            n_resamples=n_boot,
            method='percentile',
            confidence_level=confidence_level,
            random_state=rng,
        )
        ci_method = 'percentile_fallback'

    ci_lower = float(result.confidence_interval.low)
    ci_upper = float(result.confidence_interval.high)
    if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        raise ValueError('Bootstrap 区间不是有限值；请检查样本量、方差和重复值。')
    significant = not (ci_lower <= 0 <= ci_upper)

    return {
        'effect_size': float(point_estimate),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_t': n_t,
        'n_c': n_c,
        'significant': significant,
        'method': ci_method,
        'effect_type': effect_type,
        'n_boot': n_boot,
    }


def rmsep(predicted: np.ndarray, observed: np.ndarray) -> Dict:
    """Kaufman 2020 RMSEP: Root Mean Square Error of Prediction。

    代理不确定性量化标准指标，用于报告校准模型预测精度。

    Parameters
    ----------
    predicted : np.ndarray
        预测值 (n,)。
    observed : np.ndarray
        观测值 (n,)。

    Returns
    -------
    Dict
        {'rmsep': float, 'bias': float, 'n': int}
    """
    predicted = as_float_array(predicted, 'predicted', ndim=1)
    observed = as_float_array(observed, 'observed', ndim=1)
    validate_same_length(('predicted', predicted), ('observed', observed))
    valid = np.isfinite(predicted) & np.isfinite(observed)
    if not np.any(valid):
        raise ValueError('predicted 与 observed 没有共同的有限观测。')
    residuals = predicted[valid] - observed[valid]
    n = len(residuals)
    rmsep_val = np.sqrt(np.mean(residuals ** 2))
    bias = np.mean(residuals)

    return {
        'rmsep': rmsep_val,
        'bias': bias,
        'n': n,
        'n_missing': int(len(predicted) - n),
    }


def loocv(
    calibration_func: Callable,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict:
    """Birks 2010 留一交叉验证 (LOOCV) 评估校准模型。

    每次留出一个样本，用剩余样本拟合校准模型，预测留出样本。
    须在校准集上进行，不可在独立验证集上混合使用。

    Parameters
    ----------
    calibration_func : callable
        校准函数，接受 (X_train, y_train) 返回模型对象（具有 predict 方法）。
    X : np.ndarray
        特征矩阵 (n_samples, n_features)。
    y : np.ndarray
        目标值 (n_samples,)。

    Returns
    -------
    Dict
        {'predictions': np.ndarray, 'residuals': np.ndarray, 'rmsep': float, 'r2': float}
    """
    n = len(y)
    predictions = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = calibration_func(X[mask], y[mask])
        predictions[i] = model.predict(X[~mask])[0] if hasattr(model, 'predict') else model(X[~mask])[0]

    residuals = y - predictions
    rmsep_val = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        'predictions': predictions,
        'residuals': residuals,
        'rmsep': rmsep_val,
        'r2': r2,
    }


def quasi_experiment_effect_size(
    before_values: np.ndarray,
    after_values: np.ndarray,
) -> Dict:
    """准实验设计效应量计算 + 标注。

    场景三专用：对已识别为显著的变化，计算 log response ratio 量化变化幅度。
    须明确标注为准实验设计 (before-after, 非随机对照)，效应量解释为
    "关联强度" 而非 "因果效应"。

    残余混杂变量（如气候变化、人口迁移）可能贡献了观测到的变化。

    Parameters
    ----------
    before_values : np.ndarray
        事件前子时段指标值 (n,)。
    after_values : np.ndarray
        事件后子时段指标值 (n,)。

    Returns
    -------
    Dict
        {'effect_size': float, 'design': str, 'interpretation': str, 'note': str}
    """
    mean_before = np.mean(before_values)
    mean_after = np.mean(after_values)

    if mean_before <= 0 or mean_after <= 0:
        # 含零值时改用 Hedges' d
        d_result = hedges_d(after_values, before_values)
        return {
            'effect_size': d_result['d'],
            'effect_type': 'hedges_d',
            'design': 'quasi-experimental (before-after, non-randomized)',
            'interpretation': '关联强度（非因果效应）',
            'note': '准实验设计，效应量解释为关联强度而非因果效应。'
                    '残余混杂变量可能贡献了观测到的变化。',
        }

    lnrr = np.log(mean_after / mean_before)

    return {
        'effect_size': lnrr,
        'effect_type': 'log_response_ratio',
        'design': 'quasi-experimental (before-after, non-randomized)',
        'interpretation': '关联强度（非因果效应）',
        'note': '准实验设计，效应量解释为关联强度而非因果效应。'
                '残余混杂变量可能贡献了观测到的变化。',
    }

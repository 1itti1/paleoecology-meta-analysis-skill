"""
经典效应量模块（条件模块）。
涵盖 log response ratio、Hedges' d、Bootstrap BCa 置信区间、LOOCV、RMSEP。

文献来源：
- Hedges 1999 [6]: log response ratio, Hedges' d, 方差公式
- Izdebski 2022 [1]: Bootstrap BCa 方法
- Kaufman 2020 [2]: RMSEP 预测精度
- Hall 1988 [11]: BCa 置信区间理论

适用边界：区分独立两组与完整配对；配对对数比不等于研究级 ROM。
普通 bootstrap 不处理时间自相关或研究内聚类，时序资料须另行建模。
"""

from typing import Callable, Dict, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.stats import bootstrap

try:
    from ._utils import as_float_array, get_rng, validate_same_length
except ImportError:  # pragma: no cover - supports direct script imports
    from _utils import as_float_array, get_rng, validate_same_length


def _validate_design(design: str) -> None:
    if design not in ('independent', 'paired'):
        raise ValueError("design 须为 'independent' 或 'paired'；其他设计需专用模型。")


def log_response_ratio(
    x_treatment: np.ndarray,
    x_control: np.ndarray,
) -> np.ndarray:
    """逐对返回 log(T_i/C_i)，保留历史数组接口。

    其平均值是 mean(log(T/C))，不是研究级 ROM 的
    log(mean(T)/mean(C))；两者甚至可能方向相反。研究级 ROM 请用
    effect_size_bca(..., effect_type='lnrr', design=...)。

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
            "逐对 log ratio 要求 T_i 和 C_i 均为正值；"
            "其他效应量需根据研究目标和设计重新选择。"
        )
    return np.log(treatment) - np.log(control)


def log_response_ratio_variance(
    x_t: np.ndarray,
    x_c: np.ndarray,
    s_t: float,
    s_c: float,
    n_t: int,
    n_c: int,
    *,
    design: str = 'independent',
    paired_covariance: Optional[float] = None,
) -> float:
    """研究级 ROM = log(mean(T)/mean(C)) 的 delta 方差。

    独立组：s_T²/(n_T*mean_T²) + s_C²/(n_C*mean_C²)。
    配对组还须减去 2*paired_covariance/(n_pairs*mean_T*mean_C)。
    paired_covariance 是个体层面 Cov(T_i,C_i)，不是均值的协方差；
    必须显式提供，不能默认 0，也不从数组偷偷推断。
    本函数不是 mean(log(T/C)) 的方差函数。

    Parameters
    ----------
    x_t, x_c : array-like or float
        原始观测数组或单个算术均值，函数使用其均值。
    s_t, s_c : float
        处理组和对照组标准差。
    n_t, n_c : int
        处理组和对照组样本量。

    Returns
    -------
    float
        log response ratio 的方差。
    """
    _validate_design(design)
    x_t = as_float_array(np.atleast_1d(x_t), 'x_t', ndim=1, allow_nan=False)
    x_c = as_float_array(np.atleast_1d(x_c), 'x_c', ndim=1, allow_nan=False)
    for name, value in (('n_t', n_t), ('n_c', n_c)):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not np.isfinite(value) or value < 1 or int(value) != value):
            raise ValueError(f'{name} 必须为正整数。')
    if not np.all(np.isfinite([s_t, s_c])) or s_t < 0 or s_c < 0:
        raise ValueError('标准差必须有限且非负。')
    mean_t = float(np.mean(x_t))
    mean_c = float(np.mean(x_c))
    if mean_t <= 0 or mean_c <= 0:
        raise ValueError('log response ratio 的均值必须为正。')
    variance = s_t ** 2 / (n_t * mean_t ** 2) + s_c ** 2 / (n_c * mean_c ** 2)
    if design == 'paired':
        if n_t != n_c or n_t < 2:
            raise ValueError('paired ROM 方差要求相同且至少为 2 的完整配对样本量。')
        if paired_covariance is None:
            raise ValueError("design='paired' 必须显式提供 paired_covariance=Cov(T_i,C_i)。")
        if not np.isfinite(paired_covariance):
            raise ValueError('paired_covariance 必须有限。')
        bound = s_t * s_c
        if abs(paired_covariance) > bound and not np.isclose(
            abs(paired_covariance), bound, rtol=1e-12, atol=0.0
        ):
            raise ValueError('paired_covariance 必须满足 |Cov(T,C)| <= s_t*s_c。')
        variance -= 2 * paired_covariance / (n_t * mean_t * mean_c)
        # 仅吸收合法协方差边界上的浮点舍入误差。
        variance = max(0.0, variance)
    elif paired_covariance is not None:
        raise ValueError("paired_covariance 仅适用于 design='paired'，不能在独立组计算中忽略它。")
    return float(variance)


def hedges_d(
    x_treatment: np.ndarray,
    x_control: np.ndarray,
    *,
    design: str = 'independent',
) -> Dict:
    """仅独立两组的 Hedges' g（保留历史函数名和 d 字段）。

    使用 J(df) ~= 1 - 3/(4*df-1)，df=n_T+n_C-2；方差为
    (n_T+n_C)/(n_T*n_C) + g²/(2*(n_T+n_C)) 的独立组大样本近似。
    配对 SMD 有多种不同分母/目标，本接口明确拒绝 paired，不能把此方差
    当作配对方差。两组长度可不同。

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
    _validate_design(design)
    if design != 'independent':
        raise ValueError("hedges_d 仅实现 design='independent'；配对 SMD 需要专用估计量和协方差模型。")
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
        'design': design,
        'estimand': 'independent_groups_standardized_mean_difference',
        'variance_method': 'independent_groups_large_sample',
    }


def effect_size_bca(
    x_treatment: np.ndarray,
    x_control: np.ndarray,
    effect_type: str = 'lnrr',
    n_boot: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    *,
    design: str = 'independent',
) -> Dict:
    """按明确的效应目标和采样设计计算 BCa 区间。

    lnrr 是研究级 ROM：log(mean(T)/mean(C))，不是历史版本曾返回的
    mean(log(T/C))。需要历史目标时明确使用 paired_log_ratio 和 paired。
    independent 为兼容默认值，分别重抽两组，允许不等长；paired 保留
    完整配对索引并用于 jackknife。普通重抽不处理时间/空间自相关。

    Parameters
    ----------
    x_treatment : np.ndarray
        处理组值 (n,)。
    x_control : np.ndarray
        对照组值 (n,)。
    effect_type : str, optional
        'lnrr'、'paired_log_ratio' 或 'd'（仅独立组 Hedges' g）。
    n_boot : int, optional
        Bootstrap 重采样次数，默认 10000 (Izdebski 2022)。
    confidence_level : float, optional
        置信水平，默认 0.95。
    design : {'independent', 'paired'}, optional
        配对设计必须显式选择；配对 SMD 暂未实现。

    Returns
    -------
    Dict
        {'effect_size': float, 'ci_lower': float, 'ci_upper': float,
         'n_t': int, 'n_c': int, 'significant': bool, 'method': 'BCa'}
    """
    _validate_design(design)
    treatment = as_float_array(x_treatment, 'x_treatment', ndim=1, allow_nan=False, min_size=2)
    control = as_float_array(x_control, 'x_control', ndim=1, allow_nan=False, min_size=2)
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise ValueError('n_boot 必须为至少 100 的整数。')
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level 必须位于 (0, 1)。')
    n_t, n_c = len(treatment), len(control)
    if design == 'paired':
        validate_same_length(('x_treatment', treatment), ('x_control', control))
    if effect_type not in ('lnrr', 'paired_log_ratio', 'd'):
        raise ValueError("effect_type 须为 'lnrr'、'paired_log_ratio' 或 'd'。")
    if effect_type == 'paired_log_ratio' and design != 'paired':
        raise ValueError("paired_log_ratio 要求显式 design='paired'；独立组请使用 lnrr。")
    if effect_type == 'd' and design != 'independent':
        raise ValueError("effect_type='d' 仅支持 design='independent'；配对 SMD 尚未实现。")

    warning_messages = []
    if min(n_t, n_c) < 20:
        warning_messages.append(f"样本量 n_t={n_t}, n_c={n_c}，BCa 加速因子可能不稳定。")
        warnings.warn(warning_messages[-1], RuntimeWarning, stacklevel=2)

    if effect_type == 'lnrr':
        if np.any(treatment <= 0) or np.any(control <= 0):
            raise ValueError('lnrr bootstrap 要求两组观测为正，保证所有重抽均值可取对数。')

        def statistic(treatment_values, control_values, axis=-1):
            return (np.log(np.mean(treatment_values, axis=axis))
                    - np.log(np.mean(control_values, axis=axis)))

        estimand = 'log(mean(T)/mean(C))'

    elif effect_type == 'paired_log_ratio':
        # 同时校验每个配对值；bootstrap 在两个原始数组上保留同索引。
        log_response_ratio(treatment, control)

        def statistic(treatment_values, control_values, axis=-1):
            return np.mean(np.log(treatment_values) - np.log(control_values), axis=axis)

        estimand = 'mean(log(T/C))'

    elif effect_type == 'd':
        hedges_d(treatment, control, design=design)

        def statistic(treatment_values, control_values, axis=-1):
            # BCa jackknife 改变样本量，不能沿用原始 n_t/n_c。
            nt = treatment_values.shape[axis]
            nc = control_values.shape[axis]
            mean_t = np.mean(treatment_values, axis=axis)
            mean_c = np.mean(control_values, axis=axis)
            var_t = np.var(treatment_values, axis=axis, ddof=1)
            var_c = np.var(control_values, axis=axis, ddof=1)
            pooled = np.sqrt(((nt - 1) * var_t + (nc - 1) * var_c) / (nt + nc - 2))
            df = nt + nc - 2
            correction = 1 - (3 / (4 * df - 1))
            return correction * np.divide(
                mean_t - mean_c,
                pooled,
                out=np.where(mean_t == mean_c, 0.0, np.nan),
                where=pooled > 0,
            )
        estimand = 'independent_groups_standardized_mean_difference'

    data = (treatment, control)
    point_estimate = statistic(treatment, control)
    rng = get_rng(random_state)
    try:
        result = bootstrap(
            data,
            statistic=statistic,
            n_resamples=n_boot,
            method='BCa',
            paired=(design == 'paired'),
            confidence_level=confidence_level,
            random_state=rng,
        )
        if not np.all(np.isfinite(result.confidence_interval)):
            raise ValueError('BCa 区间退化或不是有限值。')
        ci_method = 'BCa'
    except (ValueError, FloatingPointError) as exc:
        warning_messages.append(f'BCa 计算失败，改用 percentile 区间：{exc}')
        warnings.warn(warning_messages[-1], RuntimeWarning, stacklevel=2)
        result = bootstrap(
            data,
            statistic=statistic,
            n_resamples=n_boot,
            method='percentile',
            paired=(design == 'paired'),
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
        'estimand': estimand,
        'design': design,
        'resampling_unit': 'pair' if design == 'paired' else 'independent_group',
        'warnings': warning_messages,
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

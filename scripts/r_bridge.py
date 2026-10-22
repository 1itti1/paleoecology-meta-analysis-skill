"""
R 桥接模块：通过 subprocess 调用 Rscript + metafor 包。

混合架构核心：
- R + metafor 可用时：调用 metafor::rma() 等函数，获得完整的经典 meta 分析能力
- R 不可用时：回退到 effect_size.py 中的 Python 实现（功能子集）

数据交换方式：Python → JSON → R → JSON → Python
不依赖 rpy2，仅需 R + metafor 安装。

文献来源：
- metafor: Viechtbauer 2010, Journal of Statistical Software
  https://www.jstatsoft.org/article/view/v036i03
- Hartung-Knapp 校正: Hartung & Knapp 2001, Biometrics
- I² 异质性: Higgins & Thompson 2002, JRSS-A
- Egger 检验: Egger et al. 1997, BMJ
"""

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _json_for_r(payload: Dict) -> str:
    """Serialize JSON safely for an R single-quoted string literal.

    Non-finite numeric values are converted to null because JSON and R do not
    share the same spelling for NaN/Inf in a quoted payload. Backslashes and
    apostrophes are escaped before embedding the JSON in R code.
    """
    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    encoded = json.dumps(clean(payload), ensure_ascii=True, separators=(',', ':'))
    return encoded.replace('\\', '\\\\').replace("'", "\\'")


# ---------------------------------------------------------------------------
# R 环境检测
# ---------------------------------------------------------------------------

# 常见 R 安装路径（Windows / macOS / Linux）
_R_PATHS = [
    # Windows
    r'C:\Program Files\R\R-4.6.1\bin\Rscript.exe',
    r'C:\Program Files\R\R-4.6.0\bin\Rscript.exe',
    r'C:\Program Files\R\R-4.5.0\bin\Rscript.exe',
    r'C:\Program Files\R\R-4.4.0\bin\Rscript.exe',
    r'C:\Program Files\R\R-4.3.0\bin\Rscript.exe',
    # macOS Homebrew
    '/usr/local/bin/Rscript',
    '/opt/homebrew/bin/Rscript',
    # Linux
    '/usr/bin/Rscript',
    '/usr/local/bin/Rscript',
]


def find_rscript() -> Optional[str]:
    """查找 Rscript 可执行文件路径。

    搜索顺序：
    1. 显式 R_SCRIPT 环境变量
    2. PATH 环境变量中的 Rscript
    3. 常见安装路径列表及 Windows 安装目录

    Returns
    -------
    str or None
        Rscript 路径，未找到返回 None。
    """
    configured = os.environ.get('R_SCRIPT', '').strip().strip('"')
    if configured:
        candidate = os.path.expandvars(os.path.expanduser(configured))
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        if os.path.isfile(candidate):
            return candidate
        raise ValueError(f'R_SCRIPT 指向不存在或不可执行的 Rscript：{configured}')

    resolved = shutil.which('Rscript')
    if resolved:
        return resolved

    # 2. 尝试常见路径
    for path in _R_PATHS:
        if os.path.isfile(path):
            return path

    # 3. Windows: 搜索 R 版本目录
    if os.name == 'nt':
        r_base = r'C:\Program Files\R'
        if os.path.isdir(r_base):
            for ver in sorted(os.listdir(r_base), reverse=True):
                candidate = os.path.join(r_base, ver, 'bin', 'Rscript.exe')
                if os.path.isfile(candidate):
                    return candidate

    return None


def check_r_environment() -> Dict:
    """检查 R 环境和已安装的 meta 分析相关包。

    Returns
    -------
    Dict
        {'r_available': bool, 'rscript_path': str or None,
         'r_version': str or None, 'packages': dict}
    """
    rscript = find_rscript()

    if rscript is None:
        return {
            'r_available': False,
            'rscript_path': None,
            'r_version': None,
            'packages': {},
            'message': 'R 未找到。将回退到 Python 实现。安装 R: https://cran.r-project.org/',
        }

    # 获取 R 版本
    try:
        result = subprocess.run(
            [rscript, '--version'],
            capture_output=True, text=True, timeout=10,
        )
        r_version = result.stdout.strip().split('\n')[0]
    except Exception:
        r_version = 'unknown'

    # 检查包
    check_script = '''
    pkgs <- c("metafor", "meta", "compute.es", "metasens")
    result <- list()
    for (p in pkgs) {
        result[[p]] <- ifelse(requireNamespace(p, quietly=TRUE), "installed", "not_installed")
    }
    cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na="null", null="null"))
    '''

    # 写临时脚本
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.R', delete=False, encoding='utf-8'
    ) as f:
        f.write('if (!requireNamespace("jsonlite", quietly=TRUE)) {\n')
        f.write('  cat(\'{"metafor":"check_failed"}\')\n')
        f.write('  quit(status=0)\n')
        f.write('}\n')
        f.write(check_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [rscript, script_path],
            capture_output=True, text=True, timeout=30,
        )
        packages = json.loads(result.stdout.strip()) if result.stdout.strip() else {}
    except (json.JSONDecodeError, subprocess.TimeoutExpired):
        packages = {}
    finally:
        os.unlink(script_path)

    return {
        'r_available': True,
        'rscript_path': rscript,
        'r_version': r_version,
        'packages': packages,
        'metafor_ready': packages.get('metafor') == 'installed',
    }


# ---------------------------------------------------------------------------
# R 调用执行器
# ---------------------------------------------------------------------------

def _run_r_script(
    r_code: str,
    rscript_path: Optional[str] = None,
    timeout: int = 120,
) -> Dict:
    """执行 R 脚本并返回 JSON 结果。

    Parameters
    ----------
    r_code : str
        R 代码字符串。须最终输出 JSON 到 stdout。
    rscript_path : str, optional
        Rscript 路径。None 时自动查找。
    timeout : int, optional
        超时秒数，默认 120。

    Returns
    -------
    Dict
        {'success': bool, 'result': dict or None, 'error': str or None,
         'stdout': str, 'stderr': str}
    """
    if rscript_path is None:
        rscript_path = find_rscript()

    if rscript_path is None:
        return {
            'success': False,
            'result': None,
            'error': 'Rscript not found',
            'stdout': '',
            'stderr': '',
        }

    # 包装脚本：确保 jsonlite 可用，捕获错误
    wrapped = f'''
if (!requireNamespace("jsonlite", quietly=TRUE)) {{
  cat('{{"success":false,"error":"jsonlite not installed"}}')
  quit(status=0)
}}
tryCatch({{
  {r_code}
}}, error=function(e) {{
  cat(jsonlite::toJSON(list(success=FALSE, error=as.character(e$message)),
                      auto_unbox=TRUE, digits=NA, na="null", null="null"))
  quit(status=0)
}})
'''

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.R', delete=False, encoding='utf-8'
    ) as f:
        f.write(wrapped)
        script_path = f.name

    try:
        result = subprocess.run(
            [rscript_path, script_path],
            capture_output=True, text=True, timeout=timeout,
            encoding='utf-8',
        )

        output = result.stdout.strip()
        if not output:
            return {
                'success': False,
                'result': None,
                'error': f'Empty output. stderr: {result.stderr.strip()[:500]}',
                'stdout': result.stdout,
                'stderr': result.stderr,
            }

        parsed = json.loads(output)
        return {
            'success': parsed.get('success', True),
            'result': parsed,
            'error': parsed.get('error'),
            'stdout': result.stdout,
            'stderr': result.stderr,
        }

    except json.JSONDecodeError as e:
        return {
            'success': False,
            'result': None,
            'error': f'JSON decode error: {e}',
            'stdout': result.stdout[:500],
            'stderr': result.stderr[:500],
        }
    except OSError as exc:
        return {
            'success': False, 'result': None, 'error': str(exc),
            'stdout': '', 'stderr': '',
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'result': None,
            'error': f'R script timed out after {timeout}s',
            'stdout': '',
            'stderr': '',
        }
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# metafor 核心功能封装
# ---------------------------------------------------------------------------

def _validate_meta_inputs(effect_sizes, variances, *, min_studies=2):
    yi = np.asarray(effect_sizes, dtype=float)
    vi = np.asarray(variances, dtype=float)
    if yi.ndim != 1 or vi.ndim != 1 or len(yi) != len(vi):
        raise ValueError('effect_sizes 与 variances 必须是一维且长度一致。')
    if (len(yi) < min_studies or np.any(~np.isfinite(yi))
            or np.any(~np.isfinite(vi)) or np.any(vi <= 0)):
        raise ValueError(f'效应量必须有限，方差必须为正，且至少有 {min_studies} 个研究。')
    return yi, vi


def _check_independent_effects(vi, study_ids=None, sampling_covariance=None):
    """普通单层模型的能力检查，不替代高层科学分析路由。"""
    k = len(vi)
    if study_ids is not None:
        ids = np.asarray(study_ids, dtype=object)
        if ids.ndim != 1 or len(ids) != k:
            raise ValueError('study_ids 必须一维且与效应量长度一致。')
        seen = set()
        for study_id in ids:
            if (study_id is None or str(study_id).strip() == ''
                    or (isinstance(study_id, (float, np.floating)) and not np.isfinite(study_id))):
                raise ValueError('study_ids 不得缺失或为空。')
            try:
                if study_id in seen:
                    raise ValueError(
                        '同研究多效应不能进入 rma.uni 或 Python 独立组回退；'
                        '请交接支持 sampling V、随机结构和独立簇的多层/多变量接口'
                        '（如 rma.mv，本桥接未实现）。'
                    )
                seen.add(study_id)
            except TypeError as exc:
                raise ValueError('study_ids 必须由可哈希的标量标识组成。') from exc
    if sampling_covariance is not None:
        covariance = np.asarray(sampling_covariance, dtype=float)
        if covariance.shape != (k, k) or not np.all(np.isfinite(covariance)):
            raise ValueError('sampling_covariance 必须是有限的 k×k 矩阵。')
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError('sampling_covariance 必须对称。')
        if not np.allclose(np.diag(covariance), vi, rtol=1e-10, atol=0.0):
            raise ValueError('sampling_covariance 对角线必须与 variances 一致。')
        if np.any(covariance[~np.eye(k, dtype=bool)] != 0):
            raise ValueError(
                '非对角 sampling_covariance 不能被 rma.uni 或 Python 回退忽略；'
                '请交接完整 V 的多层/多变量接口（如 rma.mv，本桥接未实现）。'
            )
    return {
        'n_independent': k,
        'independence_assumption': (
            'one_effect_per_declared_study' if study_ids is not None
            else 'assumed_one_independent_study_per_effect'
        ),
        'sampling_covariance': 'validated_diagonal' if sampling_covariance is not None else 'diagonal_vi',
    }


def rma_random_effects(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    method: str = 'REML',
    knha: bool = True,
    rscript_path: Optional[str] = None,
    *,
    study_ids=None,
    sampling_covariance=None,
) -> Dict:
    """metafor::rma() 随机效应模型（Viechtbauer 2010）。

    经典 meta 分析核心函数：合并多个研究的效应量，估计总体效应和异质性。
    当 R 可用时调用 metafor，不可用时回退到 Python 的 DerSimonian-Laird 实现。

    Parameters
    ----------
    effect_sizes : np.ndarray
        各研究的效应量 (n_studies,)，如 log response ratio 或 Hedges' d。
    variances : np.ndarray
        各研究效应量的方差 (n_studies,)。
    method : str, optional
        估计方法：'REML' (默认, 限制最大似然), 'DL' (DerSimonian-Laird),
        'ML' (最大似然), 'EB' (经验贝叶斯), 'HS' (Hunter-Schmidt)。
    knha : bool, optional
        是否使用未修改的 Hartung-Knapp 校正，默认 True。
        包括 k=2 时使用 t(k-1)；它不是保证保守的加宽区间。
    rscript_path : str, optional
        Rscript 路径，None 时自动查找。
    study_ids : sequence, optional
        每效应所属研究；重复 ID 明确交接多层模型，不当作独立研究。
    sampling_covariance : array-like, optional
        只接受与 vi 一致的对角矩阵；非对角 V 需另行使用多变量接口。

    Returns
    -------
    Dict
        {'pooled_effect': float, 'se': float, 'ci_lower': float, 'ci_upper': float,
         'z_value': float, 'p_value': float, 'I2': float, 'tau2': float,
         'Q': float, 'Q_pvalue': float, 'method': str, 'knha': bool,
         'backend': 'metafor' or 'python'}
    """
    yi, vi = _validate_meta_inputs(effect_sizes, variances)
    independence = _check_independent_effects(vi, study_ids, sampling_covariance)

    # --- 尝试 R + metafor ---
    data_json = _json_for_r({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
        'method': method,
        'knha': knha,
        'test': 'knha' if knha else 'z',
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- rma(yi=data$yi, vi=data$vi, method=data$method, test=data$test)
result <- list(
    success = TRUE,
    pooled_effect = as.numeric(rma_fit$b),
    se = as.numeric(rma_fit$se),
    ci_lower = as.numeric(rma_fit$ci.lb),
    ci_upper = as.numeric(rma_fit$ci.ub),
    z_value = as.numeric(rma_fit$zval),
    p_value = as.numeric(rma_fit$pval),
    I2 = as.numeric(rma_fit$I2),
    tau2 = as.numeric(rma_fit$tau2),
    Q = as.numeric(rma_fit$QE),
    Q_pvalue = as.numeric(rma_fit$QEp),
    method = rma_fit$method,
    requested_method = data$method,
    knha = identical(rma_fit$test, "knha"),
    test = rma_fit$test,
    test_statistic = as.numeric(rma_fit$zval),
    df = if (rma_fit$test == "knha") rma_fit$k - rma_fit$p else NULL,
    model = "random_effects",
    model_class = "rma.uni",
    r_version = R.version.string,
    metafor_version = as.character(packageVersion("metafor")),
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na="null", null="null"))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        result = r_result['result']
        result.update(independence)
        result['test_distribution'] = 't' if result['test'] == 'knha' else 'normal'
        result['inference_available'] = result['se'] > 0
        if not result['inference_available']:
            result.update(test_statistic=None, z_value=None, p_value=None)
            result['warnings'] = ['KH 残差尺度为零，保留点估计和区间，不自动推断。']
        else:
            result['warnings'] = []
        return result

    # --- 回退到 Python ---
    result = _rma_python_fallback(yi, vi, method, knha)
    result.update(independence)
    result['fallback_reason'] = r_result.get('error')
    return result


def _rma_python_fallback(
    yi: np.ndarray,
    vi: np.ndarray,
    method: str = 'DL',
    knha: bool = True,
) -> Dict:
    """DerSimonian-Laird 随机效应模型的 Python 实现（回退）。

    当 metafor 不可用时使用此简化版本。
    仅支持 DL 方法，不含 REML/ML 等高级估计。

    文献：DerSimonian & Laird 1986, Controlled Clinical Trials
    """
    from scipy import stats as sp_stats

    yi, vi = _validate_meta_inputs(yi, vi)
    n = len(yi)

    # 固定效应权重
    w_fixed = 1.0 / vi

    # 固定效应估计
    pooled_fixed = np.sum(w_fixed * yi) / np.sum(w_fixed)

    # Cochran's Q
    Q = np.sum(w_fixed * (yi - pooled_fixed) ** 2)
    df = n - 1
    Q_pvalue = sp_stats.chi2.sf(Q, df)

    # tau² (DerSimonian-Laird 估计)
    c = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df) / c) if c > 0 else 0

    # I²
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0

    # 随机效应权重
    w_random = 1.0 / (vi + tau2)
    pooled_random = np.sum(w_random * yi) / np.sum(w_random)
    se_random = np.sqrt(1.0 / np.sum(w_random))

    # 未修改 KH：q = sum(w*(y-mu)^2)/(k-1)，SE = sqrt(q/sum(w))。
    # k=2 仍有 df=1，不得改用 z 后继续标作 KH。
    if knha:
        q = np.sum(w_random * (yi - pooled_random) ** 2) / df
        se_random = np.sqrt(q / np.sum(w_random))

    # Hartung-Knapp 使用 t 分布；没有 HK 时才使用正态近似。
    if knha:
        critical = sp_stats.t.ppf(0.975, df)
        ci_lower = pooled_random - critical * se_random
        ci_upper = pooled_random + critical * se_random
        test_name = 'knha'
    else:
        critical = sp_stats.norm.ppf(0.975)
        ci_lower = pooled_random - critical * se_random
        ci_upper = pooled_random + critical * se_random
        test_name = 'z'

    warning_messages = []
    if se_random > 0:
        test_value = float(pooled_random / se_random)
        p_value = float(2 * (sp_stats.t.sf(abs(test_value), df) if knha
                             else sp_stats.norm.sf(abs(test_value))))
    else:
        test_value = p_value = None
        warning_messages.append('KH 残差尺度为零，保留点估计和区间，不自动推断。')
    if str(method).upper() != 'DL':
        warning_messages.append(f'请求 {method}，Python 回退实际仅实现 DL；结果不冒充请求的估计量。')
        warnings.warn(warning_messages[-1], RuntimeWarning, stacklevel=2)

    return {
        'success': True,
        'pooled_effect': float(pooled_random),
        'se': float(se_random),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'z_value': test_value,  # 历史别名；KH 下实际为 t，见 test。
        'test_statistic': test_value,
        'p_value': p_value,
        'I2': float(I2),
        'tau2': float(tau2),
        'Q': float(Q),
        'Q_pvalue': float(Q_pvalue),
        'method': 'DL',
        'requested_method': method,
        'knha': bool(knha),
        'test': test_name,
        'df': int(df) if knha else None,
        'test_distribution': 't' if knha else 'normal',
        'model': 'random_effects',
        'model_class': 'DL_inverse_variance',
        'inference_available': bool(se_random > 0),
        'warnings': warning_messages,
        'backend': 'python',
        'note': '使用 DerSimonian-Laird 估计（metafor 不可用时的回退）。'
                '安装 metafor 可获得 REML/ML 等高级估计方法。',
    }


def meta_regression(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    moderators: np.ndarray,
    method: str = 'REML',
    rscript_path: Optional[str] = None,
) -> Dict:
    """metafor::rma() 混合效应模型（meta 回归）。

    检验连续型或分类型调节变量对效应量的影响。

    Parameters
    ----------
    effect_sizes : np.ndarray
        效应量 (n_studies,)。
    variances : np.ndarray
        效应量方差 (n_studies,)。
    moderators : np.ndarray
        调节变量 (n_studies, n_moderators) 或 (n_studies,)。
    method : str, optional
        估计方法，默认 'REML'。
    rscript_path : str, optional
        Rscript 路径。

    Returns
    -------
    Dict
        {'coefficients': list, 'se': list, 'p_values': list,
         'R2': float, 'Qm': float, 'Qm_pvalue': float,
         'Qe': float, 'Qe_pvalue': float, 'backend': str}
    """
    yi = np.asarray(effect_sizes, dtype=float)
    vi = np.asarray(variances, dtype=float)
    mods = np.asarray(moderators, dtype=float)
    if mods.ndim == 1:
        mods = mods[:, np.newaxis]

    data_json = _json_for_r({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
        'mods': mods.tolist(),
        'method': method,
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- rma(yi=data$yi, vi=data$vi, mods=as.matrix(data$mods), method=data$method)
result <- list(
    success = TRUE,
    coefficients = as.numeric(rma_fit$b),
    se = as.numeric(rma_fit$se),
    p_values = as.numeric(rma_fit$pval),
    z_values = as.numeric(rma_fit$zval),
    ci_lower = as.numeric(rma_fit$ci.lb),
    ci_upper = as.numeric(rma_fit$ci.ub),
    R2 = as.numeric(rma_fit$R2),
    Qm = as.numeric(rma_fit$QM),
    Qm_pvalue = as.numeric(rma_fit$QMp),
    Qe = as.numeric(rma_fit$QE),
    Qe_pvalue = as.numeric(rma_fit$QEp),
    tau2 = as.numeric(rma_fit$tau2),
    I2 = as.numeric(rma_fit$I2),
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na="null", null="null"))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        return r_result['result']

    return {
        'success': False,
        'error': 'meta 回归需要 metafor。' + str(r_result.get('error', '')),
        'backend': 'none',
        'note': 'Python 回退不支持 meta 回归。请安装 R + metafor: install.packages("metafor")',
    }


def egger_test(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    rscript_path: Optional[str] = None,
    *,
    study_ids=None,
    sampling_covariance=None,
) -> Dict:
    """经典 Egger 小研究效应回归，不输出发表偏倚二元裁决。

    两后端使用同一参数化：yi/sei = intercept + slope*(1/sei)，
    intercept、intercept_se、p_value 均属于该截距。metafor 的等价
    加权 yi ~ sei 模型中，sei 的系数才是这个截距，不能用 limit estimate。
    test='t'，df=k-2；z_value 是保留的历史统计量别名，不表示正态检验。
    少于 10 个独立研究或精度跨度极窄时保留可估计的有限诊断值和警告，
    但 inference_allowed=False，禁止自动显著性/偏倚裁决。study_ids 的
    重复及非对角 V 需依赖结构专用接口，不得按效应行数扩大独立样本量。
    """
    yi, vi = _validate_meta_inputs(effect_sizes, variances)
    independence = _check_independent_effects(vi, study_ids, sampling_covariance)
    k = len(yi)
    warning_messages = [
        'Egger 仅描述小研究效应/漏斗不对称，不能判定发表偏倚存在或不存在。'
    ]
    if k < 10:
        warning_messages.append(f'仅 {k} 个独立研究（少于 10）；保留探索性诊断，禁止自动推断。')
    if k < 3:
        return {
            'success': False, 'error': 'Egger 回归至少需要 3 个独立研究和非零残差自由度。',
            'backend': 'none', 'inference_allowed': False, 'warnings': warning_messages,
            **independence,
        }
    sei = np.sqrt(vi)
    precision = 1.0 / sei
    spread = float(np.ptp(precision))
    if spread == 0:
        return {
            'success': False, 'error': '所有标准误相同，Egger 截距与斜率不可分别识别。',
            'backend': 'none', 'inference_allowed': False, 'warnings': warning_messages,
            **independence,
        }
    narrow_precision = spread / float(np.max(precision)) < 1e-3
    if narrow_precision:
        warning_messages.append('研究精度跨度极窄，截距外推不稳定；禁止自动推断。')

    data_json = _json_for_r({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
egger <- regtest(data$yi, vi=data$vi, model="lm", predictor="sei", ret.fit=TRUE)
coefficients <- summary(egger$fit)$coefficients
result <- list(
    success = TRUE,
    test_statistic = as.numeric(coefficients[2, 3]),
    p_value = as.numeric(coefficients[2, 4]),
    intercept = as.numeric(coefficients[2, 1]),
    intercept_se = as.numeric(coefficients[2, 2]),
    slope = as.numeric(coefficients[1, 1]),
    slope_se = as.numeric(coefficients[1, 2]),
    slope_p_value = as.numeric(coefficients[1, 4]),
    r_version = R.version.string,
    metafor_version = as.character(packageVersion("metafor")),
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na="null", null="null"))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        result = r_result['result']
    else:
        from scipy import stats as sp_stats

        standardized = yi / sei
        x_centered = precision - np.mean(precision)
        y_centered = standardized - np.mean(standardized)
        ssx = float(np.dot(x_centered, x_centered))
        slope = float(np.dot(x_centered, y_centered) / ssx)
        intercept = float(np.mean(standardized) - slope * np.mean(precision))
        residuals = y_centered - slope * x_centered
        residual_variance = float(np.dot(residuals, residuals) / (k - 2))
        slope_se = float(np.sqrt(residual_variance / ssx))
        intercept_se = float(np.sqrt(residual_variance * (1.0 / k + np.mean(precision) ** 2 / ssx)))
        test_statistic = intercept / intercept_se if intercept_se > 0 else None
        result = {
            'success': True,
            'test_statistic': test_statistic,
            'p_value': float(2 * sp_stats.t.sf(abs(test_statistic), k - 2)) if test_statistic is not None else None,
            'intercept': intercept,
            'intercept_se': intercept_se,
            'slope': slope,
            'slope_se': slope_se,
            'slope_p_value': float(2 * sp_stats.t.sf(abs(slope / slope_se), k - 2)) if slope_se > 0 else None,
            'backend': 'python',
            'fallback_reason': r_result.get('error'),
        }

    finite_inference = (result.get('intercept_se') is not None
                        and result['intercept_se'] > 0
                        and result.get('p_value') is not None
                        and np.isfinite(result['p_value'])
                        and result.get('test_statistic') is not None
                        and np.isfinite(result['test_statistic']))
    if not finite_inference:
        warning_messages.append('截距检验的残差尺度退化；保留可估计系数，不自动推断。')
        result.update(test_statistic=None, p_value=None)
    result.update(independence)
    result.update({
        'bias_intercept': result['intercept'],
        'bias_intercept_se': result['intercept_se'],
        'intercept_p_value': result['p_value'],
        'z_value': result['test_statistic'],
        't_value': result['test_statistic'],
        'test': 't',
        'df': k - 2,
        'model': 'egger_precision_ols',
        'parameterization': 'yi/sei = intercept + slope*(1/sei)',
        'inference_allowed': bool(k >= 10 and not narrow_precision and finite_inference),
        'exploratory': bool(k < 10 or narrow_precision or not finite_inference),
        'warnings': warning_messages,
        'note': warning_messages[0],
    })
    if len(warning_messages) > 1:
        warnings.warn(' '.join(warning_messages[1:]), RuntimeWarning, stacklevel=2)
    return result


def forest_plot(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    study_labels: Optional[List[str]] = None,
    output_path: str = 'forest_plot.png',
    rscript_path: Optional[str] = None,
) -> Dict:
    """metafor::forest() 森林图生成。

    Parameters
    ----------
    effect_sizes : np.ndarray
        效应量 (n_studies,)。
    variances : np.ndarray
        效应量方差 (n_studies,)。
    study_labels : list, optional
        研究标签列表。
    output_path : str, optional
        输出图片路径，默认 'forest_plot.png'。
    rscript_path : str, optional
        Rscript 路径。

    Returns
    -------
    Dict
        {'success': bool, 'output_path': str, 'backend': str}
    """
    yi = np.asarray(effect_sizes, dtype=float)
    vi = np.asarray(variances, dtype=float)

    if study_labels is None:
        study_labels = [f'Study {i+1}' for i in range(len(yi))]

    # Windows 路径反斜杠会导致 JSON 词法错误，统一转为正斜杠
    output_abs = os.path.abspath(output_path).replace('\\', '/')

    data_json = _json_for_r({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
        'labels': study_labels,
        'output': output_abs,
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- rma(yi=data$yi, vi=data$vi, method="REML", slab=data$labels)
png(data$output, width=800, height=600, res=120)
forest(rma_fit, xlab="Effect Size", header="Study")
dev.off()
result <- list(
    success = TRUE,
    output_path = data$output,
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na="null", null="null"))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        return r_result['result']

    return {
        'success': False,
        'output_path': None,
        'backend': 'none',
        'error': '森林图生成需要 metafor。' + str(r_result.get('error', '')),
        'note': '请安装 R + metafor: install.packages("metafor")',
    }


def funnel_plot(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    output_path: str = 'funnel_plot.png',
    rscript_path: Optional[str] = None,
) -> Dict:
    """metafor::funnel() 漏斗图生成。

    Parameters
    ----------
    effect_sizes : np.ndarray
        效应量 (n_studies,)。
    variances : np.ndarray
        效应量方差 (n_studies,)。
    output_path : str, optional
        输出图片路径。

    Returns
    -------
    Dict
        {'success': bool, 'output_path': str, 'backend': str}
    """
    yi = np.asarray(effect_sizes, dtype=float)
    vi = np.asarray(variances, dtype=float)

    # Windows 路径反斜杠会导致 JSON 词法错误，统一转为正斜杠
    output_abs = os.path.abspath(output_path).replace('\\', '/')

    data_json = _json_for_r({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
        'output': output_abs,
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- rma(yi=data$yi, vi=data$vi, method="REML")
png(data$output, width=600, height=600, res=120)
funnel(rma_fit, xlab="Effect Size", ylab="Standard Error")
dev.off()
result <- list(
    success = TRUE,
    output_path = data$output,
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na="null", null="null"))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        return r_result['result']

    return {
        'success': False,
        'output_path': None,
        'backend': 'none',
        'error': '漏斗图生成需要 metafor。' + str(r_result.get('error', '')),
    }


def subgroup_analysis(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    groups: np.ndarray,
    rscript_path: Optional[str] = None,
    *,
    method: str = 'REML',
    knha: bool = True,
    study_ids=None,
    sampling_covariance=None,
) -> Dict:
    """亚组随机效应估计，保留实际估计量、检验和后端。

    R 使用指定 method/test；Python 仅 DL，组间比较未实现则明确返回 None。
    Q_between 是历史字段，R+KH 时实际为 F，见 Q_between_test/df。

    Parameters
    ----------
    effect_sizes : np.ndarray
        效应量 (n_studies,)。
    variances : np.ndarray
        效应量方差 (n_studies,)。
    groups : np.ndarray
        分组标签 (n_studies,)。

    Returns
    -------
    Dict
        {'subgroups': dict, 'Q_between': float, 'Q_between_pvalue': float,
         'backend': str}
    """
    yi, vi = _validate_meta_inputs(effect_sizes, variances)
    independence = _check_independent_effects(vi, study_ids, sampling_covariance)
    groups = np.asarray(groups, dtype=object)
    if groups.ndim != 1 or len(groups) != len(yi):
        raise ValueError('groups 必须一维且与效应量长度一致。')
    for group in groups:
        if (group is None or str(group).strip() == ''
                or (isinstance(group, (float, np.floating)) and not np.isfinite(group))):
            raise ValueError('groups 不得缺失或为空。')

    data_json = _json_for_r({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
        'groups': groups.tolist(),
        'method': method,
        'test': 'knha' if knha else 'z',
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- NULL
if (length(unique(data$groups)) > 1 && length(data$yi) > length(unique(data$groups))) {{
    rma_fit <- rma(yi=data$yi, vi=data$vi, mods=~factor(data$groups),
                   method=data$method, test=data$test)
}}
# 各亚组分别估计
subgroups <- list()
for (g in unique(data$groups)) {{
    idx <- data$groups == g
    if (sum(idx) >= 2) {{
        sub_rma <- rma(yi=data$yi[idx], vi=data$vi[idx], method=data$method, test=data$test)
        subgroups[[as.character(g)]] <- list(
            n = sum(idx),
            pooled = as.numeric(sub_rma$b),
            se = as.numeric(sub_rma$se),
            ci_lower = as.numeric(sub_rma$ci.lb),
            ci_upper = as.numeric(sub_rma$ci.ub),
            p_value = as.numeric(sub_rma$pval),
            I2 = as.numeric(sub_rma$I2),
            tau2 = as.numeric(sub_rma$tau2),
            method = sub_rma$method,
            requested_method = data$method,
            test = sub_rma$test,
            knha = identical(sub_rma$test, "knha"),
            df = if (sub_rma$test == "knha") sub_rma$k - sub_rma$p else NULL,
            model = "random_effects",
            backend = "metafor"
        )
    }}
}}
result <- list(
    success = TRUE,
    subgroups = subgroups,
    Q_between = if (is.null(rma_fit)) NULL else as.numeric(rma_fit$QM),
    Q_between_pvalue = if (is.null(rma_fit)) NULL else as.numeric(rma_fit$QMp),
    Q_between_test = if (is.null(rma_fit)) NULL else if (rma_fit$test == "knha") "F" else "chi2",
    Q_between_df = if (is.null(rma_fit)) NULL else as.numeric(rma_fit$QMdf),
    method = data$method,
    requested_method = data$method,
    test = data$test,
    knha = identical(data$test, "knha"),
    model = "subgroup_random_effects",
    r_version = R.version.string,
    metafor_version = as.character(packageVersion("metafor")),
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na="null", null="null"))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        result = r_result['result']
        result.update(independence)
        result['warnings'] = []
        if result['Q_between'] is None:
            result['warnings'].append('亚组水平不足或无残差自由度，不计算组间检验。')
        return result

    # Python 回退：逐组使用同一 DL/HK 实现；组间 Q 仍需要 metafor。
    subgroups = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2:
            continue
        yi_g = yi[mask]
        vi_g = vi[mask]
        group_result = _rma_python_fallback(yi_g, vi_g, method=method, knha=knha)
        subgroups[str(g)] = {
            'n': int(mask.sum()),
            'pooled': group_result.get('pooled_effect'),
            'se': group_result.get('se'),
            'ci_lower': group_result.get('ci_lower'),
            'ci_upper': group_result.get('ci_upper'),
            'p_value': group_result.get('p_value'),
            'I2': group_result.get('I2'),
            'tau2': group_result.get('tau2'),
            'method': group_result['method'],
            'requested_method': group_result['requested_method'],
            'test': group_result['test'],
            'knha': group_result['knha'],
            'df': group_result['df'],
            'model': group_result['model'],
            'backend': group_result['backend'],
            'warnings': group_result['warnings'],
        }

    return {
        'success': True,
        'subgroups': subgroups,
        'Q_between': None,
        'Q_between_pvalue': None,
        'Q_between_test': None,
        'Q_between_df': None,
        'method': 'DL',
        'requested_method': method,
        'test': 'knha' if knha else 'z',
        'knha': bool(knha),
        'model': 'subgroup_random_effects',
        'backend': 'python',
        'warnings': ['Python 仅实现 DL 随机效应亚组估计，未计算组间检验。'],
        'fallback_reason': r_result.get('error'),
        'note': '使用 DL 随机效应回退（非固定效应）。组间检验需 metafor。',
        **independence,
    }


# ---------------------------------------------------------------------------
# 便捷函数：自动检测并选择后端
# ---------------------------------------------------------------------------

_R_ENV_CACHE = None


def get_backend() -> str:
    """检测当前可用的 meta 分析后端。

    Returns
    -------
    str
        'metafor' (R + metafor 可用) 或 'python' (仅 Python 回退)
    """
    global _R_ENV_CACHE
    if _R_ENV_CACHE is None:
        _R_ENV_CACHE = check_r_environment()

    if _R_ENV_CACHE['r_available'] and _R_ENV_CACHE.get('metafor_ready'):
        return 'metafor'
    return 'python'


def backend_info() -> Dict:
    """返回当前后端信息（用于日志和报告）。

    Returns
    -------
    Dict
        {'backend': str, 'r_available': bool, 'metafor_ready': bool,
         'r_version': str or None, 'message': str}
    """
    env = check_r_environment()
    backend = 'metafor' if env.get('metafor_ready') else 'python'

    if backend == 'metafor':
        message = f'使用 metafor (R {env["r_version"]}) 作为后端'
    elif env['r_available']:
        message = f'R 可用但 metafor 未安装。运行 install.packages("metafor") 启用。当前回退到 Python。'
    else:
        message = 'R 未找到。使用 Python 回退实现。安装 R: https://cran.r-project.org/'

    return {
        'backend': backend,
        'r_available': env['r_available'],
        'metafor_ready': env.get('metafor_ready', False),
        'r_version': env.get('r_version'),
        'rscript_path': env.get('rscript_path'),
        'message': message,
    }

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
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


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
    1. PATH 环境变量中的 Rscript
    2. 常见安装路径列表
    3. R_USER 环境变量推断

    Returns
    -------
    str or None
        Rscript 路径，未找到返回 None。
    """
    # 1. 尝试 PATH
    try:
        result = subprocess.run(
            ['Rscript', '--version'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return 'Rscript'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

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
    cat(jsonlite::toJSON(result, auto_unbox=TRUE))
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
  cat(jsonlite::toJSON(list(success=FALSE, error="jsonlite not installed")))
  quit(status=0)
}}
tryCatch({{
  {r_code}
}}, error=function(e) {{
  cat(jsonlite::toJSON(list(success=FALSE, error=as.character(e$message))))
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

def rma_random_effects(
    effect_sizes: np.ndarray,
    variances: np.ndarray,
    method: str = 'REML',
    knha: bool = True,
    rscript_path: Optional[str] = None,
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
        是否使用 Hartung-Knapp 校正（Hartung & Knapp 2001），默认 True。
        强烈推荐，可控制 I 类错误率。
    rscript_path : str, optional
        Rscript 路径，None 时自动查找。

    Returns
    -------
    Dict
        {'pooled_effect': float, 'se': float, 'ci_lower': float, 'ci_upper': float,
         'z_value': float, 'p_value': float, 'I2': float, 'tau2': float,
         'Q': float, 'Q_pvalue': float, 'method': str, 'knha': bool,
         'backend': 'metafor' or 'python'}
    """
    yi = np.asarray(effect_sizes, dtype=float)
    vi = np.asarray(variances, dtype=float)

    # --- 尝试 R + metafor ---
    data_json = json.dumps({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
        'method': method,
        'knha': knha,
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- rma(yi=data$yi, vi=data$vi, method=data$method, knha=data$knha)
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
    method = data$method,
    knha = data$knha,
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        return r_result['result']

    # --- 回退到 Python ---
    return _rma_python_fallback(yi, vi, method, knha)


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

    n = len(yi)
    if n < 2:
        return {'error': '需至少 2 个研究进行随机效应合并'}

    # 固定效应权重
    w_fixed = 1.0 / vi

    # 固定效应估计
    pooled_fixed = np.sum(w_fixed * yi) / np.sum(w_fixed)

    # Cochran's Q
    Q = np.sum(w_fixed * (yi - pooled_fixed) ** 2)
    df = n - 1
    Q_pvalue = 1 - sp_stats.chi2.cdf(Q, df) if df > 0 else np.nan

    # tau² (DerSimonian-Laird 估计)
    c = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df) / c) if c > 0 else 0

    # I²
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0

    # 随机效应权重
    w_random = 1.0 / (vi + tau2)
    pooled_random = np.sum(w_random * yi) / np.sum(w_random)
    se_random = np.sqrt(1.0 / np.sum(w_random))

    # Hartung-Knapp 校正
    if knha and n > 2:
        # HK 标准误放大因子
        hk_factor = np.sqrt(
            np.sum(w_random * (yi - pooled_random) ** 2) / (df * np.sum(w_random))
        )
        se_random = se_random * hk_factor

    # 置信区间和检验
    z_crit = sp_stats.norm.ppf(0.975)
    ci_lower = pooled_random - z_crit * se_random
    ci_upper = pooled_random + z_crit * se_random
    z_value = pooled_random / se_random
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_value)))

    return {
        'success': True,
        'pooled_effect': float(pooled_random),
        'se': float(se_random),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'z_value': float(z_value),
        'p_value': float(p_value),
        'I2': float(I2),
        'tau2': float(tau2),
        'Q': float(Q),
        'Q_pvalue': float(Q_pvalue),
        'method': 'DL',
        'knha': knha,
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

    data_json = json.dumps({
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE))
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
) -> Dict:
    """metafor::regtest() Egger 发表偏倚检验（Egger et al. 1997）。

    检验效应量与其标准误之间是否存在显著关联（小样本研究偏倚）。

    Parameters
    ----------
    effect_sizes : np.ndarray
        效应量 (n_studies,)。
    variances : np.ndarray
        效应量方差 (n_studies,)。

    Returns
    -------
    Dict
        {'z_value': float, 'p_value': float, 'intercept': float,
         'intercept_se': float, 'backend': str}
    """
    yi = np.asarray(effect_sizes, dtype=float)
    vi = np.asarray(variances, dtype=float)
    sei = np.sqrt(vi)

    data_json = json.dumps({
        'yi': yi.tolist(),
        'sei': sei.tolist(),
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- rma(yi=data$yi, sei=data$sei, method="DL")
egger <- regtest(rma_fit, model="lm")
result <- list(
    success = TRUE,
    z_value = as.numeric(egger$zval),
    p_value = as.numeric(egger$pval),
    intercept = as.numeric(egger$est),
    intercept_se = as.numeric(egger$se),
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        return r_result['result']

    # Python 回退：简单线性回归 yi ~ sei
    from scipy import stats as sp_stats

    if len(yi) < 3:
        return {'error': 'Egger 检验需至少 3 个研究'}

    slope, intercept, r, p_val, se = sp_stats.linregress(sei, yi)

    return {
        'success': True,
        'z_value': float(intercept / se),
        'p_value': float(p_val),
        'intercept': float(intercept),
        'intercept_se': float(se),
        'backend': 'python',
        'note': '使用简单线性回归回退（metafor 不可用）。',
    }


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

    data_json = json.dumps({
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE))
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

    data_json = json.dumps({
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
cat(jsonlite::toJSON(result, auto_unbox=TRUE))
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
) -> Dict:
    """亚组分析（分组估计效应量和异质性）。

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
    yi = np.asarray(effect_sizes, dtype=float)
    vi = np.asarray(variances, dtype=float)
    groups = np.asarray(groups)

    data_json = json.dumps({
        'yi': yi.tolist(),
        'vi': vi.tolist(),
        'groups': groups.tolist(),
    })

    r_code = f'''
library(metafor)
data <- jsonlite::fromJSON('{data_json}')
rma_fit <- rma(yi=data$yi, vi=data$vi, mods=~factor(data$groups), method="REML")
# 各亚组分别估计
subgroups <- list()
for (g in unique(data$groups)) {{
    idx <- data$groups == g
    if (sum(idx) >= 2) {{
        sub_rma <- rma(yi=data$yi[idx], vi=data$vi[idx], method="REML")
        subgroups[[as.character(g)]] <- list(
            n = sum(idx),
            pooled = as.numeric(sub_rma$b),
            se = as.numeric(sub_rma$se),
            ci_lower = as.numeric(sub_rma$ci.lb),
            ci_upper = as.numeric(sub_rma$ci.ub),
            p_value = as.numeric(sub_rma$pval),
            I2 = as.numeric(sub_rma$I2),
            tau2 = as.numeric(sub_rma$tau2)
        )
    }}
}}
result <- list(
    success = TRUE,
    subgroups = subgroups,
    Q_between = as.numeric(rma_fit$QM),
    Q_between_pvalue = as.numeric(rma_fit$QMp),
    backend = "metafor"
)
cat(jsonlite::toJSON(result, auto_unbox=TRUE))
'''

    r_result = _run_r_script(r_code, rscript_path)

    if r_result['success'] and r_result['result'].get('success'):
        return r_result['result']

    # Python 回退：逐组计算
    from scipy import stats as sp_stats

    subgroups = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2:
            continue
        yi_g = yi[mask]
        vi_g = vi[mask]
        w = 1.0 / vi_g
        pooled = np.sum(w * yi_g) / np.sum(w)
        se = np.sqrt(1.0 / np.sum(w))
        z = pooled / se
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        subgroups[str(g)] = {
            'n': int(mask.sum()),
            'pooled': float(pooled),
            'se': float(se),
            'ci_lower': float(pooled - 1.96 * se),
            'ci_upper': float(pooled + 1.96 * se),
            'p_value': float(p),
            'I2': None,
            'tau2': None,
        }

    return {
        'success': True,
        'subgroups': subgroups,
        'Q_between': None,
        'Q_between_pvalue': None,
        'backend': 'python',
        'note': '使用固定效应回退（metafor 不可用）。组间 Q 统计量需 metafor。',
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

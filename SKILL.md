---
name: paleoecology-meta-analysis
description: >-
  Paleoecology and paleoclimate meta-analysis for multi-proxy, multi-region
  research: BAM age ensembles, multi-site synthesis (SCC/DCC/CPS/PAI/GAM),
  Bootstrap BCa uncertainty, effect sizes, continuous proxy calibration
  (δDwax/brGDGTs). Supports taxa and continuous proxy dual-channel workflows.
  Invoke when synthesizing paleoecological time-series, validating proxy
  indicators, or attributing environmental changes across sites.
license: MIT license
compatibility: >-
  Requires Python 3.9+. Core: numpy, scipy>=1.7, pandas, statsmodels>=0.14.
  Synthesis: pygam. Bayesian: pymc>=5.0, arviz>=1.0. Geo: pysal, cartopy.
  Paleo: pylipd, pyleoclim. Optional R bridge: R 4.0+ with metafor package
  enables rma(), Egger test, forest/funnel plots via subprocess. Falls back
  to Python DerSimonian-Laird when R unavailable. BAM (pure Python) replaces
  Bacon/Clam age models.
metadata: {"version": "2.1", "skill-author": "paleoecology-research", "based-on": "paleoecology-meta-analysis.html"}
---

# Paleoecology Meta-Analysis

## Overview

面向古生态学与古气候学多代理指标、多研究区域的 meta 分析技能。支持分类群百分比型代理（花粉、硅藻、有孔虫、大植物化石等）和连续值型代理（δDwax、brGDGTs、粒度、有机碳、Mg/Ca 等）的双通道并行工作流。

采用混合方法路线：以古生态学原生综合方法（z-score 标准化、Bootstrap BCa、GAM、蒙特卡洛集合）为主体，以经典效应量（log response ratio、Hedges' d）为条件模块。方法选择以数据结构为判据——配对比较结构激活效应量模块，时序叠加结构仅用原生综合方法。

区域特定功能（如喀斯特保存偏倚、干旱区保存偏倚）以预设插件形式提供，默认 auto 自动选择，不强制任何特定地理环境。

混合 R 桥接架构（v2.1 新增）：当系统检测到 R + metafor 包时，经典 meta 分析功能（`rma()` 随机效应模型、Egger 发表偏倚检验、森林图、漏斗图、meta 回归、亚组分析）自动调用 metafor 后端；R 不可用时自动回退到 Python 的 DerSimonian-Laird 实现。数据交换通过 subprocess + Rscript + JSON，不依赖 rpy2。BAM（纯 Python）始终替代 Bacon/Clam 年龄模型，REVEALS 模型用 z-score 替代（缺口已标注）。

全部方法基于 7 篇核心文献：Izdebski 2022 (Nat Ecol Evol)、Kaufman 2020 (Sci Data)、Marlon 2008 (Nat Geosci)、Power 2008 (Clim Dyn)、Roberts 2018 (Sci Rep)、Hedges 1999 (Ecology)、Lajeunesse 2009 (Am Nat)。R 桥接模块额外基于 Viechtbauer 2010 (J Stat Soft, metafor)、Hartung & Knapp 2001 (Biometrics)、Higgins & Thompson 2002 (JRSS-A, I²)、Egger et al. 1997 (BMJ)。

## When to Use This Skill

- 多站点古生态学/古气候学数据的 meta 分析合成（任意代理类型、任意研究区域）
- 代用指标有效性评估（推断值 vs 观测真值配对比较）
- 事件归因分析（政策、战乱、气候事件、土地利用变化前后准实验比较）
- 年龄-深度建模与年龄不确定性传播
- 连续值代理校准与交叉验证（δDwax/brGDGTs 等）
- Bootstrap BCa 置信区间估计
- 需要经得起同行评审的统计严谨性检查（假设检验、多方法交叉验证、三层不确定性传播）

## Dual-Channel Architecture

本技能支持两套并行通道，根据代理数据类型自动分派：

| 通道 | 适用代理 | 核心模块 | 标准化 | 合成方法 |
|------|---------|---------|--------|---------|
| 分类群通道 | 花粉、硅藻、有孔虫、孢粉、大植物化石 | preprocessing + synthesis | z-score | SCC/DCC/CPS/PAI/GAM |
| 连续值通道 | δDwax、brGDGTs、粒度、TOC、Mg/Ca、Uk37 | continuous_proxy | z-score/minmax/robust | 加权均值 + 蒙特卡洛集合 |

两通道共享 effect_size、validation、scenarios 模块，确保统计严谨性标准一致。

## Hybrid R Bridge (v2.1+)

经典 meta 分析功能通过 `r_bridge.py` 模块实现混合后端架构：

| 后端 | 触发条件 | 功能范围 | 估计方法 |
|------|---------|---------|---------|
| **metafor** (R) | R 4.0+ + metafor 已安装 | rma、meta 回归、Egger 检验、森林图、漏斗图、亚组分析 | REML/ML/DL/EB/HS 五种 |
| **Python** (回退) | R 不可用或 metafor 未安装 | rma（DerSimonian-Laird） | 仅 DL |

**数据交换方式**：Python → JSON → Rscript subprocess → JSON → Python，不依赖 rpy2。

**自动检测**：`check_r_environment()` 在首次调用时检测 R 路径和包状态，结果缓存在后续调用中。`get_backend()` 返回当前激活的后端类型。

**关键参数**：`knha=True`（Hartung-Knapp 校正，默认开启，控制 I 类错误率）；`method='REML'`（默认估计方法，限制最大似然）。

## Installation

```bash
# 核心层
uv pip install "scipy>=1.7" "statsmodels>=0.14" pandas numpy

# 合成层（分类群通道 GAM 合成）
uv pip install "pygam>=0.8"

# 连续值通道校准验证（k-fold 交叉验证）
uv pip install "scikit-learn>=1.0"

# 贝叶斯层
uv pip install "pymc>=5.0" "arviz>=1.0"

# 地理层
uv pip install "pysal" cartopy

# 古气候层
uv pip install pylipd pyleoclim
```

**可选 R 桥接**（启用 metafor 后端）：

```bash
# 1. 安装 R 4.0+: https://cran.r-project.org/
# 2. 在 R 中安装 metafor:
install.packages("metafor")
# 3. 验证:
Rscript -e "library(metafor); cat('metafor OK\n')"
```

安装后 `r_bridge.py` 自动检测并启用 metafor 后端；未安装时回退到 Python DerSimonian-Laird 实现，功能不中断。

**版本兼容性提示**：SciPy 1.7+ 的 `scipy.stats.bootstrap` 默认使用 BCa 方法。PyGAM 的 `LinearGAM` 支持 GCV 自动平滑参数选择。PyMC 5.x 与 ArviZ 1.x 兼容，贝叶斯 GAM 报告时须显式指定 `ci_prob=0.95`。metafor 建议 3.0+ 版本以支持 Hartung-Knapp 校正。

## Scenario Selection Guide

根据数据结构特征选择场景：

**数据存在配对比较结构？**
- 推断值 vs 真值 → **场景一**：代用指标有效性评估（z-score → LOOCV → log response ratio → BCa → RMSEP）
- 事件前 vs 事件后 → **场景三**：事件归因分析（z-score + 多指标 → BCa 差异检验 → 多窗口 → 双指标 → 可选效应量）

**多点时序叠加** → **场景二**：多站点变化综合（BAM 年龄集合 → z-score → 时空对齐 → SCC/GAM/CPS 合成 → 500 成员集合 → LOESS → 多方法交叉验证）

详细决策流程图见 `references/scenarios.md`。

## Module Reference

| 模块 | 文档章节 | Reference 文件 | Script 文件 | 核心函数 |
|------|---------|---------------|------------|---------|
| 数据预处理 | 第三章 | `preprocessing.md` | `preprocessing.py` | bam_age_ensemble, zscore_standardize, resample_to_grid, spatial_clustering(auto), harmonize_names, record_preservation_bias(presets) |
| 连续值代理 | 第三章扩展 | `preprocessing.md` | `continuous_proxy.py` | standardize_continuous_proxy, calibrate_continuous_proxy, composite_continuous_proxy, propagate_continuous_uncertainty, cross_validate_calibration, proxy_comparison |
| 原生综合方法 | 第四章 | `synthesis_methods.md` | `synthesis.py` | scc_composite, gam_composite, monte_carlo_ensemble |
| 效应量模块 | 第五章 | `effect_size.md` | `effect_size.py` | log_response_ratio, effect_size_bca, rmsep |
| 三场景配置 | 第六章 | `scenarios.md` | `scenarios.py` | scenario1_proxy_validation, scenario2_multi_site_synthesis, scenario3_human_attribution, build_indicators |
| 统计严谨性 | 第七章 | `validation.md` | `validation.py` | check_normality_bootstrap, propagate_three_layer_uncertainty |
| R 桥接（可选） | 第五章扩展 | `effect_size.md` | `r_bridge.py` | rma_random_effects, meta_regression, egger_test, forest_plot, funnel_plot, subgroup_analysis |

**模块依赖关系**：`preprocessing` + `continuous_proxy` 被所有场景依赖 → `synthesis` 被场景二、三依赖 → `effect_size` 被场景一、三依赖 → `r_bridge` 可选增强 `effect_size`（R 可用时替换 Python 回退） → `validation` 被所有场景调用 → `scenarios` 依赖前六个模块。

## Statistical Rigor Quick Reference

**假设检查清单**（执行前必查）：
- 正态性：Shapiro-Wilk + Q-Q 图（n>30 时 Bootstrap 渐近稳健）
- 时间独立性：AR1 系数 + Durbin-Watson（违反时用块 Bootstrap）
- 空间独立性：Moran's I（违反时空间聚类后再合成）
- 样本量：n>20（BCa 最低）、n>30（渐近正态）

**三层不确定性传播**：
- 年龄不确定性：从 BAM/Bacon 后验分布整体采样完整年龄-深度曲线（保持地层单调性）
- 校准不确定性：代理-气候校准残差作为正态噪声，标准差来自 RMSEP
- 采样不确定性：Bootstrap 重采样自然传播

**六验证策略**：LOOCV / 多方法一致性(≥2种) / 多时间窗口(100/50/25年) / 双指标系统 / 外部数据对比 / 敏感性分析

详见 `references/validation.md` 和 `references/methodology_gaps.md`。

## Code Conventions

- **函数接口**：输入为 DataFrame/numpy 数组，输出为标准 Dict 结构（值 + 置信区间 + 元数据）
- **参数命名**：`n_members=500`（Kaufman 2020 集合成员数）、`n_boot=10000`（Izdebski 2022 Bootstrap 次数）、`n_splines=20`（GAM 样条节点）、`frac=0.2`（LOESS 平滑参数）
- **docstring**：每个函数首行标注文献来源，如 `"""Kaufman 2020 集合策略：传播三层不确定性"""`
- **代理类型无关**：函数参数命名使用通用术语（`data`/`values`/`proxy_values`），不绑定特定代理类型
- **区域无关**：空间方法默认 `auto`，区域特定功能（如保存偏倚）以预设插件提供
- **依赖限制**：仅允许 `references/python_toolchain.md` 中已验证的包，遇缺口按 `references/methodology_gaps.md` 方案处理

## Bundled Resources

### references/

| 文件 | 内容 |
|------|------|
| `preprocessing.md` | BAM 年龄模型、z-score 标准化、时空对齐、分类群命名统一、多环境保存偏倚预设 |
| `synthesis_methods.md` | SCC/DCC/CPS/PAI/GAM 五方法、Bootstrap BCa、LOESS、蒙特卡洛集合、REVEALS 缺口 |
| `effect_size.md` | log response ratio、Hedges' d、适用边界、联合报告逻辑、系统发育扩展 |
| `scenarios.md` | 三场景完整方法链、决策流程图、双通道架构、代码框架、常见陷阱 |
| `validation.md` | 四假设检验、三层不确定性传播、六验证策略 |
| `python_toolchain.md` | 12 项已验证工具表、版本兼容性、依赖限制规则 |
| `methodology_gaps.md` | 四缺口处理方案、同行评审可辩护性清单 |

### scripts/

| 文件 | 函数数 | 核心功能 |
|------|-------|---------|
| `preprocessing.py` | 7 | 年龄集合生成、z-score 标准化、时空对齐(auto)、分类群命名统一、保存偏倚预设插件 |
| `continuous_proxy.py` | 6 | 连续值代理标准化、校准、多站点合成、三层不确定性传播、交叉验证、双代理对比 |
| `synthesis.py` | 10 | 五方法合成、蒙特卡洛集合、LOESS 趋势、多方法交叉验证 |
| `effect_size.py` | 7 | log response ratio、Hedges' d、BCa 置信区间、LOOCV、RMSEP |
| `scenarios.py` | 7 | 三场景双通道编排、指标构建(用户自定义)、事件前后检验、多窗口验证 |
| `validation.py` | 9 | 假设检验、块 Bootstrap、逐一剔除、敏感性分析、三层传播 |
| `r_bridge.py` | 11 | R+metafor 桥接：rma 随机效应、meta 回归、Egger 检验、森林图、漏斗图、亚组分析（R 不可用时回退到 Python DL 实现） |

### 跨 Skill 协作

- `effect_size.py` 的 `hedges_d()` 可补充调用 `statistical-analysis` skill 的 `pg.compute_efftype(eftype='hedges')`
- `validation.py` 的 `check_normality_bootstrap()` 可调用 `statistical-analysis` skill 的 `assumption_checks.check_normality()`
- `continuous_proxy.py` 的 `cross_validate_calibration()` 可调用 `scikit-learn` skill 的 `KFold` 实现

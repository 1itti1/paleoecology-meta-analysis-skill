---
name: paleoecology-meta-analysis
description: >-
  Paleoecology meta-analysis for pollen-vegetation-climate research: BAM age
  ensembles, multi-site synthesis (SCC/DCC/CPS/PAI/GAM), Bootstrap BCa
  uncertainty, effect sizes (log response ratio, Hedges' d). Invoke when
  synthesizing paleoecological time-series data, validating proxy indicators,
  or attributing human-environment interactions across sites.
license: MIT license
compatibility: >-
  Requires Python 3.9+. Core: numpy, scipy>=1.7, pandas, statsmodels>=0.14.
  Synthesis: pygam. Bayesian: pymc>=5.0, arviz>=1.0. Geo: pysal, cartopy.
  Paleo: pylipd, pyleoclim. R is blocked; BAM (pure Python) replaces Bacon/Clam.
metadata: {"version": "1.0", "skill-author": "paleoecology-research", "based-on": "paleoecology-meta-analysis.html"}
---

# Paleoecology Meta-Analysis

## Overview

面向滇桂黔喀斯特区花粉-植被-气候研究的古生态学 meta 分析技能。采用混合方法路线：以古生态学原生综合方法（z-score 标准化、Bootstrap BCa、GAM、蒙特卡洛集合）为主体，以经典效应量（log response ratio、Hedges' d）为条件模块。方法选择以数据结构为判据——配对比较结构激活效应量模块，时序叠加结构仅用原生综合方法。

Python-only 约束：R 环境被 Smart App Control 阻断，BAM（纯 Python）替代 Bacon/Clam 年龄模型，REVEALS 模型用 z-score 替代（缺口已标注）。

全部方法基于 7 篇核心文献：Izdebski 2022 (Nat Ecol Evol)、Kaufman 2020 (Sci Data)、Marlon 2008 (Nat Geosci)、Power 2008 (Clim Dyn)、Roberts 2018 (Sci Rep)、Hedges 1999 (Ecology)、Lajeunesse 2009 (Am Nat)。

## When to Use This Skill

- 多站点古生态学数据（花粉、炭屑、分子标志物）的 meta 分析合成
- 代用指标有效性评估（推断值 vs 观测真值配对比较）
- 跨区域人地归因分析（事件前后准实验比较）
- 年龄-深度建模与年龄不确定性传播
- Bootstrap BCa 置信区间估计
- 需要经得起同行评审的统计严谨性检查（假设检验、多方法交叉验证、三层不确定性传播）

## Installation

```bash
# 核心层
uv pip install "scipy>=1.7" "statsmodels>=0.14" pandas numpy

# 合成层
uv pip install "pygam>=0.8"

# 贝叶斯层
uv pip install "pymc>=5.0" "arviz>=1.0"

# 地理层
uv pip install "pysal" cartopy

# 古气候层
uv pip install pylipd pyleoclim
```

**版本兼容性提示**：SciPy 1.7+ 的 `scipy.stats.bootstrap` 默认使用 BCa 方法。PyGAM 的 `LinearGAM` 支持 GCV 自动平滑参数选择。PyMC 5.x 与 ArviZ 1.x 兼容，贝叶斯 GAM 报告时须显式指定 `ci_prob=0.95`（ArviZ 1.x 默认 89% 区间）。

## Scenario Selection Guide

根据数据结构特征选择场景：

**数据存在配对比较结构？**
- 推断值 vs 真值 → **场景一**：代用指标有效性评估（z-score → LOOCV → log response ratio → BCa → RMSEP）
- 事件前 vs 事件后 → **场景三**：跨区域人地归因（z-score + 多指标 → BCa 差异检验 → 多窗口 → 双指标 → 可选效应量）

**多点时序叠加** → **场景二**：多站点植被变化综合（BAM 年龄集合 → z-score → 时空对齐 → SCC/GAM/CPS 合成 → 500 成员集合 → LOESS → 多方法交叉验证）

详细决策流程图见 `references/scenarios.md`。

## Module Reference

| 模块 | 文档章节 | Reference 文件 | Script 文件 | 核心函数 |
|------|---------|---------------|------------|---------|
| 数据预处理 | 第三章 | `preprocessing.md` | `preprocessing.py` | bam_age_ensemble, zscore_standardize, resample_to_grid |
| 原生综合方法 | 第四章 | `synthesis_methods.md` | `synthesis.py` | scc_composite, gam_composite, monte_carlo_ensemble |
| 效应量模块 | 第五章 | `effect_size.md` | `effect_size.py` | log_response_ratio, effect_size_bca, rmsep |
| 三场景配置 | 第六章 | `scenarios.md` | `scenarios.py` | scenario1_proxy_validation, scenario2_multi_site_synthesis, scenario3_human_attribution |
| 统计严谨性 | 第七章 | `validation.md` | `validation.py` | check_normality_bootstrap, propagate_three_layer_uncertainty |

**模块依赖关系**：`preprocessing` 被所有场景依赖 → `synthesis` 被场景二、三依赖 → `effect_size` 被场景一、三依赖 → `validation` 被所有场景调用 → `scenarios` 依赖前四个模块。

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
- **依赖限制**：仅允许 `references/python_toolchain.md` 中已验证的包，遇缺口按 `references/methodology_gaps.md` 方案处理

## Bundled Resources

### references/

| 文件 | 内容 |
|------|------|
| `preprocessing.md` | BAM 年龄模型、z-score 标准化、时空对齐、花粉命名统一、喀斯特保存偏倚 |
| `synthesis_methods.md` | SCC/DCC/CPS/PAI/GAM 五方法、Bootstrap BCa、LOESS、蒙特卡洛集合、REVEALS 缺口 |
| `effect_size.md` | log response ratio、Hedges' d、适用边界、联合报告逻辑、系统发育扩展 |
| `scenarios.md` | 三场景完整方法链、决策流程图、代码框架、常见陷阱 |
| `validation.md` | 四假设检验、三层不确定性传播、六验证策略 |
| `python_toolchain.md` | 12 项已验证工具表、版本兼容性、依赖限制规则 |
| `methodology_gaps.md` | 四缺口处理方案、同行评审可辩护性清单 |

### scripts/

| 文件 | 函数数 | 核心功能 |
|------|-------|---------|
| `preprocessing.py` | 7 | 年龄集合生成、z-score 标准化、时空对齐、花粉命名统一 |
| `synthesis.py` | 10 | 五方法合成、蒙特卡洛集合、LOESS 趋势、多方法交叉验证 |
| `effect_size.py` | 7 | log response ratio、Hedges' d、BCa 置信区间、LOOCV、RMSEP |
| `scenarios.py` | 7 | 三场景流水线编排、指标构建、事件前后检验、多窗口验证 |
| `validation.py` | 9 | 假设检验、块 Bootstrap、逐一剔除、敏感性分析、三层传播 |

### 跨 Skill 协作

- `effect_size.py` 的 `hedges_d()` 可补充调用 `statistical-analysis` skill 的 `pg.compute_effsize(eftype='hedges')`
- `validation.py` 的 `check_normality_bootstrap()` 可调用 `statistical-analysis` skill 的 `assumption_checks.check_normality()`

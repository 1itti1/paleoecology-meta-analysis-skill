# Paleoecology Meta-Analysis Skill

面向滇桂黔喀斯特区花粉-植被-气候研究的古生态学 meta 分析工具包。采用混合方法路线：以古生态学原生综合方法（z-score 标准化、Bootstrap BCa、GAM、蒙特卡洛集合）为主体，以经典效应量（log response ratio、Hedges' d）为条件模块。方法选择以数据结构为判据——配对比较结构激活效应量模块，时序叠加结构仅用原生综合方法。

全部方法基于 7 篇核心文献构建，每个函数 docstring 标注文献来源，参数命名与原文献一致（`n_members=500`、`n_boot=10000`、`n_splines=20`、`frac=0.2`），经得起同行评审的统计严谨性推敲。

## 方法学基础

古生态学领域没有统一的 meta 分析标准，存在两条分化路线：

| 维度 | 古生态学原生综合（主流） | 经典效应量 meta 分析（少用） |
|------|------------------------|--------------------------|
| 代表论文 | Izdebski 2022 (Nat Ecol Evol), Marlon 2008 (Nat Geosci), Kaufman 2020 (Sci Data) | Hedges 1999 (Ecology), Lajeunesse 2009 (Am Nat) |
| 效应量 | z-score、标准化百分比、复合指数 | log response ratio, Hedges' d |
| 统计模型 | Bootstrap BCa、贝叶斯 GAM、LOESS、蒙特卡洛集合 | 固定/随机效应模型 |
| 数据结构 | 时间序列叠加 | 配对处理/对照实验 |
| 年龄不确定性 | Bacon/Clam/BAM 集合传播 | 不适用 |

本工具包采用混合路线：原生方法为主体（覆盖时序叠加场景），效应量为条件模块（仅在配对比较结构存在时激活）。这一判据来自数据结构本身而非学科偏好。

### 核心文献

| 文献 | 核心方法 | 在本工具包中的实现 |
|------|---------|-------------------|
| Izdebski 2022 (Nat Ecol Evol) | z-score → Bootstrap 10000 次 BCa → Bayesian GAM | `zscore_standardize()`, `effect_size_bca()` |
| Kaufman 2020 (Sci Data) | 5 方法 (SCC/DCC/GAM/CPS/PAI) + 500 成员集合 | `scc_composite()`, `gam_composite()`, `monte_carlo_ensemble()` |
| Marlon 2008 (Nat Geosci) | 标准化炭屑通量 → LOESS 平滑 | `loess_trend()` |
| Power 2008 (Clim Dyn) | z-score 标准化 + 区域合成 | `zscore_standardize()`, `spatial_clustering()` |
| Roberts 2018 (Sci Rep) | REVEALS 模型花粉→土地覆被 | 缺口已标注（Python 生态空白） |
| Hedges 1999 (Ecology) | log response ratio ln(X_T/X_C) | `log_response_ratio()`, `hedges_d()` |
| Lajeunesse 2009 (Am Nat) | 系统发育 meta 分析扩展 | 参考（未实现） |
| Comboul 2014 | BAM 年龄模型 | `bam_age_ensemble()` |
| Cleveland & Devlin 1988 | LOESS 局部加权回归 | `loess_trend()` |
| Sugita 2007 | REVEALS 模型 | 缺口已标注 |

## 仓库结构

```
paleoecology-meta-analysis-skill/
├── SKILL.md                    # TRAE Skill 入口文件（frontmatter + 8 节正文）
├── README.md                   # 本文件
├── LICENSE                     # MIT
├── CITATION.cff                # 学术引用元数据
├── requirements.txt            # Python 依赖
├── .gitignore
├── references/                 # 7 篇方法学参考文档
│   ├── preprocessing.md        # 第三章：BAM 年龄模型、z-score、时空对齐
│   ├── synthesis_methods.md    # 第四章：SCC/DCC/CPS/PAI/GAM、BCa、LOESS、集合
│   ├── effect_size.md          # 第五章：log response ratio、Hedges' d、适用边界
│   ├── scenarios.md            # 第六章：三场景决策流程图
│   ├── validation.md           # 第七章：假设检验、三层不确定性、验证策略
│   ├── python_toolchain.md     # 第八章：12 项已验证工具及版本兼容性
│   └── methodology_gaps.md     # 第九+十章：四缺口处理、同行评审清单
└── scripts/                    # 5 个 Python 模块，40 个函数
    ├── preprocessing.py        # 7 函数：年龄集合、z-score、重采样、聚类
    ├── synthesis.py            # 10 函数：五方法合成、蒙特卡洛集合、LOESS
    ├── effect_size.py          # 7 函数：log RR、Hedges' d、BCa、RMSEP、LOOCV
    ├── scenarios.py            # 7 函数：三场景编排、前后检验、多窗口验证
    └── validation.py           # 9 函数：假设检验、块 Bootstrap、三层传播
```

## 安装

```bash
git clone https://github.com/paleoecology-research/paleoecology-meta-analysis-skill.git
cd paleoecology-meta-analysis-skill

# 核心依赖
pip install -r requirements.txt
```

**Python 版本**：3.9+（`scipy.stats.bootstrap` 的 BCa 方法需要 SciPy 1.7+）

**可选依赖**：`pygam`（GAM 合成）、`libpysal`+`esda`（空间独立性检验）已在 requirements.txt 中列出。贝叶斯扩展（`pymc`、`arviz`）和地理可视化（`cartopy`）为注释状态，按需启用。

## 三个应用场景

### 场景一：代用指标有效性评估

数据存在配对比较结构——代用指标推断值 vs 已知真值（仪器观测、现代训练集、独立代理交叉验证）。这是唯一天然适配经典效应量的场景。

方法链：z-score 标准化 → LOOCV 评估校准模型 → log response ratio 量化系统偏差 → Bootstrap BCa 估计置信区间 → RMSEP 报告预测精度 → REVEALS 模型对比（缺口已标注）

```python
import sys; sys.path.insert(0, 'scripts')
from effect_size import log_response_ratio, effect_size_bca, rmsep

# 量化代理推断值与观测值的系统偏差
ratios = log_response_ratio(x_proxy, x_observed)
ci = effect_size_bca(x_proxy, x_observed, n_boot=10000)
precision = rmsep(predicted, observed)
```

### 场景二：多站点植被变化综合

多点 sediment core 的时间序列，无处理/对照配对，存在年龄不确定性和自相关。经典效应量不适用。

方法链：BAM 年龄集合 → z-score 标准化 → 时空对齐 → SCC/GAM/CPS 合成 → 500 成员蒙特卡洛集合 → LOESS 趋势可视化 → 多方法交叉验证

```python
from preprocessing import bam_age_ensemble, zscore_standardize, resample_to_grid
from synthesis import gam_composite, monte_carlo_ensemble, uncertainty_band

# 年龄不确定性传播
ages = bam_age_ensemble(depths, ages, age_errors, n_members=500)
values = zscore_standardize(raw_values)
ensembles = monte_carlo_ensemble(values, ages, proxy_errors, n_members=500)
band = uncertainty_band(ensembles, percentiles=(5, 50, 95))
```

### 场景三：跨区域人地关系归因

事件前后准实验比较结构（如政策实施、战乱、气候事件前后的植被/人类活动指标变化）。

方法链：z-score + 多指标构建 → Bootstrap BCa 差异检验 → 多时间窗口稳健性（100/50/25 年）→ 双指标系统交叉验证 → 可选效应量量化幅度

```python
from scenarios import scenario3_human_attribution, multi_window_robustness

result = scenario3_human_attribution(
    before_data, after_data, event_year=-1, indicators=['pollen', 'charcoal']
)
robustness = multi_window_robustness(time_series, event_year=-1, windows=[100, 50, 25])
```

## 模块依赖关系

```
preprocessing ← scenarios
synthesis     ← scenarios
effect_size   ← scenarios
validation    ← scenarios（被所有场景调用）
```

`scenarios.py` 通过 `sys.path.insert` 跨模块导入前四个模块。独立使用任一模块时，只需将该模块所在目录加入 Python 路径。

## 统计严谨性

### 假设检查（执行前必查）

- **正态性**：Shapiro-Wilk + Q-Q 图（n>30 时 Bootstrap 渐近稳健）
- **时间独立性**：AR1 系数 + Durbin-Watson（违反时用块 Bootstrap）
- **空间独立性**：Moran's I（违反时空间聚类后再合成）
- **样本量**：n>20（BCa 最低）、n>30（渐近正态）

### 三层不确定性传播

- **年龄层**：从 BAM/Bacon 后验分布整体采样完整年龄-深度曲线（保持地层单调性）
- **校准层**：代理-气候校准残差作为正态噪声，标准差来自 RMSEP
- **采样层**：Bootstrap 重采样自然传播

### 六验证策略

LOOCV / 多方法一致性（≥2 种）/ 多时间窗口（100/50/25 年）/ 双指标系统 / 外部数据对比 / 敏感性分析

## Python-only 约束说明

R 环境被 Smart App Control 阻断，因此：
- BAM（纯 Python 实现）替代 Bacon/Clam 年龄模型，RMSE 251 年与 Bacon 198 年可比（Kaufman 2020 验证）
- REVEALS 模型用 z-score 替代（Python 生态空白，缺口已在 `references/methodology_gaps.md` 标注）
- 贝叶斯年龄模型通过 PyMC 实现（可选）

## 作为 TRAE Skill 使用

本工具包同时是一个 TRAE Skill。将 `SKILL.md` 所在目录放入 `.trae/skills/paleoecology-meta-analysis/` 后，TRAE 会在以下场景自动加载：

- 多站点古生态学数据的 meta 分析合成
- 代用指标有效性评估
- 跨区域人地归因分析
- 年龄-深度建模与年龄不确定性传播
- 需要同行评审级别的统计严谨性检查

## 引用

如果您在研究中使用此工具包，请引用：

```bibtex
@software{paleoecology_meta_analysis_skill_2026,
  title = {Paleoecology Meta-Analysis Skill},
  author = {paleoecology-research},
  version = {1.0.0},
  date = {2026-07-17},
  license = {MIT},
  url = {https://github.com/paleoecology-research/paleoecology-meta-analysis-skill}
}
```

同时请引用工具包所基于的方法学文献（见上方"核心文献"表格）。

## 许可证

MIT License — 见 [LICENSE](LICENSE)

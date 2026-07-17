# Paleoecology Meta-Analysis Skill

面向古生态学与古气候学多代理指标、多研究区域的 meta 分析工具包。采用混合方法路线：以古生态学原生综合方法（z-score 标准化、Bootstrap BCa、GAM、蒙特卡洛集合）为主体，以经典效应量（log response ratio、Hedges' d）为条件模块。方法选择以数据结构为判据——配对比较结构激活效应量模块，时序叠加结构仅用原生综合方法。

支持**双通道并行架构**：分类群百分比型代理（花粉、硅藻、有孔虫、大植物化石等）和连续值型代理（δDwax、brGDGTs、粒度、有机碳、Mg/Ca 等）各有专用模块，共享统计验证框架。区域特定功能（如喀斯特保存偏倚）以预设插件形式提供，默认 auto 自动选择，不强制任何特定地理环境。

全部方法基于 7 篇核心文献构建，每个函数 docstring 标注文献来源，参数命名与原文献一致（`n_members=500`、`n_boot=10000`、`n_splines=20`、`frac=0.2`），经得起同行评审的统计严谨性推敲。

## 双通道架构

本工具包支持两套并行通道，根据代理数据类型自动分派：

| 通道 | 适用代理 | 核心模块 | 标准化 | 合成方法 |
|------|---------|---------|--------|---------|
| 分类群通道 | 花粉、硅藻、有孔虫、孢粉、大植物化石 | `preprocessing.py` + `synthesis.py` | z-score | SCC/DCC/CPS/PAI/GAM |
| 连续值通道 | δDwax、brGDGTs、粒度、TOC、Mg/Ca、Uk37 | `continuous_proxy.py` | z-score/minmax/robust | 加权均值 + 蒙特卡洛集合 |

两通道共享 `effect_size.py`、`validation.py`、`scenarios.py` 模块，确保统计严谨性标准一致。

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
├── .gitattributes
├── references/                 # 7 篇方法学参考文档
│   ├── preprocessing.md        # 第三章：BAM 年龄模型、z-score、时空对齐、多环境保存偏倚
│   ├── synthesis_methods.md    # 第四章：SCC/DCC/CPS/PAI/GAM、BCa、LOESS、集合
│   ├── effect_size.md          # 第五章：log response ratio、Hedges' d、适用边界
│   ├── scenarios.md            # 第六章：三场景决策流程图、双通道架构
│   ├── validation.md           # 第七章：假设检验、三层不确定性、验证策略
│   ├── python_toolchain.md     # 第八章：12 项已验证工具及版本兼容性
│   └── methodology_gaps.md     # 第九+十章：四缺口处理、同行评审清单
└── scripts/                    # 6 个 Python 模块，46 个函数
    ├── preprocessing.py        # 7 函数：年龄集合、z-score、重采样、auto聚类、命名统一、保存偏倚预设
    ├── continuous_proxy.py     # 6 函数：连续值代理标准化、校准、合成、不确定性传播、交叉验证
    ├── synthesis.py            # 10 函数：五方法合成、蒙特卡洛集合、LOESS
    ├── effect_size.py          # 7 函数：log RR、Hedges' d、BCa、RMSEP、LOOCV
    ├── scenarios.py            # 7 函数：三场景双通道编排、指标构建(自定义)、前后检验
    └── validation.py           # 9 函数：假设检验、块 Bootstrap、三层传播
```

## 安装

```bash
git clone https://github.com/1itti1/paleoecology-meta-analysis-skill.git
cd paleoecology-meta-analysis-skill

# 核心依赖
pip install -r requirements.txt
```

**Python 版本**：3.9+（`scipy.stats.bootstrap` 的 BCa 方法需要 SciPy 1.7+）

**可选依赖**：`pygam`（GAM 合成）、`scikit-learn`（k-fold 交叉验证）、`libpysal`+`esda`（空间独立性检验）已在 requirements.txt 中列出。贝叶斯扩展（`pymc`、`arviz`）和地理可视化（`cartopy`）为注释状态，按需启用。

## 三个应用场景

### 场景一：代用指标有效性评估

数据存在配对比较结构——代用指标推断值 vs 已知真值（仪器观测、现代训练集、独立代理交叉验证）。这是唯一天然适配经典效应量的场景。

**分类群通道**：花粉/硅藻等百分比数据，直接用 log response ratio
**连续值通道**：δDwax/brGDGTs 等，先校准再评估

```python
import sys; sys.path.insert(0, 'scripts')
from effect_size import log_response_ratio, effect_size_bca, rmsep
from continuous_proxy import calibrate_continuous_proxy, cross_validate_calibration

# 连续值通道：校准 + 效应量
calib = calibrate_continuous_proxy(proxy_values, calib_x, calib_y)
cv = cross_validate_calibration(calib_x, calib_y, method='loocv')
ratios = log_response_ratio(proxy_values, observed_values)
ci = effect_size_bca(proxy_values, observed_values, n_boot=10000)
```

### 场景二：多站点变化综合

多点 sediment core 的时间序列，无处理/对照配对，存在年龄不确定性和自相关。经典效应量不适用。

**分类群通道**：BAM → z-score → 时空对齐 → SCC/GAM/CPS 合成 → 500 成员集合
**连续值通道**：标准化 → composite_continuous_proxy 合成 → 不确定性带

```python
from preprocessing import bam_age_ensemble, zscore_standardize
from continuous_proxy import standardize_continuous_proxy, composite_continuous_proxy

# 连续值通道：多站点合成
ages = bam_age_ensemble(depths, ages, age_errors, n_members=500)
std = standardize_continuous_proxy(values, method='zscore')
result = composite_continuous_proxy(
    site_values, site_ages, time_grid,
    age_ensembles=age_ens, proxy_errors=errors, n_members=500
)
# result['uncertainty_band'] 含 5%/50%/95% 分位数
```

### 场景三：事件归因分析

事件前后准实验比较结构（如政策实施、战乱、气候事件、土地利用变化前后的指标变化）。

```python
from scenarios import scenario3_human_attribution, build_indicators, multi_window_robustness

# 用户自定义指标体系（适用于任意研究区域）
indicators = build_indicators(data_df, {
    'crop': ['Oryza', 'Triticum'],           # 用户根据研究区域设计
    'forest': ['Quercus', 'Pinus'],
    'disturbance': ['Artemisia', 'Chenopodiaceae'],
})

result = scenario3_human_attribution(before_data, after_data, event_year=-1)
robustness = multi_window_robustness(time_series, ages, event_year=-1, windows=[100, 50, 25])
```

## 模块依赖关系

```
preprocessing ← scenarios
continuous_proxy ← scenarios
synthesis     ← scenarios
effect_size   ← scenarios
validation    ← scenarios（被所有场景调用）
```

`scenarios.py` 通过 `sys.path.insert` 跨模块导入前五个模块。独立使用任一模块时，只需将该模块所在目录加入 Python 路径。

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

## 多环境保存偏倚预设

`record_preservation_bias()` 内置 5 种环境预设，以插件形式提供：

| 预设键 | 环境 | 敏感分类群 | 耐受分类群 |
|--------|------|-----------|-----------|
| `pollen-karst` | 喀斯特碱性土壤 | 杜鹃花科 | 禾本科、莎草科 |
| `pollen-arid` | 干旱区 | 蕨类孢子 | 藜科、蒿属 |
| `pollen-tropical` | 热带氧化环境 | 桑科、野牡丹科 | 禾本科 |
| `diatom-lake` | 湖泊硅藻 | 脆杆藻 | 直链藻 |
| `foraminifera-marine` | 海洋有孔虫 | 抱球虫 | 球室虫 |

用户也可通过 `sensitive_taxa` 和 `tolerant_taxa` 参数完全自定义。

## Python-only 约束说明

R 环境被 Smart App Control 阻断，因此：
- BAM（纯 Python 实现）替代 Bacon/Clam 年龄模型，RMSE 251 年与 Bacon 198 年可比（Kaufman 2020 验证）
- REVEALS 模型用 z-score 替代（Python 生态空白，缺口已在 `references/methodology_gaps.md` 标注）
- 贝叶斯年龄模型通过 PyMC 实现（可选）

## 作为 TRAE Skill 使用

本工具包同时是一个 TRAE Skill。将 `SKILL.md` 所在目录放入 `.trae/skills/paleoecology-meta-analysis/` 后，TRAE 会在以下场景自动加载：

- 多站点古生态学/古气候学数据的 meta 分析合成
- 代用指标有效性评估（分类群或连续值代理）
- 事件归因分析（任意研究区域）
- 年龄-深度建模与年龄不确定性传播
- 连续值代理校准与交叉验证
- 需要同行评审级别的统计严谨性检查

## 引用

如果您在研究中使用此工具包，请引用：

```bibtex
@software{paleoecology_meta_analysis_skill_2026,
  title = {Paleoecology Meta-Analysis Skill},
  author = {paleoecology-research},
  version = {2.0.0},
  date = {2026-07-17},
  license = {MIT},
  url = {https://github.com/1itti1/paleoecology-meta-analysis-skill}
}
```

同时请引用工具包所基于的方法学文献（见上方"核心文献"表格）。

## 许可证

MIT License — 见 [LICENSE](LICENSE)

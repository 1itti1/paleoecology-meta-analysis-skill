# 古生态学 Meta 分析技能包

[English](README.md) | **[简体中文](README_zh-CN.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)](https://github.com/1itti1/paleoecology-meta-analysis-skill)
[![DOI](https://img.shields.io/badge/DOI-cite-orange.svg)](#引用)

> 以同行评审级的统计严谨性，合成多站点、多代理的古生态学数据。

面向古生态学与古气候学研究的混合方法 meta 分析工具包。以古生态学原生综合方法（z-score 标准化、Bootstrap BCa、GAM、蒙特卡洛集合）为主体路线，以经典效应量（log response ratio、Hedges' d）为条件模块——由数据结构触发，而非学科偏好决定。

**指南针：** 🧭 双通道 · 📊 合成 · 📏 校准 · 🛡️ 验证 · 📝 引用

---

## 目录

- [它能做什么](#-它能做什么)
- [核心特性](#-核心特性)
- [快速开始](#-快速开始)
- [心智模型](#-心智模型)
- [工作流](#-工作流)
- [支持的数据类型](#-支持的数据类型)
- [场景选择](#-场景选择)
- [双通道架构](#-双通道架构)
- [模块索引](#-模块索引)
- [统计严谨性](#-统计严谨性)
- [保存偏倚预设](#-保存偏倚预设)
- [代码示例](#-代码示例)
- [实用说明](#-实用说明)
- [文献基础](#-文献基础)
- [仓库结构](#-仓库结构)
- [常见问题](#-常见问题)
- [路线图](#-路线图)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)
- [引用](#引用)

---

## 🎯 它能做什么

本工具包针对研究者在收集多站点岩芯数据后面临的核心问题：

- 如何将 3 个以上时间分辨率不同的沉积岩芯合并为一条区域趋势曲线？
- 我用 δDwax 重建的古降水可信吗？系统偏差有多大？
- 政策/战乱/气候事件前后的植被变化，能否归因于人类活动？
- 如何在合成的每一步传播年龄不确定性？
- 我的统计选择能否经受同行评审？

特别适用于以下场景：

- 花粉、硅藻、有孔虫等分类群百分比数据
- δDwax、brGDGTs、粒度、TOC、Mg/Ca 等连续值代理
- 拥有 BAM/Bacon 年龄模型的多站点沉积岩芯
- 事件前后对比的准实验设计
- 喀斯特、干旱、热带、湖泊、海洋等多种环境

## ✨ 核心特性

| 特性 | 说明 |
|---|---|
| **双通道架构** | 分类群百分比数据（花粉、硅藻）和连续值代理（δDwax、brGDGTs）各有独立流程，共享统一的统计验证框架 |
| **6 模块 46 个函数** | 预处理、连续值代理、合成、效应量、场景编排、验证——每个函数 docstring 标注文献来源 |
| **三大场景工作流** | 代用指标验证、多站点合成、事件归因——每个场景从原始数据到可发表结论的完整方法链 |
| **三层不确定性传播** | 年龄不确定性（BAM/Bacon 集合）+ 校准不确定性（RMSEP）+ 采样不确定性（Bootstrap），通过 500 成员蒙特卡洛集合联合传播 |
| **五种环境预设** | 喀斯特、干旱、热带、湖泊、海洋的保存偏倚插件——自动选择或手动覆盖 |
| **文献锚定参数** | 每个默认值与发表文献一致：`n_members=500`（Kaufman 2020）、`n_boot=10000`（Izdebski 2022）、`n_splines=20`（GAM）、`frac=0.2`（LOESS） |
| **纯 Python 实现** | 无 R 依赖。BAM（纯 Python）替代 Bacon/Clam 年龄模型，精度可比（RMSE 251 年 vs 198 年） |
| **兼容 TRAE Skill** | 放入 `.trae/skills/` 即可在 TRAE IDE 会话中自动加载 |

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/1itti1/paleoecology-meta-analysis-skill.git
cd paleoecology-meta-analysis-skill
pip install -r requirements.txt
```

或作为 TRAE Skill 使用：

```bash
# Linux / macOS
mkdir -p ~/.trae/skills
cp -R paleoecology-meta-analysis ~/.trae/skills/

# Windows PowerShell
New-Item -ItemType Directory -Force $HOME\.trae\skills | Out-Null
Copy-Item -Recurse -Force .\paleoecology-meta-analysis $HOME\.trae\skills\
```

### 2. 冒烟测试

验证模块导入和核心流程是否正常：

```python
import sys; sys.path.insert(0, 'scripts')

from preprocessing import bam_age_ensemble, zscore_standardize
from continuous_proxy import standardize_continuous_proxy
import numpy as np

# 生成小型测试年龄集合
depths = np.array([0, 10, 20, 30, 40])
ages = np.array([0, 500, 1000, 1500, 2000])
age_errors = np.array([20, 30, 40, 50, 60])
ens = bam_age_ensemble(depths, ages, age_errors, n_members=10)
print(f"年龄集合形状: {ens['age_ensembles'].shape}")

# 标准化连续值代理
values = np.array([15.2, 14.8, 16.1, 13.5, 12.9, 14.0, 15.5])
std = standardize_continuous_proxy(values, method='zscore')
print(f"Z-score: {std['standardized']}")
```

### 3. 运行完整分析

参见[代码示例](#-代码示例)中三个场景的完整流程。

## 🧠 心智模型

```
mindmap
  root((古生态学 Meta 分析))
    🧭 双通道
      分类群通道
        花粉、硅藻、有孔虫
        z-score → SCC/DCC/CPS/PAI/GAM
      连续值通道
        δDwax、brGDGTs、粒度
        标准化 → 校准 → 合成
    📊 合成
      BAM 年龄集合（500 成员）
      蒙特卡洛不确定性传播
      LOESS 趋势可视化
      多方法交叉验证
    📏 校准
      OLS / SMA 回归
      LOOCV / k-fold 验证
      RMSEP 预测误差
      双代理交叉对比
    🛡️ 验证
      正态性 / 独立性检验
      三层不确定性
      六验证策略
      同行评审可辩护性
    📝 引用
      文献锚定参数
      docstring 来源标注
      BibTeX / CITATION.cff
```

## 🧭 工作流

```
flowchart LR
  A[原始岩芯数据] --> B{代理类型?}
  B -->|分类群百分比| C[z-score 标准化]
  B -->|连续值| D[标准化 + 校准]
  C --> E[空间对齐 / 聚类]
  D --> E
  E --> F[合成: SCC/GAM/CPS 或加权均值]
  F --> G[蒙特卡洛集合 500 成员]
  G --> H[不确定性带 5/50/95%]
  H --> I{验证}
  I -->|通过| J[同行评审级输出]
  I -->|未通过| K[检查假设 / 块 Bootstrap]
  K --> F
```

## 📥 支持的数据类型

| 输入 | 通道 | 处理方式 |
|---|---|---|
| 花粉/硅藻/有孔虫百分比 | 分类群 | z-score → 命名统一 → 保存偏倚检查 → 合成 |
| δDwax / brGDGTs / Mg/Ca / Uk37 | 连续值 | 标准化 → 校准为气候变量 → 合成 |
| 粒度 / TOC | 连续值 | 标准化（z-score 或 robust）→ 不校准直接合成 |
| BAM 年龄集合 | 共享 | 500 成员蒙特卡洛传播至所有下游步骤 |
| Bacon/Clam 年龄输出 | 共享 | `consume_bacon_ages()` 导入已有年龄模型 |
| 多站点坐标 | 共享 | `spatial_clustering()` 自动选择网格或距离聚类 |

## 🧩 场景选择

工具包根据**数据结构**选择方法，而非学科偏好：

| 数据结构 | 场景 | 是否用效应量 | 推荐方法链 |
|---|---|---|---|
| 配对比较（代理 vs 真值） | 场景一：代理验证 | ✅ 是 | z-score → LOOCV → log response ratio → BCa → RMSEP |
| 多站点时序叠加 | 场景二：区域合成 | ❌ 否 | BAM → z-score → 空间对齐 → SCC/GAM/CPS → 500 成员集合 → LOESS |
| 事件前后比较 | 场景三：事件归因 | 可选 | z-score + 指标 → BCa 差异检验 → 多窗口稳健性 |

**为什么不总是用效应量？** 经典 meta 分析（Hedges 1999）要求配对处理/对照结构。时序叠加违反独立性假设——在这种情况下使用效应量在统计上是不可辩护的。工具包自动执行这一边界。

## 📊 双通道架构

| 通道 | 适用代理 | 核心模块 | 标准化 | 合成方法 |
|---|---|---|---|---|
| 分类群 | 花粉、硅藻、有孔虫、孢粉、大植物化石 | `preprocessing.py` + `synthesis.py` | z-score | SCC / DCC / CPS / PAI / GAM |
| 连续值 | δDwax、brGDGTs、粒度、TOC、Mg/Ca、Uk37 | `continuous_proxy.py` | z-score / minmax / robust | 加权均值 + 蒙特卡洛 |

两个通道共享 `effect_size.py`、`validation.py` 和 `scenarios.py`——无论代理类型如何，统计严谨性标准一致。

## 📦 模块索引

| 模块 | 函数数 | 核心函数 | 文献基础 |
|---|---|---|---|
| `preprocessing.py` | 7 | `bam_age_ensemble`, `zscore_standardize`, `resample_to_grid`, `spatial_clustering`（auto）, `harmonize_names`, `record_preservation_bias`（预设） | Comboul 2014, Kaufman 2020, Izdebski 2022 |
| `continuous_proxy.py` | 6 | `standardize_continuous_proxy`, `calibrate_continuous_proxy`, `composite_continuous_proxy`, `propagate_continuous_uncertainty`, `cross_validate_calibration`, `proxy_comparison` | Kaufman 2020, Roberts 2018 |
| `synthesis.py` | 10 | `scc_composite`, `dcc_composite`, `cps_composite`, `pai_composite`, `gam_composite`, `monte_carlo_ensemble`, `loess_trend` | Kaufman 2020, Marlon 2008 |
| `effect_size.py` | 7 | `log_response_ratio`, `hedges_d`, `effect_size_bca`, `rmsep`, `loocv` | Hedges 1999, Izdebski 2022 |
| `scenarios.py` | 7 | `scenario1_proxy_validation`, `scenario2_multi_site_synthesis`, `scenario3_human_attribution`, `build_indicators` | 全部 7 篇 |
| `validation.py` | 9 | `check_normality_bootstrap`, `check_temporal_independence`, `check_spatial_independence`, `propagate_three_layer_uncertainty` | Izdebski 2022, Kaufman 2020 |

**模块依赖链：**

```
preprocessing ─┬─→ scenarios
continuous_proxy┘
synthesis ─────┘
effect_size ───┘
validation ────┘  （被所有场景调用）
```

## 🛡️ 统计严谨性

### 假设检查（合成前必查）

| 假设 | 检验方法 | 违反时处理 |
|---|---|---|
| 正态性 | Shapiro-Wilk + Q-Q 图 | 使用 Bootstrap（n>30 时渐近稳健） |
| 时间独立性 | AR1 系数 + Durbin-Watson | 改用块 Bootstrap |
| 空间独立性 | Moran's I | 先空间聚类再合成 |
| 样本量 | n>20（BCa）、n>30（渐近正态） | 标记为初步结果 / 补充数据 |

### 三层不确定性传播

```
年龄层              校准层                采样层
   │                   │                    │
   ▼                   ▼                    ▼
BAM/Bacon          RMSEP 作为 σ          Bootstrap
后验分布            正态噪声标准差          重采样
   │                   │                    │
   └────────┬──────────┘                    │
            ▼                                 │
     500 成员蒙特卡洛集合 ◄──────────────────┘
            │
            ▼
     5%/50%/95% 不确定性带
```

### 六验证策略

1. **LOOCV** — 校准模型的留一交叉验证
2. **多方法一致性** — 运行 ≥2 种合成方法，比较结果
3. **多窗口稳健性** — 用 100/50/25 年窗口测试
4. **双指标系统** — 用独立代理交叉验证
5. **外部数据对比** — 与仪器观测/独立记录比较
6. **敏感性分析** — 变化关键参数，评估结果稳定性

## 🔌 保存偏倚预设

`record_preservation_bias()` 内置 5 种环境预设插件：

| 预设键 | 环境 | 敏感分类群 | 耐受分类群 |
|---|---|---|---|
| `pollen-karst` | 喀斯特碱性土壤 | 杜鹃花科 | 禾本科、莎草科 |
| `pollen-arid` | 干旱区 | 蕨类孢子 | 藜科、蒿属 |
| `pollen-tropical` | 热带氧化环境 | 桑科、野牡丹科 | 禾本科 |
| `diatom-lake` | 湖泊硅藻 | 脆杆藻 | 直链藻 |
| `foraminifera-marine` | 海洋有孔虫 | 抱球虫 | 球室虫 |

通过 `sensitive_taxa` 和 `tolerant_taxa` 参数可完全自定义环境。

## 💻 代码示例

### 场景一：代理验证

验证 δDwax 重建的降水与仪器观测的一致性：

```python
import sys; sys.path.insert(0, 'scripts')
import numpy as np

from effect_size import log_response_ratio, effect_size_bca, rmsep
from continuous_proxy import calibrate_continuous_proxy, cross_validate_calibration

# --- 输入数据 ---
proxy_values = np.array([...])      # δDwax 重建的 δDp
observed_values = np.array([...])   # 仪器观测 δDp
calib_x = np.array([...])           # 校准集: δDwax
calib_y = np.array([...])           # 校准集: δDp

# --- 第一步：校准代理 → 气候变量 ---
calib = calibrate_continuous_proxy(proxy_values, calib_x, calib_y, regression='ols')
print(f"校准 R²={calib['r2']:.3f}, RMSEP={calib['rmsep']:.2f}")

# --- 第二步：交叉验证校准模型 ---
cv = cross_validate_calibration(calib_x, calib_y, method='loocv')
print(f"LOOCV RMSEP={cv['rmsep']:.2f}, R²={cv['r2']:.3f}")

# --- 第三步：量化系统偏差 ---
ratios = log_response_ratio(proxy_values, observed_values)
ci = effect_size_bca(proxy_values, observed_values, n_boot=10000)
precision = rmsep(proxy_values, observed_values)
print(f"平均 log response ratio: {np.mean(ratios):.4f}")
print(f"95% BCa 置信区间: ({ci[0]:.4f}, {ci[1]:.4f})")
print(f"RMSEP: {precision:.2f}")
```

### 场景二：多站点合成

将 3 个湖泊岩芯合并为一条区域植被趋势：

```python
from preprocessing import bam_age_ensemble, zscore_standardize
from continuous_proxy import standardize_continuous_proxy, composite_continuous_proxy
import numpy as np

# --- 输入：3 个分辨率不同的站点 ---
time_grid = np.arange(-2000, 0, 20)  # 2000 BP 至今，20 年间隔

# 站点 A：高分辨率，有年龄集合
site_a_ages = np.array([...])
site_a_values = np.array([...])
site_a_ens = bam_age_ensemble(site_a_depths, site_a_ages, site_a_errors, n_members=500)

# 站点 B 和 C：类似结构...
# （堆叠为数组：site_values (n_sites, n_depths), site_ages (n_sites, n_depths)）

# --- 第一步：标准化各站点 ---
std_a = standardize_continuous_proxy(site_a_values, method='zscore')

# --- 第二步：带不确定性传播的合成 ---
result = composite_continuous_proxy(
    site_values,     # (3, n_depths)
    site_ages,       # (3, n_depths)
    time_grid,       # (n_bins,)
    age_ensembles=site_a_ens['age_ensembles'],  # 共享年龄模型
    proxy_errors=np.array([0.5, 0.7, 0.6]),     # 各站点 RMSEP
    n_members=500,
)

# --- 第三步：提取不确定性带 ---
band = result['uncertainty_band']
print(f"中位数曲线形状: {band['median'].shape}")
print(f"1000 BP 处 90% 置信区间: [{band['lower'][50]:.2f}, {band['upper'][50]:.2f}]")
```

### 场景三：事件归因

检验历史事件（如 1726 年改土归流）后植被是否发生显著变化：

```python
from scenarios import build_indicators, scenario3_human_attribution, multi_window_robustness

# --- 第一步：自定义指标体系 ---
# （完全用户自定义——任意区域、任意分类群）
indicators = build_indicators(pollen_df, {
    'crop': ['Oryza', 'Triticum', 'Hordeum'],          # 作物
    'pasture': ['Poaceae', 'Cyperaceae'],               # 牧场
    'forest': ['Quercus', 'Pinus', 'Castanopsis'],      # 森林
    'disturbance': ['Artemisia', 'Chenopodiaceae'],     # 干扰
}, agg_func='sum')

# --- 第二步：划分事件前后 ---
event_year = 1726
before_data = indicators.loc[indicators.index < event_year].values
after_data = indicators.loc[indicators.index >= event_year].values

# --- 第三步：差异检验 ---
result = scenario3_human_attribution(
    before_data, after_data, event_year=event_year, n_boot=10000
)
for r in result['results']:
    print(f"{r['indicator']}: 差异={r['difference']:.3f}, "
          f"置信区间=({r['ci'][0]:.3f}, {r['ci'][1]:.3f}), p={r['p_value']:.4f}")

# --- 第四步：多窗口稳健性 ---
robustness = multi_window_robustness(
    time_series, ages, event_year=event_year, windows=[100, 50, 25]
)
print(f"跨窗口稳健: {robustness['robust']}")
```

## 🛠️ 实用说明

- **纯 Python**：开发者的机器上 R 被 Smart App Control 阻断。BAM（纯 Python）替代 Bacon/Clam 年龄模型——RMSE 251 年与 Bacon 的 198 年可比（Kaufman 2020 验证）。
- **REVEALS 缺口**：REVEALS 模型（Sugita 2007）无 Python 实现。工具包用 z-score 标准化作为替代，并在 `references/methodology_gaps.md` 中明确标注此缺口。
- **可选依赖**：`pygam`（GAM 合成）、`scikit-learn`（k-fold 交叉验证）、`libpysal`+`esda`（Moran's I）在 requirements.txt 中列出但非必需。缺失时工具包优雅降级。
- **参数默认值**与发表文献一致：`n_members=500`（Kaufman 2020）、`n_boot=10000`（Izdebski 2022）、`n_splines=20`（GAM）、`frac=0.2`（LOESS, Cleveland & Devlin 1988）。
- **从小开始**：先运行冒烟测试，确认模块可导入，再进行单站点分析，最后尝试多站点合成。
- **跨模块导入**：`scenarios.py` 通过 `sys.path.insert` 导入其他模块。独立使用某模块时，需将 `scripts/` 目录加入 Python 路径。

## 📚 文献基础

每个函数的 docstring 标注文献来源，参数命名与原文一致。

| 文献 | 核心方法 | 工具包实现 |
|---|---|---|
| Izdebski 2022 (Nat Ecol Evol) | z-score → Bootstrap 10000 次 BCa → 贝叶斯 GAM | `zscore_standardize()`, `effect_size_bca()` |
| Kaufman 2020 (Sci Data) | 5 方法 (SCC/DCC/GAM/CPS/PAI) + 500 成员集合 | `scc_composite()`, `gam_composite()`, `monte_carlo_ensemble()` |
| Marlon 2008 (Nat Geosci) | 炭屑通量 → LOESS 平滑 | `loess_trend()` |
| Power 2008 (Clim Dyn) | z-score + 区域合成 | `zscore_standardize()`, `spatial_clustering()` |
| Roberts 2018 (Sci Rep) | REVEALS 花粉→土地覆被 | 缺口已标注 |
| Hedges 1999 (Ecology) | log response ratio ln(X_T/X_C) | `log_response_ratio()`, `hedges_d()` |
| Lajeunesse 2009 (Am Nat) | 系统发育 meta 分析 | 仅参考 |
| Comboul 2014 | BAM 年龄模型 | `bam_age_ensemble()` |
| Cleveland & Devlin 1988 | LOESS 局部加权回归 | `loess_trend()` |
| Sugita 2007 | REVEALS 模型 | 缺口已标注 |

## 📂 仓库结构

```
paleoecology-meta-analysis-skill/
├── SKILL.md                    # TRAE Skill 入口（frontmatter + 8 节正文）
├── README.md                   # 英文文档
├── README_zh-CN.md             # 中文文档（本文件）
├── LICENSE                     # MIT
├── CITATION.cff                # 学术引用元数据
├── requirements.txt            # Python 依赖
├── .gitignore
├── .gitattributes
├── references/                 # 7 篇方法学参考文档
│   ├── preprocessing.md        # 第三章：BAM、z-score、空间对齐、保存偏倚预设
│   ├── synthesis_methods.md    # 第四章：SCC/DCC/CPS/PAI/GAM、BCa、LOESS、集合
│   ├── effect_size.md          # 第五章：log response ratio、Hedges' d、适用边界
│   ├── scenarios.md            # 第六章：三场景、决策流程图、双通道架构
│   ├── validation.md           # 第七章：假设检验、三层不确定性、六验证策略
│   ├── python_toolchain.md     # 第八章：12 项已验证工具、版本兼容性
│   └── methodology_gaps.md     # 第九+十章：四缺口、同行评审清单
└── scripts/                    # 6 个 Python 模块，46 个函数
    ├── preprocessing.py        # 7 函数：年龄集合、z-score、自动聚类
    ├── continuous_proxy.py     # 6 函数：标准化、校准、合成、交叉验证
    ├── synthesis.py            # 10 函数：五方法合成、蒙特卡洛、LOESS
    ├── effect_size.py          # 7 函数：log RR、Hedges' d、BCa、RMSEP、LOOCV
    ├── scenarios.py            # 7 函数：三场景编排、指标构建
    └── validation.py           # 9 函数：假设检验、块 Bootstrap、三层传播
```

## ❓ 常见问题

<details>
<summary><b>能否用于海洋沉积岩芯？</b></summary>

可以。工具包与区域无关。对于海洋岩芯，连续值代理（如 Mg/Ca 或 Uk37）使用 `proxy_type='continuous'`；如有孔虫组合数据，选择 `foraminifera-marine` 保存偏倚预设。BAM 年龄模型适用于任何沉积物类型。
</details>

<details>
<summary><b>为什么没有 R 实现？</b></summary>

开发者的机器上 R 被 Smart App Control 阻断。所有方法均用纯 Python 实现。BAM（Comboul 2014）替代 Bacon/Clam 年龄模型，精度可比。REVEALS 模型缺口在 `references/methodology_gaps.md` 中有说明。
</details>

<details>
<summary><b>如何导入 Bacon 年龄模型输出？</b></summary>

使用 `preprocessing.py` 中的 `consume_bacon_ages(bacon_output_path)`。它加载 Bacon 年龄集合输出（`.txt` 或 `.csv`，每列为一个集合成员），返回标准化的字典，包含 `age_ensembles`、`depths` 和 `n_members`。
</details>

<details>
<summary><b>数据违反独立性假设怎么办？</b></summary>

`validation.py` 模块提供 `check_temporal_independence()`（AR1 + Durbin-Watson）和 `check_spatial_independence()`（Moran's I）。若违反，改用 `block_bootstrap()` 替代标准 Bootstrap，并用 `spatial_clustering()` 先对邻近站点聚类再合成。
</details>

<details>
<summary><b>能否添加自定义保存偏倚预设？</b></summary>

可以。将 `sensitive_taxa` 和 `tolerant_taxa` 作为列表传入 `record_preservation_bias()`。也可以扩展 `preprocessing.py` 中的 `PRESERVATION_BIAS_PRESETS` 字典来永久注册新预设。
</details>

<details>
<summary><b>两个通道有什么区别？</b></summary>

**分类群通道**处理百分比数据（花粉、硅藻、有孔虫）——使用 z-score 标准化和五种合成方法（SCC/DCC/CPS/PAI/GAM）。**连续值通道**处理单值代理（δDwax、brGDGTs）——增加校准回归、交叉验证，使用加权均值合成和蒙特卡洛集合。两者共享相同的效应量、验证和场景模块。
</details>

## 🗺️ 路线图

- [x] v1.0 — 初始版本：5 模块、40 函数、聚焦喀斯特花粉
- [x] v2.0 — 普适化：双通道架构、6 模块、46 函数、多代理多区域
- [ ] v2.1 — 添加示例数据集和 Jupyter notebook 教程
- [ ] v2.2 — PyMC 贝叶斯 GAM 实现（替代 PyGAM，提供完整后验不确定性）
- [ ] v3.0 — REVEALS 模型 Python 移植（欢迎协作）
- [ ] v3.1 — 交互式 Web 可视化面板

## 🤝 贡献指南

欢迎贡献！以下领域特别需要帮助：

- **REVEALS 模型 Python 实现** — 最大的方法学缺口（Sugita 2007）
- **贝叶斯 GAM** — 基于 PyMC 替代 PyGAM，提供完整后验不确定性
- **测试数据集** — 匿名化的示例沉积岩芯数据，用于教程
- **更多语言翻译** — README 的日语、德语、法语、西班牙语版本
- **Bug 报告和功能请求** — 在 GitHub 上提 Issue

贡献流程：

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/new-function`）
3. 确保所有函数有文献引用的 docstring
4. 通过 `python -m py_compile scripts/*.py` 验证
5. 提交 Pull Request

## 📄 许可证

基于 [MIT License](LICENSE) 发布。

## 引用

如果您在研究中使用本工具包，请引用：

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

同时请引用您分析所依赖的方法学文献（见[文献基础](#-文献基础)）。

## 🚢 发布信息

- **仓库描述**：`面向古生态学与古气候学的混合方法 meta 分析工具包：双通道（分类群+连续值代理）合成，支持 BAM 年龄集合、Bootstrap BCa 不确定性、同行评审级统计严谨性。`
- **标语**：`以同行评审级的统计严谨性，合成多站点、多代理的古生态学数据。`
- **当前版本**：`v2.0.0 — 双通道架构，多区域普适化`

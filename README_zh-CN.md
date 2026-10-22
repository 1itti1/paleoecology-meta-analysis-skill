# 通用古生态学 Meta 分析 Skill

面向多站点古生态学与古气候代理记录的可复现、带验证的分析工作流。
设计目标是区域、沉积档案和代理类型无关：花粉、硅藻、有孔虫、大植物
化石、炭屑、生物标志物、同位素和其他地层代理均可使用统一数据契约，
同时保留各自的生态学和年代学假设。

> 开发版本：2.3.0。本项目是可复现的分析起点，不替代档案专属年代模型、
> 代理生态学判断或专家审查。

本次工作版本新增了借鉴 EasyMeta 的分析契约与数值修正，尚未发布新版本。
EasyMeta 无需另外安装；本 skill 已包含所需的精简路由和来源记录实现。

## 推荐用法：先声明分析目标，再运行

推荐入口为 `run_analysis()`：先声明要估计什么、采用什么尺度、独立重复单位
是什么以及依赖来自哪里。在仓库根目录安装 `requirements.txt` 后，可以直接
运行下面的完整合成数据示例：

```python
import json
import numpy as np
from scripts.analysis_contracts import run_analysis

with open("assets/analysis_plan.json", encoding="utf-8") as stream:
    plan = json.load(stream)

records = [
    dict(site_id="core_A", proxy_id="wetness_index", target="hydroclimate",
         ages=[0., 100., 200.], values=[1., 2., 4.], age_unit="cal_year_BP",
         direction="positive", source="synthetic example"),
    dict(site_id="core_B", proxy_id="wetness_index", target="hydroclimate",
         ages=[0., 100., 200.], values=[2., 3., 5.], age_unit="cal_year_BP",
         direction="positive", source="synthetic example"),
]
result = run_analysis(plan, records, time_grid=np.array([50., 150.]),
                      n_members=200, random_state=42)
print(result["analysis_route"]["route"])
print(result["composite"])
print(result["provenance"]["input_sha256"])
```

若要做校准、配对预测验证、事件窗口比较或研究级效应量 Meta，修改计划中的
`design` 及匹配的 `data_level`。依赖结构超出已实现方法或明确不宜合并时，
返回 `status="not_fitted"` 和原因。普通 Meta 运行前，还会实际检查重复研究、
独立簇和效应尺度，防止仅靠计划中的声明放行。
字段、研究级效应记录和 SHA-256 来源记录详见
[分析契约说明](references/analysis_contracts.md)。

## 已修正计算的迁移说明

- `effect_size_bca(..., effect_type="lnrr")` 现在明确计算
  `log(mean(T)/mean(C))`；若目标为 `mean(log(T/C))`，请显式选择
  `paired_log_ratio`，并声明配对或独立设计。
- 代理预测验证先应用校准，默认报告观测单位下的偏差和 RMSEP；不再仅因数值
  全为正值就自动采用比值效应。
- 年代组默认独立；共享后验成员需要显式声明并对齐 `chronology_group`。
- 标准化误差、Python KH 推断和 Egger 系数提取均加入数值对照测试。受旧实现
  影响的结果需要重新计算。
- 事件比较使用块重抽区间；多窗口不再按 P 值投票。BP 年代请指定
  `time_direction="BP"`。
- 单序列连续代理的不确定性默认固定样品位置，IID 配对样品重抽需要显式启用。

## 2.3 版主要改进

- 增加代理级数据契约和异质代理综合层：分别标准化、统一方向、时间对齐，
  先在站点内整合代理，再在站点间进行区域合成，避免同一档案被重复计权。
- 增加按站点聚类的 bootstrap、可选的标准化测量误差相关性、代理一致性诊断
  和逐代理剔除敏感性分析。

## 2.2 版主要改进

- 删除“纯 Python BAM 可替代所有年龄模型”的表述，改为明确标注的年代点
  扰动敏感性分析；旧函数名仅保留兼容性。
- 增加共同输入校验、安全的非外推插值、逐时间点权重归一化、局部随机数
  状态和不等长站点支持。
- 修复 LOWESS 导入、bootstrap 回调、Hedges' g 重采样、连续代理合成、
  GAM 输出契约和时间序列诊断。
- 增加 `agents/openai.yaml`、包化导入和 `tests/` 回归测试。
- 将事件前后结果重新定位为关联性检验；除非具备明确反事实设计，不宣称
  因果归因。

## 适用任务

- 多站点代理数据审计、清理和命名统一；
- 使用已有年代模型集合进行时间对齐；
- 分类群或连续代理标准化；
- 不规则、不等长时间序列的区域合成；
- 代理预测值与观测值的配对验证；
- 年代、校准和采样不确定性传播；
- 时间/空间依赖检验和逐站点剔除敏感性分析；
- 只有在具有可辩护共同目标时，才联合花粉、炭屑、同位素、生物标志物等代理；
- 检查代理冲突并进行逐代理剔除敏感性分析；
- 在不夸大因果性的前提下比较气候和人类活动指标。

## 核心约束

1. 原始计数、分母、单位和来源信息必须与百分比、z-score 和模型输出分开保存。
2. 优先使用外部拟合的年代模型后验或年龄集合。
3. `age_ensemble_from_errors` 仅表示年代点扰动敏感性分析；旧的
   `bam_age_ensemble` 函数已弃用，不能解释为真正 BAM 模型。
4. 每个站点、每个年龄成员独立插值；观测范围外返回 `NaN`，不静默外推。
5. 缺失站点导致的权重必须在每个时间点重新归一化。
6. 经典效应量只用于具有合理配对或独立研究结构的数据，不能直接用于
   相关时间序列叠加。
7. GAM/LOESS 默认是描述性平滑，除非另外提供完整推断模型。
8. 事件前后差异默认是关联，不是因果效应。
9. 同一档案的多个代理必须作为一个站点 bootstrap 簇；代理数量增加不等于
   区域独立站点数量增加。

## 快速开始

请在仓库根目录运行。最常见的流程是：

1. 安装核心依赖；
2. 为每个 `site_id` × `proxy_id` 准备一条记录；
3. 声明所有代理共同表示的 `target` 以及每个代理的 `direction`；
4. 运行合成，并检查覆盖度、不确定性、代理一致性和逐代理剔除敏感性。

首次使用时先安装核心依赖：

```bash
python -m pip install -r requirements.txt
```

下面第一个代码块是单通道 smoke test，异质多代理示例位于其后。

```python
import numpy as np

from scripts.preprocessing import resample_to_grid
from scripts.synthesis import scc_composite
from scripts.effect_size import effect_size_bca

ages = np.array([0., 100., 200.])
values = np.array([1., 2., 3.])
grid = np.array([-50., 50., 150., 250.])

aligned = resample_to_grid(ages, values, grid)
print(aligned["resampled"])  # 两端为 NaN，不使用边界值外推

sites = np.array([[1., 2., np.nan], [3., 4., 5.]])
print(scc_composite(sites)["composite"])

effect = effect_size_bca(
    np.array([1.2, 1.4, 1.3, 1.5] * 8),
    np.array([1.0, 1.1, 1.2, 1.0] * 8),
    n_boot=2000,
    random_state=42,
)
print(effect["effect_size"], effect["ci_lower"], effect["ci_upper"])
```

已有年龄集合可直接读取：

```python
from scripts.preprocessing import consume_bacon_ages
chronology = consume_bacon_ages("age_ensemble.csv")
# chronology["age_ensembles"] 形状为 (members, samples)
```

没有后验集合时，只能把年代误差作为敏感性分析：

```python
from scripts.preprocessing import age_ensemble_from_errors
perturbed = age_ensemble_from_errors(
    depths, dated_ages, dated_age_errors,
    n_members=500, random_state=42,
)
```

对于具有共同目标的异质代理，可使用明确的代理级工作流：

可将下面代码保存为 `example.py`，然后运行 `python example.py`。示例使用内存
中的合成数组；实际分析时请替换为经过审计的 `ages` 和 `values`。
两个通道是假设已具有共同目标的湿润/干旱指数，不表示花粉、炭屑等原始代理
可以直接互换生态含义。

```python
import numpy as np

from scripts.multi_proxy import multi_proxy_synthesis, leave_one_proxy_out

records = [
    {
        "site_id": "core_A",
        "proxy_id": "wetness_index",
        "proxy_type": "continuous",
        "target": "hydroclimate",
        "ages": np.array([0., 100., 200.]),
        "values": np.array([12., 18., 24.]),
        "direction": "positive",
        "measurement_error": 0.5,
    },
    {
        "site_id": "core_A",
        "proxy_id": "dryness_index",
        "proxy_type": "continuous",
        "target": "hydroclimate",
        "ages": np.array([0., 100., 200.]),
        "values": np.array([8., 5., 2.]),
        "direction": "negative",
        "measurement_error": 0.2,
    },
    {
        "site_id": "core_B",
        "proxy_id": "wetness_index",
        "proxy_type": "continuous",
        "target": "hydroclimate",
        "ages": np.array([0., 100., 200.]),
        "values": np.array([11., 17., 23.]),
        "direction": "positive",
        "measurement_error": 0.5,
    },
    {
        "site_id": "core_B",
        "proxy_id": "dryness_index",
        "proxy_type": "continuous",
        "target": "hydroclimate",
        "ages": np.array([0., 100., 200.]),
        "values": np.array([7., 4., 2.]),
        "direction": "negative",
        "measurement_error": 0.2,
    },
]

result = multi_proxy_synthesis(
    records,
    np.array([0., 50., 100., 150., 200.]),
    n_members=500,
    random_state=42,
)
loo = leave_one_proxy_out(
    records, np.array([0., 50., 100., 150., 200.]),
    n_members=500, random_state=42,
)

print(result["composite"])
print(result["uncertainty_band"])
print(result["effective_site_count"])
print(result["proxy_concordance"])
print(loo["by_excluded_proxy"])
```

最重要的输入字段如下：

| 字段 | 含义 |
|---|---|
| `site_id` | 用于站点级重抽样的独立档案/剖面 |
| `proxy_id` | 代理通道，例如 `pollen`、`charcoal` 或 `d13C` |
| `target` | 共同合成目标；所有记录必须一致 |
| `direction` | `positive` 或 `negative`，使转换后的数值方向一致 |
| `ages`、`values` | 等长的原始数组；不同记录可以不等长 |
| `age_ensembles` | 可选年代后验，形状为 `(members, samples)` |
| `measurement_error` | 可选测量误差，可为原始单位下的标量或逐样品数组 |

主要输出包括 `composite`、`uncertainty_band`、`effective_site_count`、
`effective_proxy_count`、`proxy_concordance` 和 `by_excluded_proxy`。同一站点
包含多个代理时仍只算一个区域重复单元，因为 bootstrap 单位是
`site_cluster`。

这是透明的共同目标证据合成，不是潜变量模型、REVEALS 或代理专属前向模型。
不同目标必须分开分析，再在共同时间框架下比较。

## 数据契约

每条观测尽量保留 `site_id`、`sample_id`、`age`、可选的 `depth`、原始代理
值/计数、单位、来源和测量方法。分类群数据必须保留花粉和/或代理总数等
百分比计算分母。多站点不等长数据应传入逐站点数组，不要用零填充。逐站点
年龄集合形状为 members × samples；只有在所有站点确实共用同一年代时，才可
使用共享年龄集合。

多代理分析应为每个 `site_id` × `proxy_id` 建立一条记录，并保留
`proxy_type`、`target`、`direction`、`unit`、`measurement_error`、`weight`、
`site_weight` 和 `chronology_group`。不能合并不同目标，也不能静默推断代理
方向。

## 模块

| 模块 | 功能 |
|---|---|
| `scripts/preprocessing.py` | 年龄集合读取、扰动敏感性、标准化、安全插值、命名和保存偏倚记录 |
| `scripts/analysis_contracts.py` | 可执行路由、原始/分析阶段效应校验、独立簇检查和分析分发 |
| `scripts/provenance.py` | 输入、参数、代码、来源文件和输出指纹；软件版本 |
| `scripts/synthesis.py` | SCC、DCC、CPS、PAI、描述性 GAM/LOESS 和蒙特卡洛集合 |
| `scripts/continuous_proxy.py` | 校准、不等长站点合成、不确定性、时间序列验证和双代理比较 |
| `scripts/multi_proxy.py` | 代理级校验、目标/方向统一、按站点聚类合成、相关误差传播、一致性和逐代理剔除 |
| `scripts/effect_size.py` | 配对 log-ratio、Hedges' g、BCa/percentile 区间、RMSEP、LOOCV |
| `scripts/scenarios.py` | 配对验证、多站点合成和事件窗口关联分析 |
| `scripts/validation.py` | 正态性诊断、时间/空间依赖、移动块 bootstrap 和敏感性分析 |
| `scripts/r_bridge.py` | 可选 `metafor` 后端；Python fallback 是明确标注的功能子集 |

## 安装与测试

```bash
python -m pip install -r requirements.txt
# 可选方法：
python -m pip install -r requirements-optional.txt
python -m unittest discover -s tests -v
```

核心操作依赖 NumPy、pandas、SciPy 和 statsmodels。PyGAM、scikit-learn、
PySAL/ESDA 以及 R/metafor 仅在调用相应方法时检测。缺少可选后端时必须在
结果中明确说明，不能默认为数值等价实现。

## 方法边界

本 skill 不实现 REVEALS，也不把 z-score 标准化宣称为 REVEALS 的生物学等价
替代。最重要的区分是：档案专属年代模型、由本 skill 消费的年龄集合，以及
没有后验时使用的简单敏感性扰动，三者不能混称。

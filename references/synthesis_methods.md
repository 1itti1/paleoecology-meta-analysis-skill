# 古生态学原生综合方法

> 提取自源文档第四章。本章阐述构成方法主体路线的六类原生综合技术，构成所有三个应用场景（见 [scenarios.md](scenarios.md)）的基础分析层。

## 1. 合成与叠加方法

Kaufman 2020 系统比较了四种合成方法，每种方法的适用性取决于数据是否经过校准：

| 方法 | 全称 | 适用数据 | 核心操作 |
|------|------|----------|----------|
| SCC | Standard Calibrated Composite | 已校准代理（温度、降水等物理量） | 校准值直接加权合成，面积加权求均值 |
| DCC | Dynamic Calibrated Composite | 已校准代理，校准函数随时间变化 | 动态校准后合成，适用于校准关系非平稳场景 |
| CPS | Composite Plus Scale | 未校准代理（原始 z-score） | 合成后以研究时段均值和标准差标定 |
| PAI | Pairwise Comparison | 未校准代理，关注变化方向而非幅度 | 两两比较各点位变化方向，合成方向一致性指数 |

**选择准则**：已校准数据优先使用 SCC 或 GAM（第 2 节），可提供物理量级的合成结果；未校准数据使用 CPS，提供相对变化的合成；当仅关注变化方向且数据质量参差不齐时，PAI 提供最稳健的方向性判断。

## 2. GAM 方法

广义可加模型（GAM）在 Kaufman 2020 框架中作为五方法之一，同时在 Izdebski 2022 研究中用于空间插值。优势在于可灵活拟合非线性趋势，且通过惩罚样条控制过拟合。

- **PyGAM**：提供惩罚 B 样条 GAM 实现，接口简洁，适用于一维时间序列合成：
  ```python
  from pygam import LinearGAM, s
  gam = LinearGAM(s(0, n_splines=20)).fit(time, values)
  ```
- **PyMC**：可实现贝叶斯 GAM，提供后验分布而非点估计，适用于需要完整不确定性传播的场景。Izdebski 2022 使用 R 的 AverageR 包实现贝叶斯薄板回归样条进行空间插值，Python 中可用 PyGAM 张量积样条或 PyMC 空间高斯过程近似替代。

**样条数量准则**：Kaufman 2020 使用约 20 个样条节点拟合 6000 年时序。一般准则是**每 200-300 年一个样条节点**，但须通过交叉验证或 AIC/GCV 准则确认。

## 3. Bootstrap 与置信区间

Bootstrap BCa（bias-corrected and accelerated）方法是本方案不确定性量化的统计核心。Izdebski 2022 使用 10000 次 Bootstrap 重采样 + BCa 置信区间量化花粉指标变化显著性。BCa 方法由 Hall 1988 提出，相比百分位法修正了偏度和加速因子，在分布不对称时提供更准确的区间估计。

Python 实现使用 `scipy.stats.bootstrap`（SciPy 1.7+，Python 3.9+ 默认 BCa 方法）：

```python
from scipy.stats import bootstrap
import numpy as np

result = bootstrap(
    (diff_values,),           # 待重采样的数据元组
    statistic=np.mean,        # 统计量函数
    n_resamples=10000,        # 重采样次数
    method='BCa',             # 偏差校正加速法
    confidence_level=0.95,    # 置信水平
    random_seed=42            # 可复现性
)
ci_lower, ci_upper = result.confidence_interval
```

**限制**：BCa 方法要求样本量 n > 20 以保证重采样分布的稳定性。当 n 较小时，置信区间可能过宽，须在结果中报告样本量。

## 4. LOESS 合成曲线

LOESS（局部加权回归平滑）由 Cleveland 和 Devlin 1988 提出，Marlon 2008 将其用于全球炭屑通量的趋势可视化。Python 实现使用 `statsmodels.nonparametric.lowess`，关键参数是平滑参数 `frac`（窗口宽度占比），通常设为 0.1-0.3。

**定位：LOESS 在本方案中是可视化工具而非推断工具**——用于展示合成曲线的趋势形态，不用于显著性判断。统计推断由 Bootstrap BCa（第 3 节）和 GAM（第 2 节）承担。

## 5. 集合不确定性传播

古生态学 meta 分析须传播**三层不确定性**：年龄不确定性（年龄-深度模型的后验分布）、校准不确定性（代理-气候校准函数的残差）、采样不确定性（有限样本量导致的随机误差）。Kaufman 2020 通过 500 成员蒙特卡洛集合实现三层不确定性的联合传播。

```python
import numpy as np

def monte_carlo_ensemble(records, age_ensembles, proxy_errors, n_members=500):
    """Kaufman 2020 集合策略：传播年龄+校准+采样三层不确定性"""
    ensembles = []
    for i in range(n_members):
        ages = age_ensembles[i % len(age_ensembles)]   # 整体采样保持地层单调性
        values = records + np.random.normal(0, proxy_errors)
        composite = composite_function(ages, values)
        ensembles.append(composite)
    return np.array(ensembles)  # shape: (n_members, n_timebins)

def uncertainty_band(ensembles, percentiles=(5, 50, 95)):
    return np.percentile(ensembles, percentiles, axis=0)
```

**常见陷阱**：年龄扰动须保持地层顺序——不可从年龄后验分布中独立采样每个深度的年龄。正确做法是从 BAM/Bacon 输出的完整年龄集合成员中整体采样，每个成员本身就是一条单调的年龄-深度曲线。

## 6. REVEALS 模型与 Python 生态空白

REVEALS 模型由 Sugita 2007 提出，用于将花粉百分比转换为区域土地覆被比例，考虑不同分类群的花粉生产力差异（RPP）和扩散特性。Roberts 2018 验证了 REVEALS 重建的土地覆被与遥感观测的相关性达 r=0.69。

**缺口**：REVEALS 模型目前在 Python 生态中没有成熟实现，现有工具集中在 R 语言（如 LRA R 包）。三种应对方案（rpy2 桥接 / 手动实现 / z-score 替代）详见 [methodology_gaps.md](methodology_gaps.md) 第 4 节。

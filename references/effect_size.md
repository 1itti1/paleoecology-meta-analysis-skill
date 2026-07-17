# 经典效应量模块

> 提取自源文档第五章。本章描述作为**条件模块**的经典效应量方法。该模块仅在数据存在配对比较结构时激活，不作为通用分析方法使用。

## 1. Log Response Ratio

Hedges 等人 1999 年提出的 log response ratio 定义为：

```
L = ln(X_T / X_C)
```

其中 X_T 为处理组均值，X_C 为对照组均值。取对数的原因是比值型数据的分布在原始尺度上偏态，对数变换后近似正态，便于构造置信区间。log response ratio 的方差近似为：

```
Var(L) = (s_T² / (n_T × X_T²)) + (s_C² / (n_C × X_C²))
```

其中 s_T、s_C 为两组标准差，n_T、n_C 为样本量。

**场景中 X_T、X_C 的含义随应用场景而变**：
- 场景一：X_T 为代用指标推断值、X_C 为观测真值
- 场景三：X_T 为事件后均值、X_C 为事件前均值

## 2. Hedges' d

当数据包含零值或负值时，log response ratio 无定义（因分母为零或比值为负）。此时改用 Hedges' d，即标准化均值差异：

```
d = (X_T - X_C) / s_pooled × J(n_T + n_C - 3)
```

其中 s_pooled 为合并标准差，J() 为小样本校正因子。

**零值处理**：Hedges' d 不受零值限制，但 loses the ratio interpretation（丢失比值含义），在跨研究比较时不如 log response ratio 直观。场景一中 X_proxy 或 X_observed 含零值时，须加常数偏移或改用 Hedges' d。

## 3. 适用边界

效应量模块的激活判据是**数据是否存在配对比较结构**。以下两种结构满足条件：

| 配对结构 | 处理组 (X_T) | 对照组 (X_C) | 对应场景 |
|----------|--------------|--------------|----------|
| 推断值 vs 真值 | 代用指标推断的温度/降水 | 仪器观测/现代训练集 | 场景一 |
| 事件前 vs 事件后 | 事件后子时段均值 | 事件前子时段均值 | 场景三 |

**多点时间序列叠加（场景二）不激活效应量**：不存在处理/对照配对，强行使用效应量会违反独立性假设——时间序列中的相邻观测点存在自相关，不满足效应量方法要求的独立同分布前提。详见 [methodology_gaps.md](methodology_gaps.md) 第 1 节。

## 4. 与原生方法的集成

效应量模块与原生综合方法（见 [synthesis_methods.md](synthesis_methods.md)）的关系是**联合报告**而非替代：原生方法（Bootstrap BCa）判定变化是否显著，效应量量化变化的幅度。两者构成完整的统计推断链条——显著性回答"有没有变"，效应量回答"变了多少"。

**集成流程**：
1. 首先使用 Bootstrap BCa 检验配对差异是否显著（置信区间不跨零）
2. 对显著的变化计算 log response ratio 或 Hedges' d 量化效应大小
3. 最终报告中同时给出显著性判断和效应量估计

```python
def effect_size_bca(x_proxy, x_observed, n_boot=10000):
    """Izdebski 2022 BCa 方法估计效应量置信区间"""
    ratios = np.log(x_proxy / x_observed)   # log response ratio
    result = bootstrap(
        (ratios,),
        statistic=np.mean,
        n_resamples=n_boot,
        method='BCa',
        confidence_level=0.95
    )
    return {
        'effect_size': np.mean(ratios),
        'ci_lower': result.confidence_interval.low,
        'ci_upper': result.confidence_interval.high
    }
```

## 5. 系统发育扩展（参考）

Lajeunesse 等人 2009 年将经典效应量扩展到系统发育框架，通过广义最小二乘（GLS）和布朗运动/OU 过程建模物种间的系统发育相关性。

- **适用条件**：跨分类群的 meta 分析，当研究涉及近缘分类群（如同一科内的多个属）且须控制系统发育非独立性时考虑使用。
- **定位**：本方案将其作为**参考方法**，不在三个核心场景中强制使用。

## 6. 与其他模块的衔接

- BCa 显著性判断 ← [synthesis_methods.md](synthesis_methods.md) 第 3 节 Bootstrap BCa 接口
- 配对结构判定 → [scenarios.md](scenarios.md) 场景一/场景三激活、场景二不激活
- 准实验限制标注 → [methodology_gaps.md](methodology_gaps.md) 第 3 节
- 假设检验前置 → [validation.md](validation.md) 第 1 节（独立性假设检查）

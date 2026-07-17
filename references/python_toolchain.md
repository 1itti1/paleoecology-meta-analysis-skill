# Python 工具链

> 提取自源文档第八章。以下工具链已逐一验证可用性和版本兼容性。所有推荐均限于 Python 生态，不依赖 R 环境。

## 1. 已验证工具表

| 功能 | Python 工具 | 验证状态 | 文献来源 |
|------|-------------|----------|----------|
| z-score 标准化 | scikit-learn / pandas | 原生，无需额外安装 | Izdebski 2022, Kaufman 2020 |
| Bootstrap BCa | scipy.stats.bootstrap (method='BCa') | 已验证，SciPy 1.7+ / Python 3.9+ | Izdebski 2022 |
| GAM 惩罚 B 样条 | PyGAM (LinearGAM, s) | 已验证，Kaufman 2020 使用同框架 | Kaufman 2020 |
| 贝叶斯 GAM / 空间模型 | PyMC | 已验证可用，替代 R AverageR | Izdebski 2022 替代方案 |
| LOESS 平滑 | statsmodels.nonparametric.lowess | 原生 | Marlon 2008 |
| 蒙特卡洛集合 | numpy.random | 原生 | Kaufman 2020 |
| LiPD 数据格式 | PyLiPD + LiPD utilities | 已验证 | Kaufman 2020 |
| 古气候时序分析 | Pyleoclim | 已验证 (Khider 2022) | Pyleoclim 文档 |
| 年龄-深度建模 | BAM (Python) / rpy2→Bacon | 部分缺口——BAM 纯 Python；Bacon 须 rpy2 | Kaufman 2020, Izdebski 2022 |
| REVEALS 花粉转换 | 无成熟 Python 实现 | 缺口——详见 [methodology_gaps.md](methodology_gaps.md) 第 4 节 | Roberts 2018 |
| 空间可视化 | Cartopy + Matplotlib | 原生，Pyleoclim 集成 | 标准实践 |
| 空间自相关检验 | PySAL (esda.Moran) | 已验证 | 标准实践 |

## 2. 版本兼容性提示

- **SciPy 1.7+**：`scipy.stats.bootstrap` 默认使用 BCa 方法（须 Python 3.9+）。
- **PyGAM**：`LinearGAM` 支持 B 样条和平滑参数自动选择（GCV）。
- **PyMC 5.x 与 ArviZ 1.x 兼容**：贝叶斯 GAM 须注意 ArviZ 1.x 默认 89% 区间，报告时须显式指定 `ci_prob=0.95`。

## 3. 依赖限制规则

第八章工具表中的包列表为**允许依赖的全集**。具体规范：

- **不得引入未经验证的第三方库**。遇到功能缺口时，按 [methodology_gaps.md](methodology_gaps.md) 的方案处理而非引入新依赖。
- **REVEALS 缺口**：无成熟 Python 实现，提供三种应对方案（rpy2 桥接 / 手动实现 / z-score 替代），当前推荐 z-score 替代。
- **Bacon/Clam 缺口**：须 rpy2 桥接 R 环境，推荐 BAM（纯 Python）替代。当 R 环境被阻断时，BAM 是首选。

**已标注缺口汇总**（源文档第 10.3 节 Python 工具存在性审计确认）：
- REVEALS：无成熟 Python 实现，提供三种应对方案
- Bacon/Clam：须 rpy2 桥接，推荐 BAM 替代

## 4. 代码可复用性要求

源文档第 11.4 节规定，文档中所有代码框架须遵循以下规范，确保后续可直接迁移到 skill 脚本：

- **函数接口清晰**：输入为 DataFrame 或 numpy 数组，输出为标准结构（值 + 置信区间 + 元数据字典）。避免函数内部硬编码文件路径或数据格式假设。
- **参数命名与文献一致**：`n_members` 对应 Kaufman 2020 的集合成员数；`n_boot` 对应 Izdebski 2022 的 Bootstrap 重采样次数；`n_splines` 对应 GAM 样条节点数。参数默认值取文献推荐值（如 `n_members=500`、`n_boot=10000`）。
- **每个函数附 docstring 标注文献来源**：如 `"""Kaufman 2020 集合策略：传播年龄+校准+采样三层不确定性"""`，使后续使用者可直接追溯到原始文献。
- **依赖仅限已验证的 Python 包**：本工具表为允许依赖的全集，不得引入未经验证的第三方库。

**标准函数接口示例**：

```python
def effect_size_bca(x_proxy, x_observed, n_boot=10000):
    """Izdebski 2022 BCa 方法估计效应量置信区间
    返回标准结构：值 + 置信区间 + 元数据字典
    """
    ratios = np.log(x_proxy / x_observed)
    result = bootstrap((ratios,), statistic=np.mean,
                       n_resamples=n_boot, method='BCa',
                       confidence_level=0.95)
    return {
        'effect_size': np.mean(ratios),
        'ci_lower': result.confidence_interval.low,
        'ci_upper': result.confidence_interval.high
    }
```

## 5. 从文档到实践的路径

源文档第 11.2 节将文档章节直接映射为 skill 模块结构：

| 文档章节 | Skill 模块 | 模块大小评估 | 依赖关系 |
|----------|-----------|--------------|----------|
| 第三章 数据预处理 | `preprocessing` | 适中——含 BAM 年龄模型、z-score、时空对齐 | 被所有场景模块依赖 |
| 第四章 原生综合方法 | `synthesis` | 较大——含 SCC/DCC/CPS/PAI/GAM + 集合传播 | 被场景二、三依赖 |
| 第五章 效应量模块 | `effect_size` | 适中——含 log response ratio + BCa + Hedges' d | 可复用 statistical-analysis skill；被场景一、三依赖 |
| 第六章 三场景配置 | `scenarios` | 较大——三场景独立配置，可拆分为三个子模块 | 依赖 preprocessing + synthesis + effect_size |
| 第七章 统计严谨性 | `validation` | 适中——含假设检验 + 不确定性传播 + 验证策略 | 被所有场景模块调用 |

若 `synthesis` 模块过大，可考虑独立为 `paleoecology-synthesis` skill；效应量模块可复用现有 `statistical-analysis` skill 的 Hedges' g 计算能力，避免重复实现。

## 6. 与其他模块的衔接

- z-score / Bootstrap BCa / GAM / 蒙特卡洛用法 ← [synthesis_methods.md](synthesis_methods.md)
- 工具在场景中的配置 ← [scenarios.md](scenarios.md)
- REVEALS / Bacon 缺口处理 ← [methodology_gaps.md](methodology_gaps.md)
- 假设检验工具（Moran's I、Shapiro-Wilk）← [validation.md](validation.md) 第 1 节

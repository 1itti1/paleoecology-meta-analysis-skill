"""
古生态学与古气候学数据预处理模块（多代理、多区域通用）。
涵盖外部年龄集合消费、年代点扰动敏感性、z-score 标准化、时空对齐、
分类群命名统一和保存偏倚记录。

支持两类代理数据：
- 分类群百分比型（花粉、孢粉、硅藻、有孔虫等）：harmonize_names / record_preservation_bias
- 连续值型（δDwax、brGDGTs、粒度、有机碳等）：见 continuous_proxy.py

文献来源：
- 外部年龄模型：本模块只消费其年龄集合，不替代档案专属建模
- Kaufman 2020 [2]: 时空对齐、年龄集合消费
- Izdebski 2022 [1]: z-score 标准化、地理区域聚类
- Power 2008 [4]: z-score 标准化
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

try:
    from ._utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        validate_age_ensembles,
        validate_same_length,
    )
except ImportError:  # pragma: no cover - supports direct script imports
    from _utils import (
        as_float_array,
        get_rng,
        interpolate_no_extrapolation,
        validate_age_ensembles,
        validate_same_length,
    )


# ---------------------------------------------------------------------------
# 保存偏倚预设库：proxy_type × environment → (sensitive_taxa, tolerant_taxa, note)
# 用户可直接调用预设，也可通过 custom 自定义。
# ---------------------------------------------------------------------------
PRESERVATION_BIAS_PRESETS: Dict[str, Dict] = {
    'pollen-karst': {
        'sensitive': ['Ericaceae', 'Vaccinium', 'Rhododendron'],
        'tolerant': ['Poaceae', 'Cyperaceae', 'Pinus'],
        'note': '喀斯特碱性土壤保存偏倚：喜酸分类群低估，耐碱分类群相对高估。',
    },
    'pollen-arid': {
        'sensitive': ['Pteridophyta', 'Lycopodium', 'Selaginella'],
        'tolerant': ['Chenopodiaceae', 'Artemisia', 'Ephedra'],
        'note': '干旱区保存偏倚：薄壁孢子降解，旱生耐受分类群相对富集。',
    },
    'pollen-tropical': {
        'sensitive': ['Moraceae', 'Urticaceae', 'Melastomataceae'],
        'tolerant': ['Poaceae', 'Cyathea', 'Pteridophyta'],
        'note': '热带氧化环境保存偏倚：薄壁分类群优先降解。',
    },
    'diatom-lake': {
        'sensitive': ['Fragilaria', 'Eunotia'],
        'tolerant': ['Aulacoseira', 'Stephanodiscus'],
        'note': '湖泊硅藻溶解偏倚：薄壳属种优先溶解，厚壳属种相对富集。',
    },
    'foraminifera-marine': {
        'sensitive': ['Globigerinoides', 'Globigerina'],
        'tolerant': ['Globorotalia', 'Neogloboquadrina'],
        'note': '海洋有孔虫溶解偏倚：薄壳表层种易溶解，厚壳深水种相对保存。',
    },
}


def age_ensemble_from_errors(
    depths: np.ndarray,
    ages: np.ndarray,
    age_errors: np.ndarray,
    n_members: int = 500,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """Generate a transparent horizon-perturbation age ensemble.

    This is *not* an age-depth model. It perturbs dated horizons by their
    reported errors and enforces monotonicity as a sensitivity analysis. Use a
    posterior from an archive-specific chronology model whenever possible.
    """
    depth_arr = as_float_array(depths, 'depths', ndim=1, allow_nan=False, min_size=2)
    age_arr = as_float_array(ages, 'ages', ndim=1, allow_nan=False, min_size=2)
    error_arr = as_float_array(age_errors, 'age_errors', ndim=1, allow_nan=False, min_size=2)
    validate_same_length(('depths', depth_arr), ('ages', age_arr), ('age_errors', error_arr))
    if n_members < 2:
        raise ValueError('n_members 至少需要 2。')
    if np.any(error_arr < 0):
        raise ValueError('age_errors 必须非负。')

    order = np.argsort(depth_arr, kind='mergesort')
    depth_arr, age_arr, error_arr = depth_arr[order], age_arr[order], error_arr[order]
    direction = 1.0 if age_arr[-1] >= age_arr[0] else -1.0
    rng = get_rng(random_state)
    ensembles = np.empty((n_members, len(age_arr)), dtype=float)

    for i in range(n_members):
        perturbed = age_arr + rng.normal(0.0, error_arr)
        directed = direction * perturbed
        directed = np.maximum.accumulate(directed)
        ensembles[i] = direction * directed

    return {
        'age_ensembles': ensembles,
        'n_members': n_members,
        'method': 'age_perturbation',
        'depths': depth_arr,
        'random_state': random_state if isinstance(random_state, int) else None,
        'warning': '这是年龄点扰动敏感性分析，不是 BAM/Bacon/Clam 年龄模型。',
    }


def bam_age_ensemble(
    depths: np.ndarray,
    ages: np.ndarray,
    age_errors: np.ndarray,
    n_members: int = 500,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict:
    """Backward-compatible alias for :func:`age_ensemble_from_errors`.

    The old name is retained so existing notebooks do not break, but it does
    not claim to fit the Banded Age Model (BAM).
    """
    warnings.warn(
        'bam_age_ensemble 已改为 age_ensemble_from_errors；输出不是 BAM 年龄模型。',
        DeprecationWarning,
        stacklevel=2,
    )
    return age_ensemble_from_errors(
        depths, ages, age_errors, n_members=n_members, random_state=random_state
    )


def consume_bacon_ages(bacon_output_path: str) -> Dict:
    """Kaufman 2020：消费已有 Bacon 年表输出。

    加载 Bacon 年龄集合输出，每个成员为单调年龄-深度曲线。
    适用于任何已用 Bacon/Clam 建模的沉积岩芯。

    Parameters
    ----------
    bacon_output_path : str
        Bacon 输出文件路径（.txt 或 .csv，每列为一个集合成员）。

    Returns
    -------
    Dict
        {'age_ensembles': np.ndarray (n_members, n_depths),
         'depths': np.ndarray, 'n_members': int, 'method': 'Bacon'}
    """
    try:
        df = pd.read_csv(bacon_output_path, sep=r'\s+')
    except Exception:
        df = pd.read_csv(bacon_output_path)

    if 'depth' in df.columns:
        depths = df['depth'].values
        ensemble_cols = [c for c in df.columns if c != 'depth']
        age_ensembles = df[ensemble_cols].values.T
    else:
        age_ensembles = df.values.T
        depths = np.arange(df.shape[0])

    age_ensembles = validate_age_ensembles(age_ensembles, age_ensembles.shape[1])
    return {
        'age_ensembles': age_ensembles,
        'depths': depths,
        'n_members': age_ensembles.shape[0],
        'method': 'Bacon',
    }


def zscore_standardize(
    data: Union[np.ndarray, pd.DataFrame],
    baseline_period: Optional[Tuple[float, float]] = None,
    group_col: Optional[str] = None,
    ages: Optional[np.ndarray] = None,
    value_columns: Optional[List[str]] = None,
    ddof: int = 0,
) -> Dict:
    """Izdebski 2022, Power 2008：z-score 标准化。

    对每个岩芯的每个变量（分类群百分比或连续值代理），以研究时段均值和标准差计算 z-score：
    z = (x - μ) / σ

    适用于任意代理类型（花粉、硅藻、有孔虫、δDwax、brGDGTs 等）。

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        代理值数组或 DataFrame。若 DataFrame 且 group_col 指定，按分组分别标准化。
    baseline_period : tuple, optional
        基准时段 (start, end)。指定时必须同时提供 ages。
    group_col : str, optional
        DataFrame 中的分组列名（如分类群名、岩芯 ID 或代理类型）。
    ages : np.ndarray, optional
        与数组行对应的年龄，用于 baseline_period 的筛选。
    value_columns : list, optional
        DataFrame 中需要标准化的代理列。默认使用数值列，但排除 group_col。
    ddof : int, optional
        标准差自由度，默认 0；跨模块应保持一致。

    Returns
    -------
    Dict
        {'z_scores': same type as input, 'means': dict, 'stds': dict, 'baseline': str}
    """
    if baseline_period is not None:
        if ages is None:
            raise ValueError('使用 baseline_period 时必须提供 ages。')
        age_arr = as_float_array(ages, 'ages', ndim=1, allow_nan=False)
        if len(age_arr) != len(data):
            raise ValueError('ages 必须与 data 的行数一致。')
        baseline_mask = (age_arr >= baseline_period[0]) & (age_arr <= baseline_period[1])
        if not np.any(baseline_mask):
            raise ValueError('baseline_period 内没有有效样本。')
    else:
        baseline_mask = np.ones(len(data), dtype=bool)

    if isinstance(data, pd.DataFrame):
        columns = list(value_columns) if value_columns is not None else data.select_dtypes(include=[np.number]).columns.tolist()
        if group_col in columns:
            columns.remove(group_col)
        missing = [c for c in columns if c not in data.columns]
        if missing:
            raise ValueError(f'value_columns 不存在：{missing}')
        if not columns:
            raise ValueError('没有可标准化的数值代理列。')

        groups = data[group_col].dropna().unique() if group_col is not None else [None]
        z_data = data.copy()
        means, stds = {}, {}
        for g in groups:
            group_mask = (data[group_col] == g) if group_col is not None else np.ones(len(data), dtype=bool)
            fit_mask = group_mask & baseline_mask
            values = data.loc[fit_mask, columns]
            mu = values.mean(axis=0)
            sigma = values.std(axis=0, ddof=ddof).replace(0, np.nan)
            z_data.loc[group_mask, columns] = (data.loc[group_mask, columns] - mu) / sigma
            key = g if g is not None else 'all'
            means[key] = mu.to_dict()
            stds[key] = sigma.to_dict()
        baseline = 'full_period' if baseline_period is None else str(baseline_period)
        return {
            'z_scores': z_data,
            'means': means,
            'stds': stds,
            'baseline': baseline,
            'value_columns': columns,
            'ddof': ddof,
        }

    arr = as_float_array(data, 'data', ndim=None)
    if baseline_period is not None and arr.ndim != 1:
        raise ValueError('ndarray 的 baseline_period 目前要求 data 为一维序列。')
    fit_values = arr[baseline_mask] if arr.ndim == 1 else arr
    mu = np.nanmean(fit_values, axis=0)
    sigma = np.nanstd(fit_values, axis=0, ddof=ddof)
    sigma = np.where(sigma == 0, np.nan, sigma)
    z = (arr - mu) / sigma
    baseline = 'full_period' if baseline_period is None else str(baseline_period)
    return {'z_scores': z, 'mean': mu, 'std': sigma, 'baseline': baseline, 'ddof': ddof}


def resample_to_grid(
    ages: np.ndarray,
    values: np.ndarray,
    time_grid: np.ndarray,
    age_ensembles: Optional[np.ndarray] = None,
) -> Dict:
    """Kaufman 2020：重采样到统一时间网格。

    将不同分辨率岩芯重采样到统一时间网格。
    若提供 age_ensembles，对每个成员独立插值以传播年龄不确定性。
    适用于任意代理类型。

    Parameters
    ----------
    ages : np.ndarray
        年龄数组 (n_depths,)。
    values : np.ndarray
        代理值数组 (n_depths,) 或 (n_depths, n_vars)。
    time_grid : np.ndarray
        统一时间网格 (n_bins,)。
    age_ensembles : np.ndarray, optional
        年龄集合 (n_members, n_depths)，用于传播年龄不确定性。

    Returns
    -------
    Dict
        {'resampled': np.ndarray (n_bins,) or (n_members, n_bins, n_vars),
         'time_grid': np.ndarray, 'n_bins': int}
    """
    age_arr = as_float_array(ages, 'ages', ndim=1, allow_nan=False, min_size=2)
    value_arr = as_float_array(values, 'values', ndim=None)
    if value_arr.ndim == 1:
        value_arr = value_arr[:, np.newaxis]
    if value_arr.ndim != 2:
        raise ValueError('values 必须是一维或二维数组。')
    validate_same_length(('ages', age_arr), ('values', value_arr))
    grid = as_float_array(time_grid, 'time_grid', ndim=1, allow_nan=False)

    if age_ensembles is not None:
        age_ensembles = validate_age_ensembles(age_ensembles, len(age_arr))
        n_members = age_ensembles.shape[0]
        n_vars = value_arr.shape[1]
        resampled = np.full((n_members, len(grid), n_vars), np.nan)
        for m in range(n_members):
            for t in range(n_vars):
                resampled[m, :, t] = interpolate_no_extrapolation(
                    age_ensembles[m], value_arr[:, t], grid,
                    name=f'age_member_{m}',
                )
        if n_vars == 1:
            resampled = resampled[:, :, 0]
        return {
            'resampled': resampled,
            'time_grid': grid,
            'n_bins': len(grid),
            'extrapolation': 'nan',
        }

    n_vars = value_arr.shape[1]
    resampled = np.full((len(grid), n_vars), np.nan)
    for t in range(n_vars):
        resampled[:, t] = interpolate_no_extrapolation(
            age_arr, value_arr[:, t], grid, name=f'variable_{t}'
        )
    if n_vars == 1:
        resampled = resampled[:, 0]
    return {
        'resampled': resampled,
        'time_grid': grid,
        'n_bins': len(grid),
        'extrapolation': 'nan',
    }


def spatial_clustering(
    site_coords: pd.DataFrame,
    method: str = 'auto',
    radius_km: float = 200,
    grid_resolution: float = 2.0,
) -> Dict:
    """Izdebski 2022, Kaufman 2020：空间聚类或网格化。

    将多点研究站点按地理关系分组，用于区域合成前的空间对齐。
    方法选择以站点空间分布特征为判据，不预设特定地理环境。

    Parameters
    ----------
    site_coords : pd.DataFrame
        含 'lat', 'lon' 列的点位坐标，索引为点位名。
    method : str, optional
        聚类方法：
        - 'auto' (默认)：根据站点密度自动选择。站点密集且分布不均时用 'cluster'，
          站点稀疏或均匀分布时用 'grid'。
        - 'cluster'：基于距离的地理聚类（Izdebski 2022）。
        - 'grid'：经纬度网格化；不等同于等面积投影。
        - 'karst'：'cluster' 的别名，保留向后兼容。
    radius_km : float, optional
        聚类半径 (km)，默认 200 (Izdebski 2022)。
    grid_resolution : float, optional
        网格分辨率 (度)，默认 2.0。

    Returns
    -------
    Dict
        {'clusters': dict (cluster_id -> [site_names]),
         'method': str, 'radius_km': float or None, 'grid_resolution': float or None}
    """
    from math import radians, sin, cos, asin, sqrt

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371 * 2 * asin(sqrt(a))

    if not isinstance(site_coords, pd.DataFrame) or not {'lat', 'lon'}.issubset(site_coords.columns):
        raise ValueError("site_coords 必须是包含 'lat' 和 'lon' 列的 DataFrame。")
    if site_coords.empty:
        raise ValueError('site_coords 不能为空。')
    if not np.isfinite(site_coords[['lat', 'lon']].to_numpy(dtype=float)).all():
        raise ValueError('site_coords 不能包含 NaN 或无穷值。')
    if radius_km <= 0 or grid_resolution <= 0:
        raise ValueError('radius_km 和 grid_resolution 必须为正值。')

    # 'karst' 向后兼容
    if method == 'karst':
        method = 'cluster'

    # auto 模式：根据站点密度选择
    if method == 'auto':
        n_sites = len(site_coords)
        lat_range = site_coords['lat'].max() - site_coords['lat'].min()
        lon_range = site_coords['lon'].max() - site_coords['lon'].min()
        area = max(lat_range * lon_range, 1.0)
        density = n_sites / area
        # 站点密集（>0.5 个/平方度）且数量≥6 时用聚类，否则用网格
        method = 'cluster' if (density > 0.5 and n_sites >= 6) else 'grid'

    sites = list(site_coords.index)
    n = len(sites)

    if method == 'grid':
        # 简单经纬度网格化；正式面积比较应使用等面积投影。
        clusters = {}
        for site in sites:
            lat = site_coords.loc[site, 'lat']
            lon = site_coords.loc[site, 'lon']
            grid_lat = np.floor(lat / grid_resolution) * grid_resolution
            grid_lon = np.floor(lon / grid_resolution) * grid_resolution
            key = f'{grid_lat}_{grid_lon}'
            clusters.setdefault(key, []).append(site)
        # 将 key 转为序号
        clusters = {i + 1: v for i, v in enumerate(clusters.values())}
        return {
            'clusters': clusters, 'method': 'grid',
            'radius_km': None, 'grid_resolution': grid_resolution,
        }

    # cluster 模式：基于距离的地理聚类
    # Connected components make clustering transitive and independent of row order.
    adjacency = {site: set() for site in sites}
    for i, site_i in enumerate(sites):
        for j in range(i + 1, n):
            site_j = sites[j]
            dist = haversine(
                site_coords.loc[site_i, 'lat'], site_coords.loc[site_i, 'lon'],
                site_coords.loc[site_j, 'lat'], site_coords.loc[site_j, 'lon'],
            )
            if dist <= radius_km:
                adjacency[site_i].add(site_j)
                adjacency[site_j].add(site_i)

    clusters = {}
    unvisited = set(sites)
    cluster_id = 0
    while unvisited:
        cluster_id += 1
        seed = next(iter(unvisited))
        stack = [seed]
        component = []
        unvisited.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    stack.append(neighbor)
        clusters[cluster_id] = component

    return {
        'clusters': clusters, 'method': 'cluster',
        'radius_km': radius_km, 'grid_resolution': None,
    }


def harmonize_names(
    data_df: pd.DataFrame,
    mapping_dict: Dict[str, str],
) -> Dict:
    """分类群命名统一（通用）。

    不同实验室/数据库可能使用不同分类群命名体系，须建立映射表统一到一致分类框架。
    适用于花粉、孢粉、硅藻、有孔虫、大植物化石等任意分类群百分比数据。

    Parameters
    ----------
    data_df : pd.DataFrame
        分类群百分比数据，列为分类群名。
    mapping_dict : dict
        命名映射字典 {旧名: 新名}。

    Returns
    -------
    Dict
        {'harmonized_df': pd.DataFrame, 'n_renamed': int, 'unmapped': list}
    """
    harmonized = data_df.rename(columns=mapping_dict)
    # 合并同名列
    if harmonized.columns.duplicated().any():
        dup_cols = harmonized.columns[harmonized.columns.duplicated()].unique()
        for col in dup_cols:
            mask = harmonized.columns == col
            harmonized[col] = harmonized.loc[:, mask].sum(axis=1)
        harmonized = harmonized.loc[:, ~harmonized.columns.duplicated()]

    n_renamed = sum(1 for k, v in mapping_dict.items() if k in data_df.columns)
    unmapped = [c for c in data_df.columns if c not in mapping_dict]

    return {
        'harmonized_df': harmonized,
        'n_renamed': n_renamed,
        'unmapped': unmapped,
    }


def record_preservation_bias(
    data_df: pd.DataFrame,
    proxy_type: str = 'pollen',
    environment: str = 'general',
    sensitive_taxa: Optional[List[str]] = None,
    tolerant_taxa: Optional[List[str]] = None,
    preset: Optional[str] = None,
) -> Dict:
    """分类群保存偏倚记录（多环境预设插件）。

    沉积环境导致分类群保存存在系统性偏倚。须在预处理中显式记录，
    并在结果解读时纳入考虑。

    使用方式：
    1. 调用预设：preset='pollen-karst'（自动填充 sensitive/tolerant）
    2. 自定义：传入 sensitive_taxa 和 tolerant_taxa 列表
    3. 混合：preset + 覆盖部分参数

    内置预设见 PRESERVATION_BIAS_PRESETS：
    - pollen-karst: 喀斯特碱性土壤（喜酸低估）
    - pollen-arid: 干旱区（薄壁孢子降解）
    - pollen-tropical: 热带氧化环境（薄壁分类群降解）
    - diatom-lake: 湖泊硅藻（薄壳属种溶解）
    - foraminifera-marine: 海洋有孔虫（薄壳种溶解）

    Parameters
    ----------
    data_df : pd.DataFrame
        分类群百分比数据。
    proxy_type : str, optional
        代理类型（'pollen'/'diatom'/'foraminifera' 等），用于查找预设。
    environment : str, optional
        环境类型（'karst'/'arid'/'tropical'/'lake'/'marine'/'general'）。
    sensitive_taxa : list, optional
        保存敏感分类群列表。提供时覆盖预设。
    tolerant_taxa : list, optional
        保存耐受分类群列表。提供时覆盖预设。
    preset : str, optional
        直接指定预设键名（如 'pollen-karst'），优先于 proxy_type×environment。

    Returns
    -------
    Dict
        {'sensitive_ratio': pd.Series, 'tolerant_ratio': pd.Series,
         'bias_index': pd.Series, 'preset': str, 'note': str}
    """
    # 解析预设
    preset_key = preset if preset is not None else f'{proxy_type}-{environment}'
    preset_data = PRESERVATION_BIAS_PRESETS.get(preset_key, {})

    sensitive = sensitive_taxa if sensitive_taxa is not None else preset_data.get('sensitive', [])
    tolerant = tolerant_taxa if tolerant_taxa is not None else preset_data.get('tolerant', [])
    note = preset_data.get('note', f'自定义保存偏倚（proxy={proxy_type}, env={environment}）。')

    if not sensitive or not tolerant:
        return {
            'sensitive_ratio': pd.Series(np.nan, index=data_df.index),
            'tolerant_ratio': pd.Series(np.nan, index=data_df.index),
            'bias_index': pd.Series(np.nan, index=data_df.index),
            'preset': preset_key,
            'note': f'未找到预设 {preset_key} 且未提供 sensitive/tolerant 列表。'
                    f'可用预设: {list(PRESERVATION_BIAS_PRESETS.keys())}',
        }

    sensitive_present = [t for t in sensitive if t in data_df.columns]
    tolerant_present = [t for t in tolerant if t in data_df.columns]

    sensitive_sum = data_df[sensitive_present].sum(axis=1) if sensitive_present else 0
    tolerant_sum = data_df[tolerant_present].sum(axis=1) if tolerant_present else 0
    total = data_df.sum(axis=1)

    sensitive_ratio = sensitive_sum / total.replace(0, np.nan)
    tolerant_ratio = tolerant_sum / total.replace(0, np.nan)
    # 偏倚指数 = 耐受/敏感，值越大偏倚越严重
    bias_index = tolerant_ratio / sensitive_ratio.replace(0, np.nan)

    return {
        'sensitive_ratio': sensitive_ratio,
        'tolerant_ratio': tolerant_ratio,
        'bias_index': bias_index,
        'preset': preset_key,
        'note': note + '须在结果解读时纳入考虑，可作为敏感性检验之一。',
    }


# 向后兼容别名
harmonize_taxon_names = harmonize_names

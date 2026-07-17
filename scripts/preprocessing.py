"""
古生态学与古气候学数据预处理模块（多代理、多区域通用）。
涵盖 BAM 年龄模型、z-score 标准化、时空对齐、分类群命名统一、保存偏倚记录。

支持两类代理数据：
- 分类群百分比型（花粉、孢粉、硅藻、有孔虫等）：harmonize_names / record_preservation_bias
- 连续值型（δDwax、brGDGTs、粒度、有机碳等）：见 continuous_proxy.py

文献来源：
- Comboul 2014 [9]: BAM 年龄模型
- Kaufman 2020 [2]: 时空对齐、年龄集合消费
- Izdebski 2022 [1]: z-score 标准化、地理区域聚类
- Power 2008 [4]: z-score 标准化
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


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


def bam_age_ensemble(
    depths: np.ndarray,
    ages: np.ndarray,
    age_errors: np.ndarray,
    n_members: int = 500,
) -> Dict:
    """Comboul 2014 BAM 年龄模型：生成蒙特卡洛年龄集合。

    从年龄-深度数据生成年龄集合，每个成员保持地层单调性。
    BAM 的 RMSE 为 251 年，与 Bacon (198 年) 可比 (Kaufman 2020)。
    适用于任意区域、任意沉积物类型（湖泊、海洋、泥炭、石笋等）。

    Parameters
    ----------
    depths : np.ndarray
        沉积深度数组 (n_depths,)。
    ages : np.ndarray
        对应年龄数组 (n_depths,)。
    age_errors : np.ndarray
        年龄误差 (1σ) 数组 (n_depths,)。
    n_members : int, optional
        集合成员数，默认 500 (Kaufman 2020)。

    Returns
    -------
    Dict
        {'age_ensembles': np.ndarray (n_members, n_depths),
         'n_members': int, 'method': 'BAM'}
    """
    n_depths = len(depths)
    ensembles = np.zeros((n_members, n_depths))

    for i in range(n_members):
        perturbed = ages + np.random.normal(0, age_errors)
        # 确保地层单调性：年龄随深度递增
        for j in range(1, n_depths):
            if perturbed[j] <= perturbed[j - 1]:
                perturbed[j] = perturbed[j - 1] + abs(age_errors[j])
        ensembles[i] = perturbed

    return {
        'age_ensembles': ensembles,
        'n_members': n_members,
        'method': 'BAM',
    }


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
        df = pd.read_csv(bacon_output_path, delim_whitespace=True)
    except Exception:
        df = pd.read_csv(bacon_output_path)

    if 'depth' in df.columns:
        depths = df['depth'].values
        ensemble_cols = [c for c in df.columns if c != 'depth']
        age_ensembles = df[ensemble_cols].values.T
    else:
        age_ensembles = df.values.T
        depths = np.arange(df.shape[0])

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
        基准时段 (start, end)。默认整个研究时段 (Izdebski 2022)。
    group_col : str, optional
        DataFrame 中的分组列名（如分类群名、岩芯 ID 或代理类型）。

    Returns
    -------
    Dict
        {'z_scores': same type as input, 'means': dict, 'stds': dict, 'baseline': str}
    """
    if isinstance(data, pd.DataFrame) and group_col is not None:
        groups = data[group_col].unique()
        z_data = data.copy()
        means, stds = {}, {}
        for g in groups:
            mask = data[group_col] == g
            values = data.loc[mask].select_dtypes(include=[np.number])
            mu = values.mean()
            sigma = values.std()
            z_data.loc[mask, values.columns] = (values - mu) / sigma
            means[g] = mu.to_dict()
            stds[g] = sigma.to_dict()
        baseline = 'full_period' if baseline_period is None else str(baseline_period)
        return {'z_scores': z_data, 'means': means, 'stds': stds, 'baseline': baseline}

    arr = np.asarray(data, dtype=float)
    if baseline_period is not None:
        mask = (arr >= baseline_period[0]) & (arr <= baseline_period[1])
        mu = np.nanmean(arr[mask])
        sigma = np.nanstd(arr[mask])
    else:
        mu = np.nanmean(arr)
        sigma = np.nanstd(arr)

    z = (arr - mu) / sigma
    baseline = 'full_period' if baseline_period is None else str(baseline_period)
    return {'z_scores': z, 'mean': mu, 'std': sigma, 'baseline': baseline}


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
    if values.ndim == 1:
        values = values[:, np.newaxis]

    if age_ensembles is not None:
        n_members = age_ensembles.shape[0]
        n_vars = values.shape[1]
        resampled = np.zeros((n_members, len(time_grid), n_vars))
        for m in range(n_members):
            for t in range(n_vars):
                resampled[m, :, t] = np.interp(
                    time_grid, age_ensembles[m], values[:, t]
                )
        if n_vars == 1:
            resampled = resampled[:, :, 0]
        return {'resampled': resampled, 'time_grid': time_grid, 'n_bins': len(time_grid)}

    n_vars = values.shape[1]
    resampled = np.zeros((len(time_grid), n_vars))
    for t in range(n_vars):
        resampled[:, t] = np.interp(time_grid, ages, values[:, t])
    if n_vars == 1:
        resampled = resampled[:, 0]
    return {'resampled': resampled, 'time_grid': time_grid, 'n_bins': len(time_grid)}


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
        - 'grid'：等面积经纬度网格化（Kaufman 2020）。
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
        # 等面积网格化
        clusters = {}
        for site in sites:
            lat = site_coords.loc[site, 'lat']
            lon = site_coords.loc[site, 'lon']
            grid_lat = round(lat / grid_resolution) * grid_resolution
            grid_lon = round(lon / grid_resolution) * grid_resolution
            key = f'{grid_lat}_{grid_lon}'
            clusters.setdefault(key, []).append(site)
        # 将 key 转为序号
        clusters = {i + 1: v for i, v in enumerate(clusters.values())}
        return {
            'clusters': clusters, 'method': 'grid',
            'radius_km': None, 'grid_resolution': grid_resolution,
        }

    # cluster 模式：基于距离的地理聚类
    assigned = [False] * n
    clusters = {}
    cluster_id = 0

    for i in range(n):
        if assigned[i]:
            continue
        cluster_id += 1
        clusters[cluster_id] = [sites[i]]
        assigned[i] = True
        lat_i = site_coords.loc[sites[i], 'lat']
        lon_i = site_coords.loc[sites[i], 'lon']
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            lat_j = site_coords.loc[sites[j], 'lat']
            lon_j = site_coords.loc[sites[j], 'lon']
            dist = haversine(lat_i, lon_i, lat_j, lon_j)
            if dist <= radius_km:
                clusters[cluster_id].append(sites[j])
                assigned[j] = True

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

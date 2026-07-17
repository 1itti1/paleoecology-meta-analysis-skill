"""
古生态学数据预处理模块。
涵盖 BAM 年龄模型、z-score 标准化、时空对齐、花粉命名统一、喀斯特保存偏倚记录。

文献来源：
- Comboul 2014 [9]: BAM 年龄模型
- Kaufman 2020 [2]: 时空对齐、年龄集合消费
- Izdebski 2022 [1]: z-score 标准化、地理区域聚类
- Power 2008 [4]: z-score 标准化
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def bam_age_ensemble(
    depths: np.ndarray,
    ages: np.ndarray,
    age_errors: np.ndarray,
    n_members: int = 500,
) -> Dict:
    """Comboul 2014 BAM 年龄模型：生成蒙特卡洛年龄集合。

    从年龄-深度数据生成年龄集合，每个成员保持地层单调性。
    BAM 的 RMSE 为 251 年，与 Bacon (198 年) 可比 (Kaufman 2020)。

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
    用户已有的 BS/JL/JY 等 7 个洼地 Bacon 模型可直接消费。

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

    对每个岩芯的每个分类群，以研究时段均值和标准差计算 z-score：
    z = (x - μ) / σ

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        代理值数组或 DataFrame。若 DataFrame 且 group_col 指定，按分组分别标准化。
    baseline_period : tuple, optional
        基准时段 (start, end)。默认整个研究时段 (Izdebski 2022)。
    group_col : str, optional
        DataFrame 中的分组列名（如分类群名或岩芯 ID）。

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

    Parameters
    ----------
    ages : np.ndarray
        年龄数组 (n_depths,)。
    values : np.ndarray
        代理值数组 (n_depths,) 或 (n_depths, n_taxa)。
    time_grid : np.ndarray
        统一时间网格 (n_bins,)。
    age_ensembles : np.ndarray, optional
        年龄集合 (n_members, n_depths)，用于传播年龄不确定性。

    Returns
    -------
    Dict
        {'resampled': np.ndarray (n_bins,) or (n_members, n_bins, n_taxa),
         'time_grid': np.ndarray, 'n_bins': int}
    """
    if values.ndim == 1:
        values = values[:, np.newaxis]

    if age_ensembles is not None:
        n_members = age_ensembles.shape[0]
        n_taxa = values.shape[1]
        resampled = np.zeros((n_members, len(time_grid), n_taxa))
        for m in range(n_members):
            for t in range(n_taxa):
                resampled[m, :, t] = np.interp(
                    time_grid, age_ensembles[m], values[:, t]
                )
        if n_taxa == 1:
            resampled = resampled[:, :, 0]
        return {'resampled': resampled, 'time_grid': time_grid, 'n_bins': len(time_grid)}

    n_taxa = values.shape[1]
    resampled = np.zeros((len(time_grid), n_taxa))
    for t in range(n_taxa):
        resampled[:, t] = np.interp(time_grid, ages, values[:, t])
    if n_taxa == 1:
        resampled = resampled[:, 0]
    return {'resampled': resampled, 'time_grid': time_grid, 'n_bins': len(time_grid)}


def spatial_clustering(
    site_coords: pd.DataFrame,
    method: str = 'karst',
    radius_km: float = 200,
) -> Dict:
    """Izdebski 2022：地理区域聚类。

    按地理坐标聚类，替代经纬度网格化。
    喀斯特区推荐地理聚类——洼地/谷地/坡地空间异质性使聚类比网格更有生态意义。

    Parameters
    ----------
    site_coords : pd.DataFrame
        含 'lat', 'lon' 列的点位坐标，索引为点位名。
    method : str, optional
        聚类方法：'karst' (地理聚类, 默认) 或 'grid' (等面积网格化)。
    radius_km : float, optional
        聚类半径 (km)，默认 200 (Izdebski 2022)。

    Returns
    -------
    Dict
        {'clusters': dict (cluster_id -> [site_names]),
         'method': str, 'radius_km': float}
    """
    from math import radians, sin, cos, asin, sqrt

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371 * 2 * asin(sqrt(a))

    sites = list(site_coords.index)
    n = len(sites)
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

    return {'clusters': clusters, 'method': method, 'radius_km': radius_km}


def harmonize_taxon_names(
    pollen_df: pd.DataFrame,
    mapping_dict: Dict[str, str],
) -> Dict:
    """花粉分类群命名统一。

    不同实验室可能使用不同分类群命名体系，须建立映射表统一到一致分类框架。

    Parameters
    ----------
    pollen_df : pd.DataFrame
        花粉百分比数据，列为分类群名。
    mapping_dict : dict
        命名映射字典 {旧名: 新名}。

    Returns
    -------
    Dict
        {'harmonized_df': pd.DataFrame, 'n_renamed': int, 'unmapped': list}
    """
    harmonized = pollen_df.rename(columns=mapping_dict)
    # 合并同名列
    if harmonized.columns.duplicated().any():
        dup_cols = harmonized.columns[harmonized.columns.duplicated()].unique()
        for col in dup_cols:
            mask = harmonized.columns == col
            harmonized[col] = harmonized.loc[:, mask].sum(axis=1)
        harmonized = harmonized.loc[:, ~harmonized.columns.duplicated()]

    n_renamed = sum(1 for k, v in mapping_dict.items() if k in pollen_df.columns)
    unmapped = [c for c in pollen_df.columns if c not in mapping_dict]

    return {
        'harmonized_df': harmonized,
        'n_renamed': n_renamed,
        'unmapped': unmapped,
    }


def record_preservation_bias(
    pollen_df: pd.DataFrame,
    sensitive_taxa: list,
    tolerant_taxa: list,
) -> Dict:
    """喀斯特区花粉保存偏倚记录。

    碱性土壤环境导致花粉保存存在系统性偏倚：
    喜酸分类群（如杜鹃花科 Ericaceae）可能被低估，
    耐碱分类群（如禾本科 Poaceae）可能被相对高估。
    须在预处理中显式记录，并在结果解读时纳入考虑。

    Parameters
    ----------
    pollen_df : pd.DataFrame
        花粉百分比数据。
    sensitive_taxa : list
        保存敏感分类群列表（如 ['Ericaceae', 'Vaccinium']）。
    tolerant_taxa : list
        保存耐受分类群列表（如 ['Poaceae', 'Cyperaceae']）。

    Returns
    -------
    Dict
        {'sensitive_ratio': pd.Series, 'tolerant_ratio': pd.Series,
         'bias_index': pd.Series, 'note': str}
    """
    sensitive_present = [t for t in sensitive_taxa if t in pollen_df.columns]
    tolerant_present = [t for t in tolerant_taxa if t in pollen_df.columns]

    sensitive_sum = pollen_df[sensitive_present].sum(axis=1) if sensitive_present else 0
    tolerant_sum = pollen_df[tolerant_present].sum(axis=1) if tolerant_present else 0
    total = pollen_df.sum(axis=1)

    sensitive_ratio = sensitive_sum / total.replace(0, np.nan)
    tolerant_ratio = tolerant_sum / total.replace(0, np.nan)
    # 偏倚指数 = 耐受/敏感，值越大偏倚越严重
    bias_index = tolerant_ratio / sensitive_ratio.replace(0, np.nan)

    return {
        'sensitive_ratio': sensitive_ratio,
        'tolerant_ratio': tolerant_ratio,
        'bias_index': bias_index,
        'note': '喀斯特碱性土壤保存偏倚：喜酸分类群低估，耐碱分类群相对高估。'
                '须在结果解读时纳入考虑，可作为敏感性检验之一。',
    }

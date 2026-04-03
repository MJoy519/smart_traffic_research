"""
训练数据集构建模块

功能：
  1. 加载特征提取结果（yolo / segformer / yolopv2 / composite）并合并
  2. 加载窗口级 V-A 连续值标签（由 label_generator.py 生成）
  3. 按 (route_num, video_name, window_idx) 对齐特征与标签
  4. 输出可直接用于回归训练的 (X, y_valence, meta) 数据集

运行方式：
  python dataset_builder.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

_LCC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LCC_DIR)
import importlib.util as _ilu

def _load_lcc_config():
    spec = _ilu.spec_from_file_location("lcc_config", os.path.join(_LCC_DIR, "config.py"))
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg = _load_lcc_config()

# ============================================================
#                       参数
# ============================================================

# 使用的特征子集（可组合: 'yolo', 'segformer', 'yolopv2', 'com'）
USE_EXTRACTORS = ['yolo', 'segformer', 'yolopv2', 'com']

# 输出文件名前缀
DATASET_PREFIX = 'training_dataset'

# 是否删除低 n_valid 窗口（受试者覆盖不足的窗口）
MIN_PARTICIPANTS = 3   # 至少需要 N 名受试者的情绪数据

# 是否添加交叉特征（特征乘积项）
ADD_CROSS_FEATURES = False

# ============================================================
#              特征列定义（与各 extractor 输出一致）
# ============================================================

YOLO_FEATURE_COLS = [
    'car_count', 'cyclist_motorcycle_count', 'dynamic_object_area_ratio',
    'large_vehicle_ratio', 'person_count', 'total_object_count',
    'traffic_sign_count', 'truck_bus_count',
    'car_speed_mean', 'car_accel_mean', 'car_jerk_mean',
    'person_speed_mean', 'cyclist_speed_mean',
    'pedestrian_crossing_count', 'min_ttc', 'risk_count',
]

SEGFORMER_FEATURE_COLS = [
    'building_coverage', 'building_oppression', 'green_coverage',
    'openness_index', 'road_coverage', 'sidewalk_coverage',
    'sky_visibility', 'wall_fence_coverage',
]

YOLOPV2_FEATURE_COLS = [
    'drivable_coverage', 'drivable_width_mean', 'drivable_width_min',
    'lane_count_visible', 'lane_curvature_mean', 'lane_marking_visibility',
    'lane_offset', 'road_curvature_max', 'road_curvature_mean',
]

COMPOSITE_FEATURE_COLS = [
    'drivable_occupancy_ratio', 'vru_drivable_intrusion_rate',
    'interaction_risk_integral_itcc', 'enclosure_crowding_stress',
    'green_buffer_under_congestion', 'exposed_vru_conflict_index',
    'semantic_monotony_fatigue',
]

EXTRACTOR_COLS = {
    'yolo':      YOLO_FEATURE_COLS,
    'segformer': SEGFORMER_FEATURE_COLS,
    'yolopv2':   YOLOPV2_FEATURE_COLS,
    'com':       COMPOSITE_FEATURE_COLS,
}

META_COLS = ['route_num', 'video_name', 'window_idx', 'window_start_sec', 'window_end_sec']

# ============================================================
#                   特征加载
# ============================================================

def load_extractor_csv(route: int, extractor: str) -> pd.DataFrame:
    """
    加载单条路线单提取器的特征 CSV。

    Parameters
    ----------
    route     : int  路线编号
    extractor : str  提取器名 ('yolo' | 'segformer' | 'yolopv2' | 'com')

    Returns
    -------
    pd.DataFrame 或空 DataFrame（文件不存在时）
    """
    feat_dir  = cfg.FEATURE_DIR / cfg.FEATURE_DIR_LABEL
    filename  = f'{route}_{extractor}_{cfg.WINDOW_SIZE_SEC}_{cfg.STEP_SIZE_SEC}.csv'
    path      = feat_dir / filename

    if not path.exists():
        warnings.warn(f"特征文件不存在，跳过: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df


def load_all_features(routes: list) -> pd.DataFrame:
    """
    加载所有路线、所有提取器的特征，合并为单一宽表。

    合并键：route_num, video_name, window_idx
    有特征文件不存在时给出警告但不中断。

    Returns
    -------
    pd.DataFrame  宽表，含元数据列 + 所有特征列
    """
    all_route_dfs = []

    for route in routes:
        route_dfs = []

        for ext in USE_EXTRACTORS:
            df = load_extractor_csv(route, ext)
            if df.empty:
                continue

            # 保留元数据列 + 该提取器特征列
            feat_cols = [c for c in EXTRACTOR_COLS.get(ext, []) if c in df.columns]
            if not feat_cols:
                warnings.warn(f"Route {route} {ext}: 特征列均不存在，跳过")
                continue

            keep_cols = [c for c in META_COLS if c in df.columns] + feat_cols
            route_dfs.append(df[keep_cols].copy())

        if not route_dfs:
            warnings.warn(f"Route {route}: 无有效特征文件")
            continue

        # 以第一个 df 为基础，逐步左连接其余提取器
        df_base = route_dfs[0]
        for df_other in route_dfs[1:]:
            join_keys = [c for c in META_COLS if c in df_base.columns and c in df_other.columns]
            df_base = pd.merge(df_base, df_other, on=join_keys, how='left',
                               suffixes=('', '_dup'))
            # 删除重复后缀列
            dup_cols = [c for c in df_base.columns if c.endswith('_dup')]
            df_base.drop(columns=dup_cols, inplace=True)

        all_route_dfs.append(df_base)

    if not all_route_dfs:
        raise ValueError("未找到任何有效特征文件，请先运行特征提取。")

    df_features = pd.concat(all_route_dfs, ignore_index=True)
    print(f"  特征表: {df_features.shape[0]} 行  ×  {df_features.shape[1]} 列")
    return df_features


# ============================================================
#               标签加载
# ============================================================

def load_labels() -> pd.DataFrame:
    """
    加载由 label_generator.py 生成的窗口标签文件。

    Returns
    -------
    pd.DataFrame 含 route_num, video_name, window_idx, valence_median, arousal_median,
                     label, label_name, n_valid 等列
    """
    label_file = cfg.OUTPUT_DIR / 'window_labels.csv'
    if not label_file.exists():
        raise FileNotFoundError(
            f"标签文件不存在: {label_file}\n"
            "请先运行 label_generation_cc/label_generator.py"
        )
    return pd.read_csv(label_file)


# ============================================================
#               数据集合并
# ============================================================

def build_dataset(
    routes: list = None,
) -> tuple:
    """
    构建训练数据集：合并特征与标签。

    Parameters
    ----------
    routes : list  目标路线（None 时使用 cfg.TARGET_ROUTES）

    Returns
    -------
    (df_dataset, X, y, feature_cols, meta_df)
      df_dataset  : 完整 DataFrame（含 valence_median, arousal_median）
      X           : np.ndarray  特征矩阵 (n_samples, n_features)
      y           : np.ndarray  valence_median 向量 (n_samples,)
      feature_cols: list        特征列名
      meta_df     : pd.DataFrame  元数据（route, video, window）
    """
    if routes is None:
        routes = cfg.TARGET_ROUTES

    print(f"\n  加载特征 (routes={routes}, extractors={USE_EXTRACTORS}) ...")
    df_features = load_all_features(routes)

    print(f"  加载标签 ...")
    df_labels = load_labels()

    # -------- 过滤低 n_valid 窗口 --------
    if 'n_valid' in df_labels.columns:
        before = len(df_labels)
        df_labels = df_labels[df_labels['n_valid'] >= MIN_PARTICIPANTS].copy()
        dropped = before - len(df_labels)
        if dropped > 0:
            print(f"  [过滤] 移除 n_valid < {MIN_PARTICIPANTS} 的窗口: {dropped} 行")

    # -------- 合并 --------
    join_keys = ['route_num', 'video_name', 'window_idx']

    # 确保数值类型一致
    for k in join_keys:
        if k in df_features.columns:
            df_features[k] = df_features[k].astype(str)
        if k in df_labels.columns:
            df_labels[k] = df_labels[k].astype(str)

    label_keep = join_keys + [
        c for c in [
            'valence_median', 'arousal_median',
            'valence_std', 'arousal_std',
            'valence_iqr', 'arousal_iqr',
            'n_valid',
        ]
        if c in df_labels.columns
    ]
    df_dataset = pd.merge(
        df_features,
        df_labels[label_keep],
        on=join_keys,
        how='inner',
    )

    print(f"  合并后行数: {len(df_dataset)}")

    # -------- 删除 V-A 标签缺失行 --------
    df_dataset = df_dataset.dropna(subset=['valence_median', 'arousal_median']).copy()
    print(f"  有效样本数 (V-A 非NaN): {len(df_dataset)}")

    # -------- 确定特征列 --------
    all_feat_cols = []
    for ext in USE_EXTRACTORS:
        for c in EXTRACTOR_COLS.get(ext, []):
            if c in df_dataset.columns and c not in all_feat_cols:
                all_feat_cols.append(c)

    # -------- 处理特征缺失值 --------
    df_dataset[all_feat_cols] = df_dataset[all_feat_cols].fillna(
        df_dataset[all_feat_cols].median()
    )

    # -------- 可选：交叉特征 --------
    if ADD_CROSS_FEATURES:
        _add_cross_features(df_dataset, all_feat_cols)

    # -------- 构建 X, y（y 为 valence_median，供调用方参考；回归时直接用 valence/arousal 列）--------
    X = df_dataset[all_feat_cols].values.astype(np.float32)
    y = df_dataset['valence_median'].values.astype(float)

    # route_num 恢复为数值
    df_dataset['route_num'] = pd.to_numeric(df_dataset['route_num'], errors='coerce')
    df_dataset['window_idx'] = pd.to_numeric(df_dataset['window_idx'], errors='coerce')

    meta_cols_available = [c for c in META_COLS if c in df_dataset.columns]
    meta_df = df_dataset[meta_cols_available].copy()

    return df_dataset, X, y, all_feat_cols, meta_df


def _add_cross_features(df: pd.DataFrame, feat_cols: list):
    """添加少量有物理意义的交叉特征。"""
    new_cols = []

    def _safe_add(name, col1, col2, op='multiply'):
        if col1 in df.columns and col2 in df.columns:
            if op == 'multiply':
                df[name] = df[col1] * df[col2]
            elif op == 'ratio':
                df[name] = df[col1] / (df[col2] + 1e-6)
            feat_cols.append(name)
            new_cols.append(name)

    _safe_add('risk_x_congestion',   'risk_count',           'enclosure_crowding_stress')
    _safe_add('vru_x_speed',         'person_count',         'car_speed_mean')
    _safe_add('green_openness',      'green_coverage',       'openness_index')
    _safe_add('drivable_x_lane',     'drivable_coverage',    'lane_count_visible')

    if new_cols:
        print(f"  [交叉特征] 新增 {len(new_cols)} 列: {new_cols}")


# ============================================================
#               数据集可视化
# ============================================================

def plot_feature_correlation(df_dataset: pd.DataFrame, feature_cols: list,
                              top_n: int = 30):
    """绘制特征相关矩阵热力图（取前 top_n 个特征）。"""
    cols = feature_cols[:top_n]
    corr = df_dataset[cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=~mask, annot=False, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.3, square=True)
    ax.set_title(f'Feature Correlation Matrix (Top {top_n} Features)', fontsize=14)
    plt.tight_layout()
    _savefig(fig, 'feature_correlation.png')


def _savefig(fig, filename: str):
    if cfg.SAVE_FIGURES:
        cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = cfg.FIGURES_DIR / filename
        fig.savefig(save_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f"  图片已保存: {save_path}")
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)


# ============================================================
#                         主流程
# ============================================================

def main():
    print(f"\n{'=' * 65}")
    print(f"  数据集构建  |  提取器: {USE_EXTRACTORS}")
    print(f"  路线: {cfg.TARGET_ROUTES}  |  最低受试者数: {MIN_PARTICIPANTS}")
    print(f"{'=' * 65}")

    df_dataset, X, y, feature_cols, meta_df = build_dataset()

    print(f"\n  特征维度: {X.shape[1]}  |  样本量: {X.shape[0]}")
    y_v = df_dataset['valence_median'].dropna()
    y_a = df_dataset['arousal_median'].dropna()
    print(f"  Valence: mean={y_v.mean():.4f}  std={y_v.std():.4f}"
          f"  range=[{y_v.min():.3f}, {y_v.max():.3f}]")
    print(f"  Arousal: mean={y_a.mean():.4f}  std={y_a.std():.4f}"
          f"  range=[{y_a.min():.3f}, {y_a.max():.3f}]")

    # 保存数据集
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_file = cfg.OUTPUT_DIR / f'{DATASET_PREFIX}.csv'
    df_dataset.to_csv(dataset_file, index=False)
    print(f"\n  数据集已保存: {dataset_file}")

    # 保存特征列名
    feat_col_file = cfg.OUTPUT_DIR / 'feature_columns.txt'
    with open(feat_col_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"  特征列表已保存: {feat_col_file}")

    # 可视化
    if cfg.SAVE_FIGURES or cfg.SHOW_FIGURES:
        print(f"\n  生成可视化 → {cfg.FIGURES_DIR}")
        plot_feature_correlation(df_dataset, feature_cols)

    print(f"\n{'=' * 65}")
    print(f"  数据集构建完成！")
    print(f"{'=' * 65}")
    return df_dataset, X, y, feature_cols, meta_df


if __name__ == '__main__':
    main()

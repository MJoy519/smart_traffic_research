"""
GMM 无监督聚类脚本
对情绪时间窗口特征进行高斯混合模型（GMM）聚类。

支持两种聚类模式（由 CLUSTERING_MODE 控制）：

  three_class  —— 聚成 3 类（Positive / Neutral / Negative）
    判定逻辑：
      positive_score = Joy 激活比例均值
      negative_score = [Anger, Contempt, Disgust, Fear, Sadness, Confusion] 均值之均值
      net_score      = positive_score - negative_score
      排序：最高 → Positive，最低 → Negative，中间 → Neutral

  two_class    —— 聚成 2 类（Negative-Active / Inactive）
    判定逻辑：
      neg_score = [Anger, Contempt, Disgust, Fear, Sadness, Confusion] 均值之均值
      排序：较高 → Negative-Active，较低 → Inactive
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
#                       全局可控参数
# ============================================================
from config import WINDOW_SIZE, STEP_SIZE, EMOTION_TARGETS
# -- 窗口参数（需与 3. windowing.py 保持一致）--
WINDOW_SIZE_SEC = WINDOW_SIZE
STEP_SIZE_SEC   = STEP_SIZE

# -- 目标情绪列 --
EMOTION_COLUMNS = EMOTION_TARGETS

# -- 聚类模式 --
# 'three_class' : 聚成 3 类（Positive / Neutral / Negative）
# 'two_class'   : 聚成 2 类（Negative-Active / Inactive）
CLUSTERING_MODE = 'two_class'

# -- 情绪极性配置（用于标签判定）--
POSITIVE_EMOTIONS = ['Joy']
NEGATIVE_EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Sadness', 'Confusion']

# -- GMM 参数 --
# N_COMPONENTS 由 CLUSTERING_MODE 自动决定，无需手动修改
N_COMPONENTS    = 3 if CLUSTERING_MODE == 'three_class' else 2
COVARIANCE_TYPE = 'full'        # 协方差类型: 'full' | 'tied' | 'diag' | 'spherical'
N_INIT          = 10            # 多次初始化取最优（提高稳定性）
RANDOM_STATE    = 42

# -- 可视化参数 --
SAVE_FIGURES    = True          # 是否保存可视化图片
SHOW_FIGURES    = False         # 是否在屏幕显示（服务器环境建议 False）

# -- 处理线路选择 --
TARGET_ROUTES = [1, 2, 3]

# ============================================================
#                       数据路径
# ============================================================

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'label_generation' / 'results'
INPUT_FILE  = RESULTS_DIR / f'emotion_windows_ws{WINDOW_SIZE_SEC}_st{STEP_SIZE_SEC}_median.csv'

# 输出文件名因模式不同而不同，三分类保持原有命名以保证向后兼容
_mode_suffix = '' if CLUSTERING_MODE == 'three_class' else '_2class'
OUTPUT_FILE = RESULTS_DIR / (
    f'emotion_windows_ws{WINDOW_SIZE_SEC}_st{STEP_SIZE_SEC}_median_labeled{_mode_suffix}.csv'
)
FIG_DIR     = BASE_DIR / 'label_generation' / 'figures' / 'clustering'

# ============================================================
#                       聚类核心函数
# ============================================================


def assign_polarity_labels(cluster_centers_orig):
    """
    根据各聚类中心的情绪均值判定正面/中性/负面标签。

    Parameters
    ----------
    cluster_centers_orig : dict  {cluster_id: {emotion: mean_value, ...}, ...}

    Returns
    -------
    label_map : dict  {cluster_id: 'Positive'/'Neutral'/'Negative'}
    net_scores : dict {cluster_id: net_score}
    """
    net_scores = {}
    for cid, center in cluster_centers_orig.items():
        pos_score = np.mean([center.get(e, 0.0) for e in POSITIVE_EMOTIONS])
        neg_score = np.mean([center.get(e, 0.0) for e in NEGATIVE_EMOTIONS])
        net_scores[cid] = float(pos_score - neg_score)

    sorted_ids = sorted(net_scores, key=lambda x: net_scores[x])   # 升序
    label_map = {
        sorted_ids[0]: 'Negative',
        sorted_ids[1]: 'Neutral',
        sorted_ids[2]: 'Positive',
    }
    return label_map, net_scores


def assign_binary_labels(cluster_centers_orig):
    """
    两类模式：按各聚类中心的 neg_score 高低判定标签。

    Parameters
    ----------
    cluster_centers_orig : dict  {cluster_id: {emotion: mean_value, ...}, ...}

    Returns
    -------
    label_map  : dict  {cluster_id: 'Negative-Active'/'Inactive'}
    neg_scores : dict  {cluster_id: neg_score}
    """
    neg_scores = {}
    for cid, center in cluster_centers_orig.items():
        neg_score = np.mean([center.get(e, 0.0) for e in NEGATIVE_EMOTIONS])
        neg_scores[cid] = float(neg_score)

    sorted_ids = sorted(neg_scores, key=lambda x: neg_scores[x])   # 升序
    label_map = {
        sorted_ids[0]: 'Inactive',
        sorted_ids[1]: 'Negative-Active',
    }
    return label_map, neg_scores


def run_gmm_clustering(df_windows):
    """
    执行 GMM 聚类，返回带标签的 DataFrame 及聚类相关对象。

    Parameters
    ----------
    df_windows : pd.DataFrame  窗口数据集

    Returns
    -------
    df_labeled       : pd.DataFrame   添加 cluster_id 和 emotion_label 列
    gmm              : GaussianMixture
    scaler           : StandardScaler
    label_map        : dict  {cluster_id → label}
    cluster_centers  : dict  {cluster_id → {emotion → mean}}
    X_scaled_valid   : np.ndarray  标准化后有效样本
    cluster_ids_valid: np.ndarray  有效样本对应聚类编号
    valid_mask       : np.ndarray  bool，指示哪些行是有效样本
    """
    available_emos = [e for e in EMOTION_COLUMNS if e in df_windows.columns]
    X = df_windows[available_emos].values.astype(float)

    valid_mask = ~np.isnan(X).any(axis=1)
    X_valid    = X[valid_mask]

    print(f"\n  特征维度: {len(available_emos)}  |  有效窗口: {valid_mask.sum()} / {len(X)}")

    # 标准化
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # GMM 拟合
    print(f"  训练 GMM  (n_components={N_COMPONENTS}, "
          f"covariance_type='{COVARIANCE_TYPE}', n_init={N_INIT}) ...")
    gmm = GaussianMixture(
        n_components    = N_COMPONENTS,
        covariance_type = COVARIANCE_TYPE,
        n_init          = N_INIT,
        random_state    = RANDOM_STATE,
        max_iter        = 200,
    )
    gmm.fit(X_scaled)

    cluster_ids_valid = gmm.predict(X_scaled)
    proba_valid       = gmm.predict_proba(X_scaled)

    # 原始空间聚类中心
    centers_orig = scaler.inverse_transform(gmm.means_)
    cluster_centers = {
        cid: dict(zip(available_emos, centers_orig[cid]))
        for cid in range(N_COMPONENTS)
    }

    print(f"\n  各聚类中心（原始空间）:")
    for cid, center in cluster_centers.items():
        vals_str = '  '.join(f'{e}={v:.4f}' for e, v in center.items())
        size     = (cluster_ids_valid == cid).sum()
        print(f"    Cluster {cid} (n={size}): {vals_str}")

    # 标签判定（按模式分支）
    if CLUSTERING_MODE == 'three_class':
        label_map, scores = assign_polarity_labels(cluster_centers)
        score_name = 'net_score'
    else:
        label_map, scores = assign_binary_labels(cluster_centers)
        score_name = 'neg_score'

    print(f"\n  标签映射 ({CLUSTERING_MODE}):")
    for cid, label in label_map.items():
        print(f"    Cluster {cid}  →  {label}  ({score_name}={scores[cid]:+.4f})")

    # 写回标签
    df_labeled = df_windows.copy()
    df_labeled['cluster_id']       = np.nan
    df_labeled['emotion_label']    = ''
    df_labeled['label_confidence'] = np.nan

    df_labeled.loc[valid_mask, 'cluster_id'] = cluster_ids_valid.astype(float)
    df_labeled.loc[valid_mask, 'emotion_label'] = [
        label_map[c] for c in cluster_ids_valid
    ]
    df_labeled.loc[valid_mask, 'label_confidence'] = proba_valid.max(axis=1)

    # 分布统计
    dist  = df_labeled['emotion_label'].value_counts()
    total = dist.sum()
    print(f"\n  标签分布:")
    for label in _label_order():
        cnt = dist.get(label, 0)
        print(f"    {label:16s}: {cnt:5d}  ({cnt / total * 100:.1f}%)")

    return (df_labeled, gmm, scaler, label_map, cluster_centers,
            X_scaled, cluster_ids_valid, valid_mask)


# ============================================================
#                       可视化函数
# ============================================================

COLOR_MAP = {
    'Positive':        '#27AE60',   # 绿
    'Neutral':         '#3498DB',   # 蓝
    'Negative':        '#E74C3C',   # 红
    'Negative-Active': '#C0392B',   # 深红（二分类负面激活）
    'Inactive':        '#95A5A6',   # 灰（二分类未激活）
}


def _label_order():
    """返回当前模式下的标签显示顺序。"""
    if CLUSTERING_MODE == 'three_class':
        return ['Positive', 'Neutral', 'Negative']
    return ['Negative-Active', 'Inactive']


def _save_or_show(fig, filename):
    if SAVE_FIGURES:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIG_DIR / filename
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图片已保存: {save_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_pca_scatter(X_scaled, cluster_ids, label_map):
    """PCA 降维后绘制聚类散点图。"""
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    label_names = [label_map[c] for c in cluster_ids]

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in _label_order():
        idx = [i for i, l in enumerate(label_names) if l == label]
        if idx:
            ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                       c=COLOR_MAP[label], label=label,
                       alpha=0.35, s=8, edgecolors='none')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('GMM Clustering — Emotion Windows (PCA Projection)', fontsize=14)
    ax.legend(markerscale=4, fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save_or_show(fig, 'gmm_pca_scatter.png')


def plot_radar_chart(cluster_centers, label_map):
    """绘制各聚类中心情绪雷达图。"""
    emotions = list(next(iter(cluster_centers.values())).keys())
    N        = len(emotions)
    angles   = [n / float(N) * 2 * np.pi for n in range(N)]
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

    for label in _label_order():
        cid_list = [cid for cid, lbl in label_map.items() if lbl == label]
        if not cid_list:
            continue
        center = cluster_centers[cid_list[0]]
        values = [center[e] for e in emotions] + [center[emotions[0]]]
        ax.plot(angles, values, linewidth=2.0,
                label=label, color=COLOR_MAP[label])
        ax.fill(angles, values, alpha=0.12, color=COLOR_MAP[label])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotions, size=12)
    ax.set_title('Cluster Centers — Emotion Activation Profiles',
                 fontsize=14, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, 'gmm_radar.png')


def plot_cluster_barplot(cluster_centers, label_map):
    """绘制各聚类中心各情绪激活比例柱状图。"""
    emotions     = list(next(iter(cluster_centers.values())).keys())
    order        = _label_order()
    cid_by_label = {v: k for k, v in label_map.items()}

    x      = np.arange(len(emotions))
    n_bars = len(order)
    width  = 0.7 / n_bars                           # 自适应宽度
    offsets = [(i - (n_bars - 1) / 2) * width for i in range(n_bars)]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(order):
        if label not in cid_by_label:
            continue
        cid    = cid_by_label[label]
        center = cluster_centers[cid]
        vals   = [center[e] for e in emotions]
        ax.bar(x + offsets[i], vals, width,
               label=label, color=COLOR_MAP[label], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=11)
    ax.set_ylabel('Mean Activation Proportion', fontsize=11)
    ax.set_title('Cluster Centers — Mean Emotion Activation per Cluster', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, 'gmm_cluster_barplot.png')


def plot_label_distribution(df_labeled):
    """绘制各线路/全局标签分布饼图。"""
    routes = sorted(df_labeled['route'].dropna().unique())
    ncols  = len(routes) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))

    def _pie(ax, data, title):
        labels_order = _label_order()
        sizes  = [data.get(l, 0) for l in labels_order]
        colors = [COLOR_MAP[l] for l in labels_order]
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels_order, colors) if s > 0]
        if not non_zero:
            ax.set_visible(False)
            return
        s_, l_, c_ = zip(*non_zero)
        ax.pie(s_, labels=l_, colors=c_, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title(title, fontsize=12)

    # 各线路
    for i, route in enumerate(routes):
        sub  = df_labeled[df_labeled['route'] == route]
        dist = sub['emotion_label'].value_counts().to_dict()
        _pie(axes[i], dist, f'Route {route}')

    # 全局
    dist_all = df_labeled['emotion_label'].value_counts().to_dict()
    _pie(axes[-1], dist_all, 'All Routes')

    plt.suptitle('Emotion Window Label Distribution', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, 'gmm_label_distribution.png')


# ============================================================
#                        主入口
# ============================================================

def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] 输入文件不存在: {INPUT_FILE}")
        print("  请先运行 3. windowing.py 生成窗口数据集。")
        return

    mode_desc = ('三分类: Positive / Neutral / Negative'
                 if CLUSTERING_MODE == 'three_class'
                 else '二分类: Negative-Active / Inactive')
    print(f"\n{'=' * 60}")
    print(f"  GMM 情绪窗口聚类")
    print(f"  模式: {CLUSTERING_MODE}  ——  {mode_desc}")
    print(f"  输入: {INPUT_FILE.name}")
    print(f"  N_COMPONENTS={N_COMPONENTS}  COVARIANCE_TYPE='{COVARIANCE_TYPE}'")
    print(f"{'=' * 60}")

    df_windows = pd.read_csv(INPUT_FILE)

    # 线路过滤
    if TARGET_ROUTES:
        df_windows = df_windows[df_windows['route'].isin(TARGET_ROUTES)].copy()
        df_windows.reset_index(drop=True, inplace=True)

    print(f"  总窗口数: {len(df_windows)}  (线路: {sorted(df_windows['route'].unique())})")

    # GMM 聚类
    (df_labeled, gmm, scaler, label_map, cluster_centers,
     X_scaled, cluster_ids, valid_mask) = run_gmm_clustering(df_windows)

    # 保存结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  标注结果已保存: {OUTPUT_FILE}")

    # 可视化
    if SAVE_FIGURES or SHOW_FIGURES:
        print("\n  生成可视化图表 ...")
        vis_tasks = [
            ("PCA 散点图",     lambda: plot_pca_scatter(X_scaled, cluster_ids, label_map)),
            ("情绪雷达图",     lambda: plot_radar_chart(cluster_centers, label_map)),
            ("聚类柱状图",     lambda: plot_cluster_barplot(cluster_centers, label_map)),
            ("标签分布饼图",   lambda: plot_label_distribution(df_labeled)),
        ]
        for desc, fn in tqdm(vis_tasks, desc="Visualize", unit="plot"):
            tqdm.write(f"  绘制 {desc} ...")
            fn()

    print(f"\n{'=' * 60}")
    print(f"  完成！[{CLUSTERING_MODE}] 聚类标签已写入: {OUTPUT_FILE.name}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

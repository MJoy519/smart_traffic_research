"""
标签生成主流程

pipeline:
  1. 读取特征提取结果（获取每条路线、每个视频的窗口列表）
  2. 对每个窗口：聚合所有受试者的 V-A 坐标（中位数/均值）
  3. 与自我报告数据对比，验证标签质量
  4. 输出窗口级 V-A 连续值 CSV 和可视化

运行方式：
  cd label_generation_cc
  python label_generator.py
  或
  python -m label_generation_cc.label_generator  (从 Code_0321 根目录)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

import importlib.util as _ilu
_LCC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _LCC_DIR)

def _load_lcc_config():
    spec = _ilu.spec_from_file_location("lcc_config", os.path.join(_LCC_DIR, "config.py"))
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg = _load_lcc_config()
from russell_va_mapping import (
    get_participants,
    load_participant_video,
    compute_window_va,
    aggregate_window_va,
    project_to_va,
)

# ============================================================
#                       可控参数
# ============================================================

# 是否重新计算（False = 若结果已存在则跳过）
FORCE_RECOMPUTE   = True

# 是否生成可视化
SAVE_FIGURES      = cfg.SAVE_FIGURES
SHOW_FIGURES      = cfg.SHOW_FIGURES

# ============================================================
#                  特征窗口加载
# ============================================================

def load_feature_windows(route: int) -> pd.DataFrame:
    """
    加载指定路线的特征结果 CSV（使用复合特征文件获取窗口元数据）。

    Returns
    -------
    pd.DataFrame: route_num, video_name, window_idx, window_start_sec, window_end_sec
    """
    feat_dir  = cfg.FEATURE_DIR / cfg.FEATURE_DIR_LABEL
    feat_file = feat_dir / f'{route}_com_{cfg.WINDOW_SIZE_SEC}_{cfg.STEP_SIZE_SEC}.csv'

    if not feat_file.exists():
        warnings.warn(f"特征文件不存在: {feat_file}")
        return pd.DataFrame()

    df = pd.read_csv(feat_file)
    meta_cols = ['route_num', 'video_name', 'window_idx', 'window_start_sec', 'window_end_sec']
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"特征文件缺少列: {missing}")

    return df[meta_cols].drop_duplicates().reset_index(drop=True)


# ============================================================
#              核心：构建窗口级别的 V-A 标签
# ============================================================

def build_window_labels(route: int) -> pd.DataFrame:
    """
    对单条路线的所有视频窗口计算跨受试者聚合 V-A 坐标。

    Parameters
    ----------
    route : int  路线编号

    Returns
    -------
    pd.DataFrame 包含:
      route_num, video_name, window_idx, window_start_sec, window_end_sec,
      valence_median, arousal_median, valence_mean, arousal_mean,
      valence_std, arousal_std, valence_iqr, arousal_iqr, n_valid
    """
    df_windows = load_feature_windows(route)
    if df_windows.empty:
        return pd.DataFrame()

    participants = get_participants(route)
    if not participants:
        warnings.warn(f"Route {route} 未找到受试者目录")
        return pd.DataFrame()

    print(f"\n  Route {route} | 受试者: {participants} | 窗口数: {len(df_windows)}")

    # 缓存受试者数据（避免重复读取磁盘）
    participant_cache = {}

    records = []
    for _, row in tqdm(df_windows.iterrows(), total=len(df_windows),
                       desc=f"  Route {route}", unit="window",
                       bar_format="{l_bar}{bar:25}{r_bar}"):
        video_name   = row['video_name']
        window_idx   = int(row['window_idx'])
        window_start = float(row['window_start_sec'])
        window_end   = float(row['window_end_sec'])

        agg = aggregate_window_va(
            route, video_name, window_start, window_end,
            participant_cache=participant_cache,
        )

        records.append({
            'route_num':       route,
            'video_name':      video_name,
            'window_idx':      window_idx,
            'window_start_sec': window_start,
            'window_end_sec':  window_end,
            **{k: v for k, v in agg.items() if k != 'participant_va'},
        })

    return pd.DataFrame(records)


# ============================================================
#            自我报告对比验证
# ============================================================

def load_self_report(route: int):
    """加载指定路线的自我报告 V-A 数据。"""
    path = cfg.SELF_REPORT_FILE.get(route)
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def validate_with_self_report(df_labels: pd.DataFrame, route: int) -> dict:
    """
    计算生成的 Valence 标签与自我报告 Valence 的相关系数。

    自我报告数据精度为视频级别（不分窗口），
    因此取每个视频的所有窗口的 valence_median 均值后与自我报告对比。

    Returns
    -------
    dict: {video_name: {'our_v': ..., 'report_v': ..., 'diff': ...}}
    """
    df_sr = load_self_report(route)
    if df_sr is None:
        return {}

    results = {}
    df_route = df_labels[df_labels['route_num'] == route].copy()

    for _, sr_row in df_sr.iterrows():
        for col in df_sr.columns:
            if col.startswith('Valence'):
                try:
                    vid_num = int(col.replace('Valence', ''))
                except ValueError:
                    continue
                video_name = f'CUT {vid_num}'
                df_vid = df_route[df_route['video_name'] == video_name]
                if df_vid.empty:
                    continue
                our_v     = float(df_vid['valence_median'].mean())
                report_v  = float(sr_row[col])
                results[f"P{sr_row['Participant_id']}-{video_name}"] = {
                    'our_v':    our_v,
                    'report_v': report_v,
                    'diff':     our_v - report_v,
                }

    return results


# ============================================================
#                     可视化
# ============================================================

def _savefig(fig, filename: str):
    if SAVE_FIGURES:
        cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = cfg.FIGURES_DIR / filename
        fig.savefig(save_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
        print(f"  图片已保存: {save_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_va_scatter(df_labels: pd.DataFrame):
    """绘制所有窗口的 V-A 散点图（按路线着色）。"""
    df_plot = df_labels.dropna(subset=['valence_median', 'arousal_median'])
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    for route, grp in df_plot.groupby('route_num'):
        ax.scatter(grp['valence_median'], grp['arousal_median'],
                   label=f'Route {route}', alpha=0.5, s=15, edgecolors='none')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Valence (V)', fontsize=12)
    ax.set_ylabel('Arousal (A)', fontsize=12)
    ax.set_title('V-A Scatter (by Route)', fontsize=13)
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.suptitle('Russell V-A Space — Window V-A Values', fontsize=14, y=1.01)
    plt.tight_layout()
    _savefig(fig, 'va_scatter.png')



def plot_va_heatmap(df_labels: pd.DataFrame):
    """绘制 V-A 空间密度热力图。"""
    df_plot = df_labels.dropna(subset=['valence_median', 'arousal_median'])
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    h = ax.hist2d(
        df_plot['valence_median'], df_plot['arousal_median'],
        bins=30, cmap='YlOrRd', range=[[-1, 1], [-1, 1]]
    )
    plt.colorbar(h[3], ax=ax, label='Window Count')
    ax.axhline(0, color='white', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.axvline(0, color='white', linewidth=1.2, linestyle='--', alpha=0.7)
    ax.set_xlabel('Valence', fontsize=12)
    ax.set_ylabel('Arousal', fontsize=12)
    ax.set_title('V-A Space Density Heatmap (All Windows)', fontsize=13)

    # 添加 Russell 四象限标注
    for (x, y, text) in [
        (0.5,  0.5,  'Excited\n(HV-HA)'), (-0.5,  0.5,  'Stressed\n(LV-HA)'),
        (-0.5, -0.5, 'Depressed\n(LV-LA)'), (0.5, -0.5, 'Calm\n(HV-LA)'),
    ]:
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                color='white', alpha=0.8, style='italic')

    plt.tight_layout()
    _savefig(fig, 'va_heatmap.png')


def plot_video_va_timeline(df_labels: pd.DataFrame):
    """绘制各视频的 V-A 时序曲线（按路线分面）。"""
    routes = sorted(df_labels['route_num'].dropna().unique())

    for route in routes:
        df_r = df_labels[df_labels['route_num'] == route].copy()
        videos = sorted(df_r['video_name'].unique())
        n_vid  = len(videos)

        fig, axes = plt.subplots(n_vid, 1, figsize=(14, 3 * n_vid), sharex=False)
        if n_vid == 1:
            axes = [axes]

        for ax, vid in zip(axes, videos):
            df_v = df_r[df_r['video_name'] == vid].sort_values('window_start_sec')
            x = df_v['window_start_sec'].values

            ax.plot(x, df_v['valence_median'].values, 'b-o', markersize=3,
                    linewidth=1.5, label='Valence')
            ax.plot(x, df_v['arousal_median'].values, 'r--^', markersize=3,
                    linewidth=1.5, label='Arousal')
            ax.fill_between(x,
                            df_v['valence_median'] - df_v['valence_std'],
                            df_v['valence_median'] + df_v['valence_std'],
                            alpha=0.15, color='blue')
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
            ax.set_ylabel('V / A', fontsize=10)
            ax.set_title(f'Route {route} — {vid}', fontsize=11)
            ax.set_ylim(-1, 1)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel('Window Start (s)', fontsize=11)
        plt.suptitle(f'Route {route}: V-A Timeline per Video', fontsize=13, y=1.01)
        plt.tight_layout()
        _savefig(fig, f'va_timeline_route{route}.png')


def plot_russell_reference():
    """绘制罗素情感环参考图（含7种情绪标注位置）。"""
    fig, ax = plt.subplots(figsize=(8, 8))

    circle = plt.Circle((0, 0), 1.0, fill=False, color='lightgray',
                         linewidth=1.5, linestyle='--')
    ax.add_patch(circle)

    colors_emo = {
        'Joy': '#27AE60', 'Anger': '#E74C3C', 'Fear': '#C0392B',
        'Disgust': '#8E44AD', 'Contempt': '#E67E22', 'Sadness': '#2980B9',
        'Confusion': '#7F8C8D',
    }
    for emo, coords in cfg.RUSSELL_COORDINATES.items():
        v, a = coords['valence'], coords['arousal']
        ax.scatter(v, a, s=120, c=colors_emo.get(emo, 'gray'), zorder=5)
        ax.annotate(emo, (v, a), textcoords='offset points',
                    xytext=(8, 5), fontsize=11,
                    color=colors_emo.get(emo, 'gray'), fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('Valence (Negative ← → Positive)', fontsize=12)
    ax.set_ylabel('Arousal (Calm ← → Excited)', fontsize=12)
    ax.set_title("Russell's Circumplex Model — Emotion Coordinates Used\n"
                 "(Prior Knowledge for V-A Projection)", fontsize=13)
    ax.grid(True, alpha=0.2)

    for (x, y, lbl) in [(0.7, 0.7, 'Excited'), (-0.7, 0.7, 'Distressed'),
                         (-0.7, -0.7, 'Depressed'), (0.7, -0.7, 'Relaxed')]:
        ax.text(x, y, lbl, ha='center', fontsize=10, color='gray', style='italic')

    plt.tight_layout()
    _savefig(fig, 'russell_reference.png')


def plot_self_report_comparison(df_labels: pd.DataFrame):
    """绘制生成 Valence 与自我报告 Valence 的散点对比图。"""
    from scipy.stats import pearsonr, spearmanr

    all_our, all_report, route_labels = [], [], []

    for route in cfg.TARGET_ROUTES:
        df_sr_path = cfg.SELF_REPORT_FILE.get(route)
        if df_sr_path is None or not df_sr_path.exists():
            continue
        df_sr = pd.read_csv(df_sr_path)
        df_r  = df_labels[df_labels['route_num'] == route]

        for _, sr_row in df_sr.iterrows():
            for col in df_sr.columns:
                if not col.startswith('Valence'):
                    continue
                try:
                    vid_num = int(col.replace('Valence', ''))
                except ValueError:
                    continue
                video_name = f'CUT {vid_num}'
                df_vid = df_r[df_r['video_name'] == video_name]
                if df_vid.empty:
                    continue
                our_v    = float(df_vid['valence_median'].mean())
                report_v = float(sr_row[col])
                all_our.append(our_v)
                all_report.append(report_v)
                route_labels.append(route)

    if not all_our:
        print("  [INFO] 未找到自我报告数据，跳过对比可视化。")
        return

    r_pearson, p_pearson   = pearsonr(all_our, all_report)
    r_spearman, p_spearman = spearmanr(all_our, all_report)

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {1: '#3498DB', 2: '#E74C3C', 3: '#27AE60'}
    for route in sorted(set(route_labels)):
        idx = [i for i, r in enumerate(route_labels) if r == route]
        ax.scatter([all_our[i] for i in idx], [all_report[i] for i in idx],
                   c=colors.get(route, 'gray'), label=f'Route {route}', alpha=0.7, s=40)

    lims = [-1.2, 1.2]
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Our Generated Valence (Russell Projection)', fontsize=12)
    ax.set_ylabel('Self-Report Valence', fontsize=12)
    ax.set_title(
        f'Validation: Generated Valence vs Self-Report\n'
        f'Pearson r={r_pearson:.3f} (p={p_pearson:.3f})   '
        f'Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f})',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig(fig, 'self_report_validation.png')
    print(f"\n  [验证] Pearson r={r_pearson:.3f} (p={p_pearson:.3f}), "
          f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.3f})")


# ============================================================
#                         主流程
# ============================================================

def main():
    print(f"\n{'=' * 65}")
    print(f"  罗素情感环 V-A 映射")
    print(f"  窗口: {cfg.WINDOW_SIZE_SEC}s / 步长: {cfg.STEP_SIZE_SEC}s  |  "
          f"聚合方式: {cfg.PARTICIPANT_AGG}")
    print(f"  目标路线: {cfg.TARGET_ROUTES}")
    print(f"  情绪列: {cfg.EMOTION_COLUMNS}")
    print(f"{'=' * 65}")

    output_file = cfg.OUTPUT_DIR / 'window_labels.csv'
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------- 检查是否已有缓存 --------
    if output_file.exists() and not FORCE_RECOMPUTE:
        print(f"\n  [INFO] 已存在标签文件，直接加载: {output_file}")
        df_all = pd.read_csv(output_file)
    else:
        # -------- 逐路线生成窗口 V-A 坐标 --------
        all_dfs = []
        for route in cfg.TARGET_ROUTES:
            df_route = build_window_labels(route)
            if not df_route.empty:
                all_dfs.append(df_route)

        if not all_dfs:
            print("\n  [ERROR] 未生成任何窗口标签，请检查特征和 iMotion 数据路径。")
            return

        df_all = pd.concat(all_dfs, ignore_index=True)

        # -------- 保存 --------
        df_all.to_csv(output_file, index=False)
        print(f"\n  V-A 标签文件已保存: {output_file}")

    # -------- 统计摘要 --------
    print(f"\n  总窗口数: {len(df_all)}")
    print(f"  有效 V-A 窗口: {df_all['valence_median'].notna().sum()}")
    v_valid = df_all['valence_median'].dropna()
    a_valid = df_all['arousal_median'].dropna()
    print(f"  Valence: mean={v_valid.mean():.4f}  std={v_valid.std():.4f}"
          f"  range=[{v_valid.min():.3f}, {v_valid.max():.3f}]")
    print(f"  Arousal: mean={a_valid.mean():.4f}  std={a_valid.std():.4f}"
          f"  range=[{a_valid.min():.3f}, {a_valid.max():.3f}]")

    # -------- 可视化 --------
    if SAVE_FIGURES or SHOW_FIGURES:
        print(f"\n  生成可视化图表 → {cfg.FIGURES_DIR}")
        plot_russell_reference()
        plot_va_scatter(df_all)
        plot_va_heatmap(df_all)
        plot_video_va_timeline(df_all)
        plot_self_report_comparison(df_all)

    print(f"\n{'=' * 65}")
    print(f"  V-A 标签生成完成！结果保存至: {cfg.OUTPUT_DIR}")
    print(f"{'=' * 65}")
    return df_all


if __name__ == '__main__':
    main()

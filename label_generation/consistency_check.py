"""
窗口情绪标签与 Self-Report Valence 一致性检验脚本

假设：
  - 负面窗口比例越高  → Valence 越低（负相关）
  - 正面窗口比例越高  → Valence 越高（正相关）

统计方法：
  【A】被试内排序分析（主要方法）
      对每位被试，计算其各视频的 prop_negative/prop_positive 与 Valence
      的 Spearman ρ，再对所有被试的 ρ 做单样本 t 检验（H0: 均值=0）。
      该方法天然控制个体间情绪基线差异，适合情绪激活整体稀疏的驾驶场景。

  【B】跨被试全局分析（辅助方法）
      1. Spearman 等级相关：prop_positive/prop_negative 与 Valence
      2. Kruskal-Wallis 检验：不同主导标签组之间 Valence 是否有显著差异
      3. 线性回归：Valence ~ prop_positive + prop_negative
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import combinations
from scipy import stats
from scipy.stats import spearmanr, kruskal, mannwhitneyu, ttest_1samp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
#                       全局可控参数
# ============================================================
from config import WINDOW_SIZE, STEP_SIZE, EMOTION_TARGETS
# -- 窗口参数（需与 windowing.py / gmm_clustering.py 保持一致）--
WINDOW_SIZE_SEC = WINDOW_SIZE
STEP_SIZE_SEC   = STEP_SIZE

# -- 目标情绪列 --
EMOTION_COLUMNS = EMOTION_TARGETS

# -- 聚类模式（必须与 4. gmm_clustering.py 中的 CLUSTERING_MODE 保持一致）--
# 'three_class' : Positive / Neutral / Negative
# 'two_class'   : Negative-Active / Inactive
CLUSTERING_MODE = 'two_class'

# -- 处理线路选择（可选 1, 2, 3 的任意组合）--
TARGET_ROUTES = [1, 2, 3]

# -- 显著性水平 --
ALPHA = 0.05

# -- 可视化参数 --
SAVE_FIGURES = True
SHOW_FIGURES = False

# ============================================================
#              由 CLUSTERING_MODE 自动推导的配置
# ============================================================

if CLUSTERING_MODE == 'three_class':
    # 所有有效标签（按正→中→负排列）
    _ALL_LABELS   = ['Positive', 'Neutral', 'Negative']
    # label → 对应 prop 列名（列名中不含特殊字符）
    _PROP_COL_MAP = {
        'Positive': 'prop_positive',
        'Neutral':  'prop_neutral',
        'Negative': 'prop_negative',
    }
    # (prop_col, 描述, 预期方向符号, 短键名)
    _WITHIN_TESTS = [
        ('prop_negative', 'prop_negative vs valence', '< 0', 'neg'),
        ('prop_positive', 'prop_positive vs valence', '> 0', 'pos'),
    ]
    _ACTIVE_PROP  = 'prop_negative'    # 主关注的"激活"比例列
    _ACTIVE_LABEL = 'Negative'
else:  # two_class
    _ALL_LABELS   = ['Negative-Active', 'Inactive']
    _PROP_COL_MAP = {
        'Negative-Active': 'prop_negative_active',
        'Inactive':        'prop_inactive',
    }
    _WITHIN_TESTS = [
        ('prop_negative_active', 'prop_negative_active vs valence', '< 0', 'act'),
    ]
    _ACTIVE_PROP  = 'prop_negative_active'
    _ACTIVE_LABEL = 'Negative-Active'

# ============================================================
#                       数据路径
# ============================================================

BASE_DIR         = Path(__file__).resolve().parent.parent
RESULTS_DIR      = BASE_DIR / 'label_generation' / 'results'
SELF_REPORT_DIR  = BASE_DIR / 'self_report'
FIG_DIR          = BASE_DIR / 'label_generation' / 'figures' / 'consistency'

# GMM 聚类输出的窗口级标注文件（来自 gmm_clustering.py），随模式自动切换
_mode_suffix = '' if CLUSTERING_MODE == 'three_class' else '_2class'
LABELED_FILE        = RESULTS_DIR / (
    f'emotion_windows_ws{WINDOW_SIZE_SEC}_st{STEP_SIZE_SEC}_median_labeled{_mode_suffix}.csv'
)
OUTPUT_MERGED       = RESULTS_DIR / 'consistency_merged.csv'
OUTPUT_STATS        = RESULTS_DIR / 'consistency_stats.csv'
OUTPUT_WITHIN_STATS = RESULTS_DIR / 'consistency_within_participant.csv'

# ============================================================
#                       数据加载函数
# ============================================================


def load_self_report(route):
    """
    读取某线路的 self-report 文件，融合为
    (participant_id, video, valence) 长格式 DataFrame。
    """
    fpath = SELF_REPORT_DIR / f'self-va-processed-{route}.csv'
    if not fpath.exists():
        print(f"[WARN] Self-report 文件不存在: {fpath}")
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    records = []
    for _, row in df.iterrows():
        try:
            pid = int(row['Participant_id'])
        except (ValueError, KeyError):
            continue
        for col in df.columns:
            if col.startswith('Valence'):
                try:
                    vid = int(col.replace('Valence', ''))
                    val = float(row[col])
                    records.append({
                        'route':          route,
                        'participant_id': pid,
                        'video':          vid,
                        'valence':        val,
                    })
                except (ValueError, TypeError):
                    pass

    return pd.DataFrame(records)


def load_all_self_reports():
    """加载所有指定线路的 self-report 数据并拼接。"""
    dfs = []
    for route in tqdm(TARGET_ROUTES, desc="加载 Self-Report", unit="route"):
        df = load_self_report(route)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_labeled_windows():
    """加载 GMM 标注后的窗口数据集。"""
    if not LABELED_FILE.exists():
        print(f"[ERROR] 标注文件不存在: {LABELED_FILE}")
        print("  请先运行 4. gmm_clustering.py 生成标注结果。")
        return pd.DataFrame()
    return pd.read_csv(LABELED_FILE)


# ============================================================
#                       数据聚合函数
# ============================================================


def parse_participant_id(p):
    """将受试者编号统一转为整数，如 'P26' → 26。"""
    if isinstance(p, str):
        return int(p.strip().upper().replace('P', ''))
    return int(p)


def aggregate_window_labels(df_labeled):
    """
    按 (route, participant_id, video) 聚合窗口标签，
    计算各标签的数量与比例，并确定主导标签。
    列名由 _PROP_COL_MAP 决定，随 CLUSTERING_MODE 自动适配。
    """
    df = df_labeled.copy()
    df = df[df['emotion_label'].isin(_ALL_LABELS)].copy()
    df['participant_id'] = df['participant'].apply(parse_participant_id)

    grouped = df.groupby(['route', 'participant_id', 'video'])['emotion_label']
    agg     = grouped.value_counts().unstack(fill_value=0)

    # 补充缺失标签列
    for label in _ALL_LABELS:
        if label not in agg.columns:
            agg[label] = 0

    # label → n_xxx 列名（将 '-' 转为 '_'，加前缀 n_）
    n_col = {lbl: 'n_' + _PROP_COL_MAP[lbl].replace('prop_', '')
             for lbl in _ALL_LABELS}
    agg = agg.rename(columns=n_col)

    n_cols         = list(n_col.values())
    agg['n_total'] = agg[n_cols].sum(axis=1)

    # 比例列
    for lbl in _ALL_LABELS:
        nc = n_col[lbl]
        agg[_PROP_COL_MAP[lbl]] = agg[nc] / agg['n_total']

    # 主导标签（从 n_xxx 列名还原）
    n_to_label = {v: k for k, v in n_col.items()}
    agg['dominant_label'] = agg[n_cols].idxmax(axis=1).map(n_to_label)

    return agg.reset_index()


# ============================================================
#                       统计检验函数
# ============================================================


def spearman_test(x, y, label_x, label_y):
    """执行 Spearman 相关检验并返回结果字典。"""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_, y_ = x[mask], y[mask]
    if len(x_) < 5:
        return None
    r, p = spearmanr(x_, y_)
    return {
        'test':     'Spearman',
        'var_x':    label_x,
        'var_y':    label_y,
        'n':        len(x_),
        'statistic': r,
        'p_value':  p,
        'significant': p < ALPHA,
        'direction': 'positive' if r > 0 else 'negative',
    }


def kruskal_wallis_test(df_merged, group_col='dominant_label', value_col='valence'):
    """Kruskal-Wallis 检验：各主导标签组间 Valence 差异。"""
    groups = {}
    for label in _ALL_LABELS:
        sub = df_merged.loc[df_merged[group_col] == label, value_col].dropna().values
        if len(sub) >= 3:
            groups[label] = sub

    if len(groups) < 2:
        return None, {}

    stat, p = kruskal(*groups.values())
    return {
        'test':       'Kruskal-Wallis',
        'groups':     list(groups.keys()),
        'n_per_group': {k: len(v) for k, v in groups.items()},
        'statistic':  stat,
        'p_value':    p,
        'significant': p < ALPHA,
    }, groups


def mannwhitney_posthoc(groups):
    """
    Kruskal-Wallis 显著后进行事后 Mann-Whitney U 检验
    （Bonferroni 校正）。
    """
    pairs    = list(combinations(groups.keys(), 2))
    n_tests  = len(pairs)
    results  = []
    for a, b in pairs:
        stat, p = mannwhitneyu(groups[a], groups[b], alternative='two-sided')
        p_adj   = min(p * n_tests, 1.0)
        results.append({
            'group_a':    a,
            'group_b':    b,
            'U_statistic': stat,
            'p_raw':      p,
            'p_bonferroni': p_adj,
            'significant': p_adj < ALPHA,
        })
    return pd.DataFrame(results)


def linear_regression_test(df_merged):
    """
    线性回归：
      three_class → Valence ~ prop_positive + prop_negative
      two_class   → Valence ~ prop_negative_active
    """
    if CLUSTERING_MODE == 'three_class':
        pred_cols = ['prop_positive', 'prop_negative']
    else:
        pred_cols = ['prop_negative_active']

    needed = pred_cols + ['valence']
    sub    = df_merged[needed].dropna()
    if len(sub) < 10:
        return None

    X      = sub[pred_cols].values
    y      = sub['valence'].values
    model  = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2     = r2_score(y, y_pred)

    n, k   = len(y), len(pred_cols)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return None
    f_stat = ((ss_tot - ss_res) / k) / (ss_res / (n - k - 1))
    p_f    = 1 - stats.f.cdf(f_stat, k, n - k - 1)

    result = {
        'test':        'LinearRegression',
        'n':           n,
        'intercept':   model.intercept_,
        'R2':          r2,
        'F_statistic': f_stat,
        'p_F':         p_f,
        'significant': p_f < ALPHA,
    }
    for col, coef in zip(pred_cols, model.coef_):
        result[f'coef_{col}'] = coef
    return result


def within_participant_ranking_test(df_merged):
    """
    【A】被试内排序一致性分析

    对每位被试（在同一线路内），计算其所有视频的
    prop_negative / prop_positive 与 Valence 的 Spearman ρ。
    再对所有被试的 ρ 做单样本 t 检验（H0: 均值 = 0）。

    Parameters
    ----------
    df_merged : pd.DataFrame  含 route, participant_id, video,
                               prop_positive, prop_negative, valence

    Returns
    -------
    df_within : pd.DataFrame  每位被试的 ρ 结果
    summary   : dict          汇总统计（均值、t 检验等）
    """
    MIN_VIDEOS = 3   # 被试内至少有这么多视频才纳入分析

    prop_cols = [t[0] for t in _WITHIN_TESTS]
    records   = []

    for (route, pid), grp in df_merged.groupby(['route', 'participant_id']):
        sub = grp.dropna(subset=prop_cols + ['valence'])
        if len(sub) < MIN_VIDEOS:
            continue

        row = {'route': route, 'participant_id': pid, 'n_videos': len(sub)}
        for prop_col, _, _, key in _WITHIN_TESTS:
            rho, p   = spearmanr(sub[prop_col], sub['valence'])
            row[f'rho_{key}'] = rho
            row[f'p_{key}']   = p
            row[f'sig_{key}'] = p < ALPHA
        records.append(row)

    if not records:
        print("  [WARN] 被试内分析：没有满足最低视频数要求的被试。")
        return pd.DataFrame(), {}

    df_within = pd.DataFrame(records)

    # ---- 单样本 t 检验 ----
    summary = {}
    print(f"\n{'─' * 60}")
    print(f"  【A】被试内排序一致性分析  (min_videos={MIN_VIDEOS})")
    print(f"  纳入被试数: {len(df_within)}  | 线路: {sorted(df_within['route'].unique())}")
    print(f"{'─' * 40}")

    for _, desc, hypo_dir_sym, key in _WITHIN_TESTS:
        rho_col  = f'rho_{key}'
        sig_col  = f'sig_{key}'
        hypo_dir = f'预期 ρ {hypo_dir_sym}'

        if rho_col not in df_within.columns:
            continue
        rhos     = df_within[rho_col].dropna().values
        mean_rho = rhos.mean()
        std_rho  = rhos.std()
        n_sig    = df_within[sig_col].sum() if sig_col in df_within.columns else 0

        t_stat, p_two = ttest_1samp(rhos, popmean=0)
        p_one = p_two / 2 if (
            ('<' in hypo_dir_sym and mean_rho < 0) or
            ('>' in hypo_dir_sym and mean_rho > 0)
        ) else 1 - p_two / 2

        sig_mark = '**' if p_one < 0.01 else ('*' if p_one < ALPHA else 'ns')
        dir_ok   = '✓' if (
            ('<' in hypo_dir_sym and mean_rho < 0) or
            ('>' in hypo_dir_sym and mean_rho > 0)
        ) else '✗'

        print(f"  {desc}:")
        print(f"    ρ̄={mean_rho:+.3f}  SD={std_rho:.3f}  "
              f"t={t_stat:+.3f}  p(one-sided)={p_one:.4f}  {sig_mark}  "
              f"({hypo_dir} {dir_ok})")
        print(f"    个体显著比例: {n_sig}/{len(df_within)} "
              f"({n_sig/len(df_within)*100:.0f}%)")

        summary[key] = {
            'test':               'WithinParticipant_t',
            'variable':           desc,
            'n_subjects':         len(df_within),
            'mean_rho':           mean_rho,
            'std_rho':            std_rho,
            't_statistic':        t_stat,
            'p_two_sided':        p_two,
            'p_one_sided':        p_one,
            'significant':        p_one < ALPHA,
            'direction_correct':  dir_ok == '✓',
            'n_individually_sig': int(n_sig),
        }

    print(f"{'─' * 60}\n")
    return df_within, summary


def run_all_tests(df_merged):
    """【B】跨被试全局统计检验（辅助方法）。"""
    print(f"\n{'─' * 60}")
    print(f"  【B】跨被试全局分析  (α = {ALPHA})")
    print(f"{'─' * 60}")

    results = []

    # ---- 1. Spearman 相关 ----
    for route in [None] + sorted(df_merged['route'].unique().tolist()):
        if route is None:
            sub    = df_merged
            tag    = 'ALL'
        else:
            sub    = df_merged[df_merged['route'] == route]
            tag    = f'Route{route}'

        for prop_col, _desc, hypo_sym, _key in _WITHIN_TESTS:
            hypo_dir = '+' if '>' in hypo_sym else '-'
            res = spearman_test(
                sub[prop_col].values, sub['valence'].values,
                prop_col, f'valence[{tag}]'
            )
            if res:
                res['scope'] = tag
                results.append(res)
                sig_mark = '**' if res['p_value'] < 0.01 else ('*' if res['significant'] else 'ns')
                exp_ok   = '✓' if res['direction'] == ('positive' if hypo_dir == '+' else 'negative') else '✗'
                print(f"  Spearman  [{tag:8s}]  {prop_col:25s} vs valence: "
                      f"r={res['statistic']:+.3f}  p={res['p_value']:.4f}  {sig_mark}  "
                      f"(预期方向: {hypo_dir}{exp_ok})")

    # ---- 2. Kruskal-Wallis ----
    print(f"\n{'─' * 40}")
    kw_res, groups = kruskal_wallis_test(df_merged)
    if kw_res:
        sig_mark = '**' if kw_res['p_value'] < 0.01 else ('*' if kw_res['significant'] else 'ns')
        print(f"  Kruskal-Wallis  主导标签组 vs valence: "
              f"H={kw_res['statistic']:.3f}  p={kw_res['p_value']:.4f}  {sig_mark}")
        for label, arr in groups.items():
            print(f"    {label:10s}: n={len(arr):3d}  "
                  f"median={np.median(arr):+.3f}  mean={np.mean(arr):+.3f}")
        results.append(kw_res)

        # 事后检验
        if kw_res['significant'] and len(groups) >= 2:
            print(f"\n  事后 Mann-Whitney U（Bonferroni 校正）:")
            posthoc = mannwhitney_posthoc(groups)
            for _, row in posthoc.iterrows():
                sig_mark = '*' if row['significant'] else 'ns'
                print(f"    {row['group_a']:10s} vs {row['group_b']:10s}: "
                      f"U={row['U_statistic']:.1f}  "
                      f"p_adj={row['p_bonferroni']:.4f}  {sig_mark}")

    # ---- 3. 线性回归 ----
    print(f"\n{'─' * 40}")
    lr_res = linear_regression_test(df_merged)
    if lr_res:
        pred_cols   = [t[0] for t in _WITHIN_TESTS]
        formula_str = ' + '.join(pred_cols)
        sig_mark    = '**' if lr_res['p_F'] < 0.01 else ('*' if lr_res['significant'] else 'ns')
        print(f"  线性回归  Valence ~ {formula_str}:")
        print(f"    n={lr_res['n']}  R²={lr_res['R2']:.4f}  "
              f"F={lr_res['F_statistic']:.3f}  p={lr_res['p_F']:.4f}  {sig_mark}")
        coef_strs = '  '.join(
            f"coef({c})={lr_res.get(f'coef_{c}', float('nan')):+.4f}"
            for c in pred_cols
        )
        print(f"    {coef_strs}")
        results.append(lr_res)

    print(f"{'─' * 60}\n")
    return results


# ============================================================
#                       可视化函数
# ============================================================

COLOR_MAP = {
    'Positive':        '#27AE60',
    'Neutral':         '#3498DB',
    'Negative':        '#E74C3C',
    'Negative-Active': '#C0392B',
    'Inactive':        '#95A5A6',
}


def _save_or_show(fig, filename):
    if SAVE_FIGURES:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIG_DIR / filename
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图片已保存: {save_path}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_scatter_correlation(df_merged):
    """散点图：各激活比例 vs Valence（分线路着色）。"""
    n_plots = len(_WITHIN_TESTS)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    routes  = sorted(df_merged['route'].dropna().unique())
    palette = plt.cm.tab10(np.linspace(0, 0.6, len(routes)))

    for ax, (prop_col, desc, hypo_sym, _key) in zip(axes, _WITHIN_TESTS):
        hypo_str = f'Expected: r {">" if ">" in hypo_sym else "<"} 0'
        for route, color in zip(routes, palette):
            sub = df_merged[df_merged['route'] == route]
            ax.scatter(sub[prop_col], sub['valence'],
                       color=color, alpha=0.55, s=18, label=f'Route {route}')

        x_all = df_merged[prop_col].dropna().values
        y_all = df_merged.loc[df_merged[prop_col].notna(), 'valence'].values
        if len(x_all) > 5:
            z  = np.polyfit(x_all, y_all, 1)
            xr = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(xr, np.polyval(z, xr),
                    color='black', linewidth=1.5, linestyle='--', label='Trend')
            r, p = spearmanr(x_all, y_all)
            ax.set_title(f'{desc}\n(Spearman r={r:+.3f}, p={p:.4f})', fontsize=11)
        else:
            ax.set_title(desc, fontsize=11)

        ax.set_xlabel(prop_col, fontsize=11)
        ax.set_ylabel('Self-Report Valence', fontsize=11)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.text(0.97, 0.03, hypo_str, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9, color='gray')
        ax.legend(fontsize=8, markerscale=1.5)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Consistency: Window Label Proportions vs Self-Report Valence',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, 'consistency_scatter.png')


def plot_boxplot_by_dominant(df_merged):
    """箱线图：各主导标签组的 Valence 分布。"""
    order   = _ALL_LABELS
    present = [l for l in order if l in df_merged['dominant_label'].values]

    fig, axes = plt.subplots(1, len(TARGET_ROUTES) + 1,
                             figsize=(5 * (len(TARGET_ROUTES) + 1), 5),
                             sharey=True)

    def _draw_box(ax, data, title):
        groups = {l: data.loc[data['dominant_label'] == l, 'valence'].dropna().values
                  for l in present}
        # 只保留该子集中实际有数据的标签，避免空数组导致 boxplot 报错
        valid_labels = [l for l in present if len(groups[l]) > 0]
        if not valid_labels:
            ax.set_title(f'{title}\n(无数据)', fontsize=11)
            return
        positions = list(range(len(valid_labels)))
        bp = ax.boxplot([groups[l] for l in valid_labels],
                        positions=positions,
                        widths=0.5,
                        patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch, label in zip(bp['boxes'], valid_labels):
            patch.set_facecolor(COLOR_MAP[label])
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(valid_labels, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Self-Report Valence', fontsize=10)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.grid(True, axis='y', alpha=0.25)
        # 标注中位数
        for i, l in enumerate(valid_labels):
            ax.text(i, np.median(groups[l]) + 0.02,
                    f'{np.median(groups[l]):.2f}',
                    ha='center', va='bottom', fontsize=8)

    for i, route in enumerate(sorted(df_merged['route'].unique())):
        sub = df_merged[df_merged['route'] == route]
        _draw_box(axes[i], sub, f'Route {route} (n={len(sub)})')

    _draw_box(axes[-1], df_merged, f'All Routes (n={len(df_merged)})')

    plt.suptitle('Valence Distribution by Dominant Emotion Label',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, 'consistency_boxplot.png')


def plot_heatmap_proportion_valence(df_merged):
    """将 Valence 按各激活比例区间分组后绘制条形热图。"""
    prop_cols = [t[0] for t in _WITHIN_TESTS]
    needed    = prop_cols + ['valence']
    df        = df_merged.copy().dropna(subset=needed)

    bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    blbls  = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    fig, axes = plt.subplots(1, len(prop_cols), figsize=(7 * len(prop_cols), 4))
    if len(prop_cols) == 1:
        axes = [axes]

    bar_colors = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#27AE60']

    for ax, prop_col in zip(axes, prop_cols):
        df['_bin'] = pd.cut(df[prop_col], bins=bins, labels=blbls, include_lowest=True)
        pivot = df.pivot_table(values='valence', index='_bin',
                               aggfunc='mean').reindex(blbls)
        if pivot.empty or pivot['valence'].isna().all():
            ax.set_visible(False)
            continue
        bars = ax.bar(range(len(pivot)), pivot['valence'].values,
                      color=bar_colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(pivot.index, fontsize=9)
        ax.set_xlabel('Window Proportion Bin', fontsize=10)
        ax.set_ylabel('Mean Self-Report Valence', fontsize=10)
        ax.set_title(f'Mean Valence by {prop_col}', fontsize=11)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
        ax.grid(True, axis='y', alpha=0.25)
        for bar, val in zip(bars, pivot['valence'].values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + (0.01 if val >= 0 else -0.03),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Mean Valence Across Window Label Proportion Bins', fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, 'consistency_heatmap.png')


def plot_route_summary(df_merged):
    """每条线路：正面/负面比例与 Valence 的散点矩阵。"""
    routes = sorted(df_merged['route'].dropna().unique())
    n = len(routes)
    if n == 0:
        return

    n_cols = len(_WITHIN_TESTS)
    fig, axes = plt.subplots(n, n_cols, figsize=(6 * n_cols, 4 * n))
    if n == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    prop_colors = ['#27AE60', '#E74C3C', '#3498DB', '#C0392B', '#95A5A6']

    for i, route in enumerate(routes):
        prop_cols = [t[0] for t in _WITHIN_TESTS]
        sub = df_merged[df_merged['route'] == route].dropna(
            subset=prop_cols + ['valence'])

        for j, (prop_col, desc, _sym, _key) in enumerate(_WITHIN_TESTS):
            ax    = axes[i, j]
            color = prop_colors[j % len(prop_colors)]
            ax.scatter(sub[prop_col], sub['valence'],
                       color=color, alpha=0.6, s=20)
            if len(sub) > 5:
                z  = np.polyfit(sub[prop_col], sub['valence'], 1)
                xr = np.linspace(sub[prop_col].min(), sub[prop_col].max(), 100)
                ax.plot(xr, np.polyval(z, xr), 'k--', linewidth=1.2)
                r, p = spearmanr(sub[prop_col], sub['valence'])
                ax.set_title(
                    f'Route {route} | {prop_col} vs Valence\n'
                    f'r={r:+.3f}  p={p:.4f}  n={len(sub)}',
                    fontsize=10)
            else:
                ax.set_title(f'Route {route} | {prop_col} vs Valence  (n={len(sub)})',
                             fontsize=10)
            ax.set_xlabel(prop_col, fontsize=9)
            ax.set_ylabel('Valence', fontsize=9)
            ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
            ax.grid(True, alpha=0.2)

    plt.suptitle('Per-Route Consistency: Window Labels vs Self-Report Valence',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, 'consistency_per_route.png')


def plot_within_participant(df_within, summary):
    """
    被试内排序分析可视化：
      上图：每位被试的 ρ 柱状图（按线路着色）
      下图：ρ 分布直方图 + 均值与置信区间
    """
    if df_within.empty:
        return

    routes  = sorted(df_within['route'].unique())
    palette = {r: c for r, c in zip(routes,
               plt.cm.tab10(np.linspace(0, 0.6, max(len(routes), 1))))}

    # 按 _WITHIN_TESTS 构造列配置
    _PLOT_CFG = []
    _colors   = [('#E74C3C', '#C0392B'), ('#27AE60', '#1E8449'),
                 ('#3498DB', '#2980B9'), ('#C0392B', '#922B21')]
    for idx, (prop_col, desc, hypo_sym, key) in enumerate(_WITHIN_TESTS):
        c1, c2 = _colors[idx % len(_colors)]
        _PLOT_CFG.append((
            f'rho_{key}', f'sig_{key}',
            f'ρ({prop_col}, valence)\n预期 {hypo_sym}',
            c1, c2, key
        ))

    n_tests = len(_PLOT_CFG)
    fig, axes = plt.subplots(2, n_tests, figsize=(7 * n_tests, 9))
    fig.suptitle('Within-Participant Ranking Consistency Analysis', fontsize=14)
    if n_tests == 1:
        axes = axes.reshape(2, 1)

    for col_idx, (rho_col, sig_col, label, color_pos, color_neg, sum_key) in enumerate(_PLOT_CFG):
        rhos = df_within[rho_col].values

        # ---- 上图：个体 ρ 柱状图 ----
        ax_bar = axes[0, col_idx]
        df_sorted = df_within.sort_values(['route', rho_col]).reset_index(drop=True)

        bar_colors = []
        for _, row in df_sorted.iterrows():
            base = palette[row['route']]
            bar_colors.append(base)

        bars = ax_bar.bar(range(len(df_sorted)), df_sorted[rho_col],
                          color=bar_colors, alpha=0.75, edgecolor='white')

        # 显著的被试加深颜色并打星
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            if row[sig_col]:
                bars[i].set_alpha(1.0)
                ax_bar.text(i, row[rho_col] + (0.02 if row[rho_col] >= 0 else -0.06),
                            '*', ha='center', va='bottom', fontsize=10,
                            color='black', fontweight='bold')

        ax_bar.axhline(0, color='gray', linestyle='--', linewidth=1.0)
        if sum_key in summary:
            mean_rho = summary[sum_key]['mean_rho']
            ax_bar.axhline(mean_rho, color=color_pos, linestyle='-',
                           linewidth=2.0, label=f'Mean ρ={mean_rho:+.3f}')
            ax_bar.legend(fontsize=9)

        ax_bar.set_xlabel('Participant (sorted by ρ)', fontsize=10)
        ax_bar.set_ylabel('Spearman ρ', fontsize=10)
        ax_bar.set_title(label, fontsize=11)
        ax_bar.set_xticks(range(len(df_sorted)))
        ax_bar.set_xticklabels(
            [f"P{int(r['participant_id'])}" for _, r in df_sorted.iterrows()],
            rotation=60, fontsize=7)
        ax_bar.set_ylim(-1.1, 1.1)
        ax_bar.grid(True, axis='y', alpha=0.2)

        # 添加线路图例
        for route in routes:
            ax_bar.bar([], [], color=palette[route], label=f'Route {route}', alpha=0.75)
        ax_bar.legend(fontsize=8, loc='upper left')

        # ---- 下图：ρ 分布直方图 + 均值 CI ----
        ax_hist = axes[1, col_idx]
        ax_hist.hist(rhos, bins=10, color=color_pos, alpha=0.6,
                     edgecolor='white', density=False)
        ax_hist.axvline(0, color='gray', linestyle='--', linewidth=1.0, label='ρ=0')

        if sum_key in summary:
            s = summary[sum_key]
            mean_r = s['mean_rho']
            se = s['std_rho'] / np.sqrt(s['n_subjects'])
            ci95 = 1.96 * se
            ax_hist.axvline(mean_r, color=color_neg, linewidth=2.0,
                            label=f"Mean={mean_r:+.3f}")
            ax_hist.axvspan(mean_r - ci95, mean_r + ci95,
                            color=color_neg, alpha=0.15, label='95% CI')
            sig_str = ('**' if s['p_one_sided'] < 0.01
                       else ('*' if s['p_one_sided'] < ALPHA else 'ns'))
            ax_hist.set_title(
                f't({s["n_subjects"]-1})={s["t_statistic"]:+.3f}  '
                f'p(one)={s["p_one_sided"]:.4f}  {sig_str}',
                fontsize=10)
        ax_hist.set_xlabel('Spearman ρ', fontsize=10)
        ax_hist.set_ylabel('Count', fontsize=10)
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.2)

    plt.tight_layout()
    _save_or_show(fig, 'consistency_within_participant.png')


# ============================================================
#                        主入口
# ============================================================

def main():
    mode_desc = ('三分类: Positive / Neutral / Negative'
                 if CLUSTERING_MODE == 'three_class'
                 else '二分类: Negative-Active / Inactive')
    print(f"\n{'=' * 60}")
    print(f"  窗口情绪标签 × Self-Report Valence 一致性检验")
    print(f"  聚类模式: {CLUSTERING_MODE}  ——  {mode_desc}")
    print(f"  输入文件: {LABELED_FILE.name}")
    print(f"  窗口参数: ws={WINDOW_SIZE_SEC}s  step={STEP_SIZE_SEC}s")
    print(f"  线路: {TARGET_ROUTES}  |  α={ALPHA}")
    print(f"{'=' * 60}\n")

    # ---- 1. 加载数据 ----
    print(">> 加载 GMM 标注窗口数据 ...")
    df_labeled = load_labeled_windows()
    if df_labeled.empty:
        return

    print(">> 加载 Self-Report 数据 ...")
    df_self = load_all_self_reports()
    if df_self.empty:
        print("[ERROR] 未能加载任何 Self-Report 数据。")
        return

    print(f"   窗口总数: {len(df_labeled)}  |  Self-Report 条目: {len(df_self)}")

    # ---- 2. 聚合窗口标签 ----
    print("\n>> 聚合窗口标签（per video）...")
    df_agg = aggregate_window_labels(df_labeled)
    print(f"   聚合后 (route, participant, video) 条目: {len(df_agg)}")

    # ---- 3. 合并 Self-Report ----
    print("\n>> 合并 Self-Report Valence ...")
    df_merged = df_agg.merge(
        df_self,
        on=['route', 'participant_id', 'video'],
        how='inner',
    )
    print(f"   成功匹配条目: {len(df_merged)}")

    if df_merged.empty:
        print("[WARN] 合并结果为空，请检查数据中参与者 ID、线路、视频编号是否对应。")
        return

    # 线路过滤
    if TARGET_ROUTES:
        df_merged = df_merged[df_merged['route'].isin(TARGET_ROUTES)].copy()

    # 保存合并数据
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(OUTPUT_MERGED, index=False)
    print(f"   合并数据已保存: {OUTPUT_MERGED}")

    # 打印各线路样本量
    print("\n  各线路匹配样本量:")
    for route in sorted(df_merged['route'].unique()):
        sub = df_merged[df_merged['route'] == route]
        print(f"    Route {route}: {len(sub)} 条  "
              f"(Valence 均值={sub['valence'].mean():.3f}, "
              f"std={sub['valence'].std():.3f})")

    # ---- 4a. 被试内排序分析（主要方法）----
    print("\n>> 【A】执行被试内排序一致性分析 ...")
    df_within, within_summary = within_participant_ranking_test(df_merged)

    # 保存被试内分析结果
    if not df_within.empty:
        df_within.to_csv(OUTPUT_WITHIN_STATS, index=False)
        print(f"  被试内分析结果已保存: {OUTPUT_WITHIN_STATS}")

    # ---- 4b. 跨被试全局分析（辅助方法）----
    print("\n>> 【B】执行跨被试全局统计检验 ...")
    test_results = run_all_tests(df_merged)

    # 保存全局统计结果
    rows = []
    for r in test_results:
        if isinstance(r, dict):
            rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_STATS, index=False)
        print(f"  全局统计结果已保存: {OUTPUT_STATS}")

    # ---- 5. 可视化 ----
    if SAVE_FIGURES or SHOW_FIGURES:
        print("\n>> 生成可视化图表 ...")
        vis_tasks = [
            ("被试内排序分析图", lambda: plot_within_participant(df_within, within_summary)),
            ("相关散点图",       lambda: plot_scatter_correlation(df_merged)),
            ("主导标签箱线图",   lambda: plot_boxplot_by_dominant(df_merged)),
            ("比例-Valence热图", lambda: plot_heatmap_proportion_valence(df_merged)),
            ("分线路散点图",     lambda: plot_route_summary(df_merged)),
        ]
        for desc, fn in tqdm(vis_tasks, desc="Visualize", unit="plot"):
            tqdm.write(f"  绘制 {desc} ...")
            fn()

    print(f"\n{'=' * 60}")
    print(f"  完成！结果保存至: {RESULTS_DIR}")
    print(f"  图表保存至:       {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

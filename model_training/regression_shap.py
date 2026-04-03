"""
回归模型 SHAP 特征重要性分析 + 敏感性定量分析模块

核心功能：
  1. SHAP 全局重要性（mean |SHAP|）
  2. SHAP Beeswarm 图（特征值方向性）
  3. 偏依赖图（PDP）—— 反映单特征对 V/A 的非线性效应
  4. 敏感性定量表（Sensitivity Table）—— 科学计算各特征变化对 V/A 的量化影响：
       ∂ŷ/∂x   : SHAP 线性斜率（每单位特征变化对应的预测变化量）
       Δŷ/σ    : 标准化效应（每 1 个标准差变化对应的预测变化量）
       PDP 范围 : 该特征从 P10 到 P90 的预测变化区间
  5. V-A 双目标特征影响对比图
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import linregress

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ============================================================
#                   辅助工具
# ============================================================

def _savefig(fig: plt.Figure, fname: str):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / fname
    fig.savefig(path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    print(f"  图片已保存: {path}")
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def _get_shap_values_regression(model, X: np.ndarray) -> np.ndarray:
    """
    对回归模型计算 SHAP 值。
    回归模型 SHAP 输出为 2D array (n_samples, n_features)，无需 list 处理。
    """
    import shap
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # 某些版本返回 list，取第一个
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    # 3D → 2D（极少数情况）
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    return shap_values, explainer.expected_value


# ============================================================
#          1. SHAP 全局重要性
# ============================================================

def plot_shap_importance_regression(
    shap_values: np.ndarray,
    feature_cols: List[str],
    model_name: str,
    target_name: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    绘制回归 SHAP 全局重要性柱状图，返回重要性 DataFrame。
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    std_abs  = np.abs(shap_values).std(axis=0)

    df = pd.DataFrame({
        'feature':          feature_cols,
        'importance_mean':  mean_abs,
        'importance_std':   std_abs,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    top = df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top)))
    bars   = ax.barh(range(len(top)), top['importance_mean'].values[::-1],
                     xerr=top['importance_std'].values[::-1],
                     color=colors[::-1], edgecolor='white', linewidth=0.5,
                     error_kw={'elinewidth': 0.8, 'capsize': 2})
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=9)
    ax.set_xlabel('mean |SHAP value|', fontsize=10)
    target_label = 'Valence' if target_name == 'valence' else 'Arousal'
    ax.set_title(
        f'{model_name.replace("_", " ").title()} — SHAP Feature Importance\n'
        f'Target: {target_label} (Regression)',
        fontsize=11,
    )
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    _savefig(fig, f'reg_shap_importance_{target_name}_{model_name}.png')
    return df


# ============================================================
#          2. SHAP Beeswarm
# ============================================================

def plot_shap_beeswarm_regression(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_cols: List[str],
    model_name: str,
    target_name: str,
    top_n: int = 15,
):
    """Beeswarm：特征值高低与 SHAP 方向，揭示正/负效应方向。"""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]

    sv_top   = shap_values[:, top_idx]
    X_top    = X_explain[:, top_idx]
    feats    = [feature_cols[i] for i in top_idx]

    # 翻转使最重要的在顶部
    feats_rev = feats[::-1]
    sv_rev    = sv_top[:, ::-1]
    X_rev     = X_top[:, ::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
    cmap = plt.cm.RdBu_r

    for i, fname in enumerate(feats_rev):
        sv   = sv_rev[:, i]
        xval = X_rev[:, i]
        norm = (xval - xval.min()) / (xval.max() - xval.min() + 1e-9)
        colors = cmap(norm)
        y = np.full(len(sv), i) + np.random.uniform(-0.3, 0.3, len(sv))
        ax.scatter(sv, y, c=colors, alpha=0.4, s=10, edgecolors='none')

    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats_rev, fontsize=9)
    ax.set_xlabel('SHAP Value  (↑ increases prediction, ↓ decreases)', fontsize=10)
    target_label = 'Valence' if target_name == 'valence' else 'Arousal'
    ax.set_title(
        f'{model_name.replace("_", " ").title()} — SHAP Beeswarm\n'
        f'Target: {target_label}  |  Red = high feature value, Blue = low',
        fontsize=11,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Feature Value (normalized)', fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    _savefig(fig, f'reg_shap_beeswarm_{target_name}_{model_name}.png')


# ============================================================
#          3. 偏依赖图（PDP）
# ============================================================

def compute_pdp(
    model,
    X: np.ndarray,
    feature_idx: int,
    n_grid: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单特征的偏依赖（Partial Dependence）。
    在 feature_idx 对应的特征上建立等间距网格，
    每个网格点将该特征替换到所有样本，取预测均值。

    Returns
    -------
    grid_vals   : 特征值网格
    pdp_vals    : 各网格点对应的平均预测值
    """
    X_copy = X.copy()
    col    = X[:, feature_idx]
    grid   = np.linspace(np.percentile(col, 5), np.percentile(col, 95), n_grid)
    pdp    = np.zeros(n_grid)

    for k, val in enumerate(grid):
        X_copy[:, feature_idx] = val
        pdp[k] = model.predict(X_copy).mean()

    return grid, pdp


def plot_pdp_grid(
    model,
    X: np.ndarray,
    feature_cols: List[str],
    importance_df: pd.DataFrame,
    model_name: str,
    target_name: str,
    top_n: int = 9,
    n_grid: int = 50,
):
    """
    绘制 top_n 个特征的 PDP 子图网格。
    每个子图包含：PDP 曲线 + 特征值 rug plot + P10/P90 区间标注。
    """
    top_feats = importance_df.head(top_n)['feature'].tolist()
    n_cols    = 3
    n_rows    = (top_n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    target_label = 'Valence' if target_name == 'valence' else 'Arousal'

    for k, feat in enumerate(top_feats):
        if feat not in feature_cols:
            continue
        feat_idx  = feature_cols.index(feat)
        grid, pdp = compute_pdp(model, X, feat_idx, n_grid)

        ax = axes_flat[k]
        ax.plot(grid, pdp, color='#1565C0', linewidth=2)
        ax.fill_between(grid, pdp, alpha=0.15, color='#1565C0')

        # Rug plot（特征真实分布）
        actual_col = X[:, feat_idx]
        ax.plot(np.clip(actual_col, grid.min(), grid.max()),
                np.full(len(actual_col), pdp.min() - 0.002),
                '|', color='gray', alpha=0.3, markersize=4)

        # P10/P90 参考线
        p10 = np.percentile(actual_col, 10)
        p90 = np.percentile(actual_col, 90)
        ax.axvline(p10, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='P10/P90')
        ax.axvline(p90, color='orange', linestyle=':', linewidth=1, alpha=0.7)

        # 计算 P10→P90 效应
        g_p10 = np.linspace(np.percentile(actual_col, 5), np.percentile(actual_col, 95), n_grid)
        idx10 = np.argmin(np.abs(g_p10 - p10))
        idx90 = np.argmin(np.abs(g_p10 - p90))
        delta = float(pdp[min(idx90, n_grid-1)] - pdp[min(idx10, n_grid-1)])

        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(f'Predicted {target_label}', fontsize=8)
        ax.set_title(f'{feat}\nΔ(P10→P90) = {delta:+.4f}', fontsize=8)
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # 隐藏多余子图
    for k in range(len(top_feats), len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.suptitle(
        f'Partial Dependence Plots — {target_label} Regression\n'
        f'Model: {model_name.replace("_", " ").title()}  |  '
        f'Average effect of each feature (others fixed at median)',
        fontsize=11,
    )
    plt.tight_layout()
    _savefig(fig, f'reg_pdp_{target_name}_{model_name}.png')


# ============================================================
#          4. 敏感性定量分析
# ============================================================

def compute_sensitivity_table(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_cols: List[str],
    model,
    target_name: str,
    top_n: int = 15,
    n_grid: int = 50,
) -> pd.DataFrame:
    """
    科学定量计算各特征变化对预测 Valence/Arousal 的影响。

    方法：
      1. SHAP 线性斜率 (β_shap):
           对每个特征，用线性回归拟合 shap_i ~ x_i，
           斜率 β_shap 即为 ∂ŷ/∂x（每单位特征变化对应的预测变化量）
      2. 标准化效应 (Δŷ/σ):
           β_shap × std(x_i)，即每 1 个标准差变化对应的预测变化量
      3. P10→P90 PDP 效应 (pdp_delta):
           将特征从 P10 移动到 P90，通过偏依赖计算平均预测变化量，
           反映特征在实际分布范围内的最大效应
      4. 单位效应示例:
           以特征标准差的 0.1 倍为单位，给出具体变化量描述

    Returns
    -------
    pd.DataFrame 含完整敏感性指标，按 |Δŷ/σ| 降序排列
    """
    mean_abs   = np.abs(shap_values).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]
    rows       = []

    for feat_idx in top_idx:
        feat     = feature_cols[feat_idx]
        sv_col   = shap_values[:, feat_idx]   # SHAP 值
        x_col    = X_explain[:, feat_idx]      # 特征值

        feat_std = float(np.std(x_col))
        feat_mean= float(np.mean(x_col))
        p10      = float(np.percentile(x_col, 10))
        p90      = float(np.percentile(x_col, 90))

        # ── 1. SHAP 线性斜率 ──────────────────────────────────
        # shap_i = α + β_shap × x_i  →  β_shap ≈ ∂ŷ/∂x
        if feat_std > 1e-9:
            slope, intercept, r_val, p_val, se = linregress(x_col, sv_col)
            beta_shap = float(slope)
            r_shap    = float(r_val)
        else:
            beta_shap = 0.0
            r_shap    = 0.0

        # ── 2. 标准化效应：Δŷ per 1σ ──────────────────────────
        delta_per_sigma = beta_shap * feat_std

        # ── 3. PDP P10→P90 效应 ───────────────────────────────
        grid, pdp = compute_pdp(model, X_explain, feat_idx, n_grid)
        # 在 grid 中找最近的 P10、P90
        idx10 = int(np.argmin(np.abs(grid - p10)))
        idx90 = int(np.argmin(np.abs(grid - p90)))
        pdp_at_p10 = float(pdp[idx10])
        pdp_at_p90 = float(pdp[idx90])
        pdp_delta  = float(pdp_at_p90 - pdp_at_p10)

        # ── 4. 单位效应描述（以 0.1×原始范围 为单位）────────────
        feat_range     = p90 - p10
        unit_change    = feat_range * 0.1 if feat_range > 1e-9 else feat_std * 0.1
        delta_per_unit = beta_shap * unit_change

        rows.append({
            'feature':           feat,
            'target':            target_name,
            'shap_importance':   float(mean_abs[feat_idx]),
            'feat_mean':         feat_mean,
            'feat_std':          feat_std,
            'feat_p10':          p10,
            'feat_p90':          p90,
            'beta_shap':         beta_shap,       # ∂ŷ/∂x（每单位变化）
            'delta_per_sigma':   delta_per_sigma, # Δŷ/σ（标准化效应）
            'pdp_at_p10':        pdp_at_p10,
            'pdp_at_p90':        pdp_at_p90,
            'pdp_delta_p10_p90': pdp_delta,       # P10→P90 PDP 效应
            'unit_change_ref':   unit_change,
            'delta_per_unit':    delta_per_unit,
            'shap_r':            r_shap,          # SHAP~feature 线性拟合 R
        })

    df = pd.DataFrame(rows)
    df = df.reindex(df['delta_per_sigma'].abs().sort_values(ascending=False).index)
    df['sens_rank'] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def plot_sensitivity_bar(
    sens_v: pd.DataFrame,
    sens_a: pd.DataFrame,
    top_n: int = 12,
):
    """
    双目标敏感性对比柱状图：
      左子图：每 1σ 变化对应的 Valence 预测变化量 (Δŷ/σ)
      右子图：每 1σ 变化对应的 Arousal 预测变化量 (Δŷ/σ)
    颜色：正值（红，升高）/ 负值（蓝，降低）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.45)))

    for ax, sens_df, target_label in zip(
        axes,
        [sens_v, sens_a],
        ['Valence', 'Arousal'],
    ):
        top = sens_df.nlargest(top_n, 'shap_importance')
        top = top.sort_values('delta_per_sigma')

        colors = ['#C62828' if v > 0 else '#1565C0' for v in top['delta_per_sigma']]
        y_pos  = range(len(top))

        ax.barh(y_pos, top['delta_per_sigma'], color=colors, alpha=0.8, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top['feature'], fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)

        for i, (val, row) in enumerate(zip(top['delta_per_sigma'], top.itertuples())):
            ha  = 'left' if val > 0 else 'right'
            off = 0.0005 if val > 0 else -0.0005
            ax.text(val + off, i, f'{val:+.4f}', va='center', ha=ha, fontsize=8)

        ax.set_xlabel(f'Δ {target_label} per 1σ increase in feature', fontsize=10)
        ax.set_title(
            f'Feature Sensitivity — {target_label}\n'
            f'(Red=increases {target_label}, Blue=decreases {target_label})',
            fontsize=10,
        )
        ax.grid(True, axis='x', alpha=0.3)

    plt.suptitle(
        'Standardized Sensitivity Analysis: Δ Emotion per 1σ Feature Change',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    _savefig(fig, 'reg_sensitivity_comparison.png')


def plot_pdp_delta_heatmap(
    sens_v: pd.DataFrame,
    sens_a: pd.DataFrame,
    top_n: int = 15,
):
    """
    V-A 双目标 PDP 效应热力图：
    行=特征，列=V/A，值=PDP P10→P90 效应（颜色反映方向和大小）
    """
    # 取两者共同的 top 特征（按 V 的重要性排序）
    top_feats = sens_v.nlargest(top_n, 'shap_importance')['feature'].tolist()

    df_v = sens_v.set_index('feature')['pdp_delta_p10_p90'].reindex(top_feats)
    df_a = sens_a.set_index('feature')['pdp_delta_p10_p90'].reindex(top_feats)

    heatmap_data = pd.DataFrame({'Valence': df_v, 'Arousal': df_a})

    fig, ax = plt.subplots(figsize=(5, max(5, top_n * 0.45)))
    vmax = float(heatmap_data.abs().max().max())
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax, vmax=vmax,
        annot=True, fmt='.3f',
        linewidths=0.5,
        annot_kws={'fontsize': 8},
        cbar_kws={'label': 'PDP Effect (P10→P90)'},
    )
    ax.set_title(
        'PDP Effect Size: Feature P10→P90 on V/A\n'
        '(Red=increases emotion, Blue=decreases)',
        fontsize=10,
    )
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    _savefig(fig, 'reg_pdp_delta_heatmap.png')


# ============================================================
#          5. 综合文字报告
# ============================================================

def print_sensitivity_report(
    sens_v: pd.DataFrame,
    sens_a: pd.DataFrame,
    top_n: int = 10,
):
    """在控制台打印可读的敏感性分析报告。"""
    target_label = {'valence': 'Valence（效价）', 'arousal': 'Arousal（唤醒度）'}

    for sens_df, target in [(sens_v, 'valence'), (sens_a, 'arousal')]:
        print(f"\n  {'═'*65}")
        print(f"  {target_label[target]} 敏感性分析 TOP {top_n}")
        print(f"  {'═'*65}")
        print(f"  {'特征名':<35} {'Δ/σ':>8} {'PDP(P10→P90)':>14} {'方向'}")
        print(f"  {'─'*65}")

        top = sens_df.nlargest(top_n, 'shap_importance')
        for _, row in top.iterrows():
            direction = '↑ 升高' if row['delta_per_sigma'] > 0 else '↓ 降低'
            print(
                f"  {row['feature']:<35} "
                f"{row['delta_per_sigma']:>+8.4f} "
                f"{row['pdp_delta_p10_p90']:>+14.4f}  {direction}"
            )
            # 具体示例描述
            feat_range = row['feat_p90'] - row['feat_p10']
            unit       = row['unit_change_ref']
            delta_unit = row['delta_per_unit']
            if abs(delta_unit) > 1e-6 and feat_range > 1e-6:
                print(
                    f"    └ 示例: {row['feature']} 增加 {unit:.4f}"
                    f"（约 P10-P90 范围的 10%）→ "
                    f"预测 {target_label[target].split('（')[0]} "
                    f"变化 {delta_unit:+.5f}"
                )


# ============================================================
#          6. 主入口
# ============================================================

def run_regression_shap(
    reg_output: dict,
    feature_cols: List[str],
    best_model_name: str = 'lightgbm',
    top_n_importance: int = 20,
    top_n_pdp: int = 9,
    top_n_sensitivity: int = 15,
):
    """
    对 Valence 和 Arousal 回归模型运行完整 SHAP + 敏感性分析。

    Parameters
    ----------
    reg_output       : train_regression.py 的 run_va_regression() 返回值
    feature_cols     : 特征名列表
    best_model_name  : 用于 SHAP/PDP 分析的主模型（默认 lightgbm）
    """
    print(f"\n{'='*65}")
    print(f"  V-A 回归 SHAP 与敏感性分析")
    print(f"  主分析模型: {best_model_name}")
    print(f"{'='*65}")

    sensitivity_tables = {}

    for target_name in ['valence', 'arousal']:
        target_data = reg_output[target_name]
        results     = target_data['results']
        X_train     = target_data['X_train']
        X_test      = target_data['X_test']

        # 选用最优模型
        summary     = target_data['summary']
        actual_best = summary['r2'].idxmax().lower().replace(' ', '_')
        model_key   = best_model_name if best_model_name in results else actual_best
        model       = results[model_key]['model']

        target_label = 'Valence' if target_name == 'valence' else 'Arousal'
        print(f"\n  [{target_label}] 使用模型: {model_key}  "
              f"(R²={results[model_key]['metrics']['r2']:.4f})")

        # ── SHAP 计算 ──────────────────────────────────────────────────
        try:
            shap_values, expected_value = _get_shap_values_regression(model, X_test)
        except Exception as e:
            print(f"  [WARN] SHAP 计算失败: {e}，跳过 {target_name}")
            continue

        # 对齐特征维度（防止模型丢弃常数特征）
        shap_n = shap_values.shape[1]
        fc     = feature_cols[:shap_n]
        X_expl = X_test[:, :shap_n]

        # ── 1. 全局重要性 ──────────────────────────────────────────────
        imp_df = plot_shap_importance_regression(
            shap_values, fc, model_key, target_name, top_n=top_n_importance
        )
        imp_file = cfg.OUTPUT_DIR / f'reg_shap_importance_{target_name}_{model_key}.csv'
        imp_df.to_csv(imp_file, index=False)
        print(f"  SHAP 重要性已保存: {imp_file}")

        print(f"\n  Top 10 特征 ({target_label}):")
        for _, row in imp_df.head(10).iterrows():
            print(f"    #{int(row['rank']):2d}  {row['feature']:<40}  {row['importance_mean']:.5f}")

        # ── 2. Beeswarm ────────────────────────────────────────────────
        plot_shap_beeswarm_regression(
            shap_values, X_expl, fc, model_key, target_name, top_n=min(15, top_n_importance)
        )

        # ── 3. PDP 网格 ────────────────────────────────────────────────
        print(f"  计算偏依赖图 (PDP) Top {top_n_pdp} 特征 ...")
        plot_pdp_grid(
            model, X_expl, fc, imp_df,
            model_key, target_name, top_n=top_n_pdp
        )

        # ── 4. 敏感性定量分析 ──────────────────────────────────────────
        print(f"  计算敏感性量化表 ...")
        sens_df = compute_sensitivity_table(
            shap_values, X_expl, fc, model, target_name,
            top_n=top_n_sensitivity
        )
        sens_file = cfg.OUTPUT_DIR / f'reg_sensitivity_{target_name}_{model_key}.csv'
        sens_df.to_csv(sens_file, index=False)
        print(f"  敏感性表已保存: {sens_file}")
        sensitivity_tables[target_name] = sens_df

    # ── 5. 跨目标对比可视化 ────────────────────────────────────────────
    if 'valence' in sensitivity_tables and 'arousal' in sensitivity_tables:
        print(f"\n  生成 V-A 双目标对比图 ...")
        plot_sensitivity_bar(
            sensitivity_tables['valence'],
            sensitivity_tables['arousal'],
            top_n=12,
        )
        plot_pdp_delta_heatmap(
            sensitivity_tables['valence'],
            sensitivity_tables['arousal'],
            top_n=15,
        )

    # ── 6. 控制台报告 ──────────────────────────────────────────────────
    if sensitivity_tables:
        print_sensitivity_report(
            sensitivity_tables.get('valence', pd.DataFrame()),
            sensitivity_tables.get('arousal', pd.DataFrame()),
            top_n=10,
        )

    print(f"\n{'='*65}")
    print(f"  V-A 回归分析完成！图表已保存至: {cfg.FIGURES_DIR}")
    print(f"{'='*65}")

    return sensitivity_tables

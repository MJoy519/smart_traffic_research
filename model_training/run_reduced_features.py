"""
特征裁剪后的完整重训练脚本

功能：
  - 去除 Valence 回归 SHAP 重要性排名后 10 位的特征（含所有 SHAP=0 的特征）
  - 在裁剪后的 30 维特征上重新运行：
      1. V-A 连续值回归：XGBoost / LightGBM / RF
      2. 回归 SHAP + 敏感性定量分析（PDP / Δŷ/σ）
  - 所有结果保存至独立新目录（results_reduced / figures_reduced / saved_models_reduced）
  - 末尾输出与原始 40 维特征结果的性能对比

运行方式:
  cd Code_0321
  python model_training/run_reduced_features.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR / 'label_generation_cc'))
sys.path.insert(0, str(_THIS_DIR))

import config as cfg

# ============================================================
#  新输出路径（不覆盖原始结果）
# ============================================================

REDUCED_OUTPUT_DIR  = _THIS_DIR / 'results_reduced'
REDUCED_FIGURES_DIR = _THIS_DIR / 'figures_reduced'
REDUCED_MODELS_DIR  = _THIS_DIR / 'saved_models_reduced'

REDUCED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REDUCED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REDUCED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 临时覆盖 cfg 中的路径，让下游函数自动写到新目录
_orig_output_dir  = cfg.OUTPUT_DIR
_orig_figures_dir = cfg.FIGURES_DIR
_orig_model_dir   = cfg.MODEL_SAVE_DIR

cfg.OUTPUT_DIR    = REDUCED_OUTPUT_DIR
cfg.FIGURES_DIR   = REDUCED_FIGURES_DIR
cfg.MODEL_SAVE_DIR= REDUCED_MODELS_DIR

# ============================================================
#  确定要去除的后 10 位特征
# ============================================================

SHAP_IMP_FILE = _orig_output_dir / 'reg_shap_importance_valence_lightgbm.csv'

def get_drop_features(top_n_drop: int = 10) -> list:
    """读取 Valence 回归 SHAP 重要性表，返回排名最后 top_n_drop 个特征名。"""
    if not SHAP_IMP_FILE.exists():
        raise FileNotFoundError(
            f"找不到 SHAP 重要性文件: {SHAP_IMP_FILE}\n"
            "请先运行 run_pipeline.py 生成完整 SHAP 结果。"
        )
    df = pd.read_csv(SHAP_IMP_FILE).sort_values('rank', ascending=False)
    drop = df.head(top_n_drop)['feature'].tolist()
    return drop

# ============================================================
#  主流程
# ============================================================

def main():
    print(f"\n{'═'*65}")
    print(f"  特征裁剪重训练 — 去除 Valence SHAP 后 10 位特征")
    print(f"  输出目录: {REDUCED_OUTPUT_DIR.name} / {REDUCED_FIGURES_DIR.name}")
    print(f"{'═'*65}")

    # ── 1. 确定丢弃特征 ──────────────────────────────────────────────
    drop_feats = get_drop_features(10)
    print(f"\n  [DROP] 去除以下 {len(drop_feats)} 个特征:")
    for f in drop_feats:
        print(f"    - {f}")

    # ── 2. 加载数据集 ────────────────────────────────────────────────
    print(f"\n{'━'*65}")
    print(f"  Step 1 — 加载特征 + 标签")
    print(f"{'━'*65}")

    from dataset_builder import build_dataset
    df_dataset, X_full, y_valence_full, feature_cols_full, meta_df = build_dataset()

    # 加载 V-A 连续值标签
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "lcc_config",
        str(_ROOT_DIR / "label_generation_cc" / "config.py"),
    )
    _lcc_cfg = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_lcc_cfg)
    label_df = pd.read_csv(_lcc_cfg.OUTPUT_DIR / 'window_labels.csv')
    merge_keys = ['route_num', 'video_name', 'window_idx']
    meta_with_va = meta_df.merge(
        label_df[merge_keys + ['valence_median', 'arousal_median']],
        on=merge_keys, how='left',
    )
    y_valence = meta_with_va['valence_median'].values.astype(float)
    y_arousal = meta_with_va['arousal_median'].values.astype(float)

    # ── 3. 裁剪特征 ──────────────────────────────────────────────────
    keep_mask    = [f not in drop_feats for f in feature_cols_full]
    feature_cols = [f for f in feature_cols_full if f not in drop_feats]
    X            = X_full[:, keep_mask]

    print(f"\n  原始特征数: {len(feature_cols_full)}  →  裁剪后: {len(feature_cols)}")
    print(f"  样本数: {len(X)}")

    # ── 4. V-A 回归训练 ──────────────────────────────────────────────
    print(f"\n{'━'*65}")
    print(f"  Step 2 — V-A 回归训练（{len(feature_cols)} 维特征）")
    print(f"{'━'*65}")

    from train_regression import run_va_regression
    reg_output = run_va_regression(X, y_valence, y_arousal, feature_cols)

    # ── 5. 回归 SHAP + 敏感性分析 ────────────────────────────────────
    print(f"\n{'━'*65}")
    print(f"  Step 3 — 回归 SHAP + 敏感性定量分析")
    print(f"{'━'*65}")

    from regression_shap import run_regression_shap
    sensitivity_tables = run_regression_shap(
        reg_output      = reg_output,
        feature_cols    = feature_cols,
        best_model_name = 'xgboost',
    )

    # ── 6. 与原始结果对比 ────────────────────────────────────────────
    print(f"\n{'━'*65}")
    print(f"  Step 4 — 与原始 40 维特征结果对比")
    print(f"{'━'*65}")
    _compare_with_original(reg_output)

    print(f"\n{'━'*65}")
    print(f"  所有结果已保存至:")
    print(f"    训练结果: {REDUCED_OUTPUT_DIR}")
    print(f"    图表:     {REDUCED_FIGURES_DIR}")
    print(f"    模型:     {REDUCED_MODELS_DIR}")
    print(f"{'━'*65}")


# ============================================================
#  与原始结果对比
# ============================================================

def _compare_with_original(reg_output_new: dict):
    """
    加载原始结果（40维）与裁剪后结果（30维）进行回归性能对比，输出并保存对比表。
    """
    orig_reg_v = _orig_output_dir / 'regression_valence_comparison.csv'
    orig_reg_a = _orig_output_dir / 'regression_arousal_comparison.csv'

    rows_reg = []

    for target, orig_file, reg_key in [
        ('valence', orig_reg_v, 'valence'),
        ('arousal', orig_reg_a, 'arousal'),
    ]:
        if orig_file.exists():
            df_orig = pd.read_csv(orig_file)
            df_orig['feature_set'] = '40 features (original)'
            df_orig['target']      = target

            df_new = reg_output_new[reg_key]['summary'].reset_index()
            df_new.rename(columns={'index': 'model'}, inplace=True)
            df_new['feature_set'] = '30 features (reduced)'
            df_new['target']      = target

            df_orig['model'] = df_orig['model'].str.lower().str.replace(' ', '_')
            df_new['model']  = df_new['model'].str.lower().str.replace(' ', '_')

            cols = ['model', 'target', 'feature_set', 'r2', 'rmse', 'mae', 'pearson_r']
            cols_o = [c for c in cols if c in df_orig.columns]
            cols_n = [c for c in cols if c in df_new.columns]
            rows_reg.append(pd.concat([df_orig[cols_o], df_new[cols_n]], ignore_index=True))

    if rows_reg:
        df_reg_compare = pd.concat(rows_reg, ignore_index=True)
        print(f"\n  ┌ 回归模型对比（R² / RMSE / Pearson r）")
        try:
            pivot = df_reg_compare.pivot_table(
                index=['target', 'model'],
                columns='feature_set',
                values=['r2', 'rmse'],
                aggfunc='first',
            )
            print(pivot.round(4).to_string())
        except Exception:
            print(df_reg_compare.round(4).to_string(index=False))

        df_reg_compare.to_csv(REDUCED_OUTPUT_DIR / 'comparison_regression.csv', index=False)
        print(f"  已保存: {REDUCED_OUTPUT_DIR / 'comparison_regression.csv'}")

    _plot_comparison(pd.concat(rows_reg, ignore_index=True) if rows_reg else pd.DataFrame())


def _plot_comparison(reg_df: pd.DataFrame):
    """绘制 40 维 vs 30 维回归性能对比柱状图。"""
    if reg_df.empty or 'r2' not in reg_df.columns:
        return

    targets = [t for t in ['valence', 'arousal'] if t in reg_df['target'].values]
    n_panels = len(targets)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = {'40 features (original)': '#78909C', '30 features (reduced)': '#1565C0'}

    for ax, target in zip(axes, targets):
        sub = reg_df[reg_df['target'] == target]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index='model', columns='feature_set',
                                values='r2', aggfunc='first')
        pivot.plot(kind='bar', ax=ax, color=[colors[c] for c in pivot.columns],
                   edgecolor='white', width=0.7)
        ax.set_title(f'{target.title()} Regression — R²', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('R²')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=8, title='Feature Set')
        ax.grid(True, axis='y', alpha=0.3)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.005,
                        f'{p.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Regression Performance: 40 Features vs 30 Features (Reduced)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path = REDUCED_FIGURES_DIR / 'comparison_regression.png'
    fig.savefig(out_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    print(f"  对比图已保存: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()

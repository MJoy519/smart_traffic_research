"""
Valence / Arousal 连续值回归训练模块

功能：
  - 以 valence_median 和 arousal_median 为回归目标，训练 XGBoost / LightGBM / RandomForest
  - 5-Fold KFold 交叉验证 + 独立测试集评估
  - 评估指标：R², RMSE, MAE, Pearson r
  - 输出：
      results/regression_{target}_comparison.csv
      figures/regression_pred_vs_actual_{model}_{target}.png
      figures/regression_cv_summary.png
      saved_models/regressor_{target}_{model}.pkl
"""

import sys
import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

# ============================================================
#                     回归模型工厂
# ============================================================

REGRESSOR_NAMES = ['xgboost', 'lightgbm', 'random_forest']

def build_regressor(name: str):
    """构建回归器实例。"""
    if name == 'xgboost':
        from xgboost import XGBRegressor
        p = dict(cfg.XGBOOST_DEFAULT)
        p['eval_metric'] = 'rmse'
        p['objective']   = 'reg:squarederror'
        return XGBRegressor(**p)

    elif name == 'lightgbm':
        import lightgbm as lgb
        p = dict(cfg.LGBM_DEFAULT)
        p['objective'] = 'regression'
        return lgb.LGBMRegressor(**p)

    elif name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**cfg.RF_DEFAULT)

    raise ValueError(f"未知回归模型: {name}")


# ============================================================
#                     评估指标
# ============================================================

def compute_reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算回归评估指标。"""
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r, p = pearsonr(y_true, y_pred)
    return {'r2': r2, 'rmse': rmse, 'mae': mae, 'pearson_r': float(r), 'pearson_p': float(p)}


# ============================================================
#                     交叉验证
# ============================================================

def run_cv_regression(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """5-Fold KFold 交叉验证，返回各折 R²/RMSE 均值±标准差。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
    r2_scores, rmse_scores = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        r2_scores.append(r2_score(y_va, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_va, y_pred)))

    return {
        'cv_r2_mean':   float(np.mean(r2_scores)),
        'cv_r2_std':    float(np.std(r2_scores)),
        'cv_rmse_mean': float(np.mean(rmse_scores)),
        'cv_rmse_std':  float(np.std(rmse_scores)),
    }


# ============================================================
#                     单目标回归训练
# ============================================================

def train_regression_target(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    target_name: str,    # 'valence' or 'arousal'
) -> Tuple[dict, pd.DataFrame]:
    """
    对单个连续回归目标训练多个模型并输出对比。

    Returns
    -------
    results     : {model_name: {'model': ..., 'metrics': ..., 'y_pred': ...}}
    summary_df  : 模型对比 DataFrame
    """
    # 缺失值填充
    imputer = SimpleImputer(strategy='median')
    X_imp   = imputer.fit_transform(X)

    # 时序安全分割（不 shuffle，保持时间顺序；用随机种子分层可改为 shuffle=True）
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        shuffle=True,
    )

    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  目标: {target_name}  均值={y.mean():.4f}  std={y.std():.4f}  range=[{y.min():.3f}, {y.max():.3f}]")

    results = {}
    rows    = []

    print(f"\n  5-Fold 交叉验证 ...")
    for name in REGRESSOR_NAMES:
        model = build_regressor(name)
        cv    = run_cv_regression(model, X_train, y_train)
        print(f"    CV: {name:<15s}  R²={cv['cv_r2_mean']:.3f}±{cv['cv_r2_std']:.3f}"
              f"  RMSE={cv['cv_rmse_mean']:.4f}±{cv['cv_rmse_std']:.4f}")

        # 在全训练集上重新拟合
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m      = compute_reg_metrics(y_test, y_pred)

        print(f"  训练 {name.upper():<15s}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}"
              f"  MAE={m['mae']:.4f}  r={m['pearson_r']:.4f}")

        results[name] = {
            'model':    model,
            'imputer':  imputer,
            'metrics':  m,
            'y_pred':   y_pred,
            'y_test':   y_test,
            'X_train':  X_train,
            'X_test':   X_test,
            'y_train':  y_train,
        }
        rows.append({
            'model':    name.title().replace('_', ' '),
            'target':   target_name,
            'r2':       m['r2'],
            'rmse':     m['rmse'],
            'mae':      m['mae'],
            'pearson_r': m['pearson_r'],
            **{f'cv_{k}': v for k, v in cv.items()},
        })

    summary_df = pd.DataFrame(rows).set_index('model')
    return results, summary_df, imputer, X_train, X_test, y_train, y_test


# ============================================================
#                     可视化
# ============================================================

def _savefig(fig: plt.Figure, fname: str):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / fname
    fig.savefig(path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
    print(f"  图片已保存: {path}")
    if cfg.SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def plot_pred_vs_actual(results: dict, target_name: str):
    """预测值 vs 实际值散点图（各模型子图）。"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    target_label = 'Valence' if target_name == 'valence' else 'Arousal'

    for ax, (name, res) in zip(axes, results.items()):
        y_test = res['y_test']
        y_pred = res['y_pred']
        m      = res['metrics']

        ax.scatter(y_test, y_pred, alpha=0.45, s=20, color='steelblue', edgecolors='none')
        lims = [min(y_test.min(), y_pred.min()) - 0.05,
                max(y_test.max(), y_pred.max()) + 0.05]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect fit')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(f'Actual {target_label}', fontsize=10)
        ax.set_ylabel(f'Predicted {target_label}', fontsize=10)
        ax.set_title(
            f"{name.replace('_',' ').title()}\n"
            f"R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}  r={m['pearson_r']:.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Predicted vs Actual — {target_label}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, f'regression_pred_actual_{target_name}.png')


def plot_residuals(results: dict, target_name: str):
    """残差分布图。"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    target_label = 'Valence' if target_name == 'valence' else 'Arousal'

    for ax, (name, res) in zip(axes, results.items()):
        residuals = res['y_test'] - res['y_pred']
        ax.hist(residuals, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Residual', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f"{name.replace('_',' ').title()} — Residuals\n"
                     f"mean={residuals.mean():.4f}  std={residuals.std():.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Residual Distribution — {target_label}', fontsize=12)
    plt.tight_layout()
    _savefig(fig, f'regression_residuals_{target_name}.png')


def plot_regression_cv_summary(summary_v: pd.DataFrame, summary_a: pd.DataFrame):
    """V 和 A 回归的 CV 对比图（R² 和 RMSE）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for ax_idx, (summary_df, target_label) in enumerate(
        [(summary_v, 'Valence'), (summary_a, 'Arousal')]
    ):
        ax = axes[ax_idx]
        models = list(summary_df.index)
        r2s    = summary_df['r2'].values
        x      = np.arange(len(models))

        bars = ax.bar(x, r2s, color=colors[:len(models)], alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel('R² Score', fontsize=10)
        ax.set_title(f'{target_label} Regression — Test R²', fontsize=11)
        ax.set_ylim(0, min(1.0, max(r2s) + 0.15))
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('V-A Regression Model Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, 'regression_cv_summary.png')


# ============================================================
#                     主入口
# ============================================================

def run_va_regression(
    X: np.ndarray,
    y_valence: np.ndarray,
    y_arousal: np.ndarray,
    feature_cols: List[str],
) -> dict:
    """
    对 Valence 和 Arousal 分别训练多个回归模型。

    Returns
    -------
    {
      'valence': {'results': ..., 'summary': ..., 'imputer': ...,
                  'X_train': ..., 'X_test': ..., 'y_train': ..., 'y_test': ...},
      'arousal': {...}
    }
    """
    print(f"\n{'=' * 65}")
    print(f"  V-A 连续值回归训练")
    print(f"  样本量: {len(X)}  |  特征数: {len(feature_cols)}")
    print(f"  模型: {REGRESSOR_NAMES}")
    print(f"{'=' * 65}")

    output = {}

    for target_name, y_target in [('valence', y_valence), ('arousal', y_arousal)]:
        print(f"\n{'─' * 50}")
        print(f"  目标变量: {target_name.upper()}")
        print(f"{'─' * 50}")

        results, summary_df, imputer, X_train, X_test, y_train, y_test = \
            train_regression_target(X, y_target, feature_cols, target_name)

        # 打印结果表
        print(f"\n  {'=' * 60}")
        print(f"  {target_name.upper()} 回归结果汇总（测试集）:")
        print(f"  {'=' * 60}")
        print(summary_df[['r2', 'rmse', 'mae', 'pearson_r']].round(4).to_string())

        # 保存 CSV
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = cfg.OUTPUT_DIR / f'regression_{target_name}_comparison.csv'
        summary_df.reset_index().to_csv(out_file, index=False)
        print(f"  回归对比已保存: {out_file}")

        # 保存最优模型（按 R² 选）
        best_name = summary_df['r2'].idxmax()
        best_res  = results[best_name.lower().replace(' ', '_')]
        cfg.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        for name, res in results.items():
            pkl_path = cfg.MODEL_SAVE_DIR / f'regressor_{target_name}_{name}.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'model':       res['model'],
                    'imputer':     res['imputer'],
                    'feature_cols': feature_cols,
                    'target':      target_name,
                    'metrics':     res['metrics'],
                }, f)
        print(f"  最优模型 ({best_name}) 已保存。")

        # 可视化
        if cfg.SAVE_FIGURES or cfg.SHOW_FIGURES:
            plot_pred_vs_actual(results, target_name)
            plot_residuals(results, target_name)

        output[target_name] = {
            'results':  results,
            'summary':  summary_df,
            'imputer':  imputer,
            'X_train':  X_train,
            'X_test':   X_test,
            'y_train':  y_train,
            'y_test':   y_test,
        }

    # 联合对比图
    if cfg.SAVE_FIGURES or cfg.SHOW_FIGURES:
        plot_regression_cv_summary(
            output['valence']['summary'],
            output['arousal']['summary'],
        )

    print(f"\n{'=' * 65}")
    print(f"  回归训练完成！")
    print(f"{'=' * 65}")

    return output

"""
完整 ML 训练流程总入口（V-A 回归版）

执行顺序:
  Step 1: 标签生成（罗素情感环 V-A 映射，可跳过）
  Step 2: 数据集构建（特征 + V-A 连续标签对齐）
  Step 3: V-A 连续值回归训练（XGBoost / LightGBM / RandomForest）
  Step 4: 回归 SHAP + 偏依赖图（PDP）+ 敏感性定量分析（Δŷ/σ）

运行方式:
  cd Code_0321
  python model_training/run_pipeline.py

  或指定步骤跳过已完成内容:
  python model_training/run_pipeline.py --skip-label        # 跳过标签生成
  python model_training/run_pipeline.py --skip-regression   # 跳过回归训练，仅重新生成 SHAP 图

推荐使用裁剪特征版本（30维，去除低贡献特征）:
  python model_training/run_reduced_features.py
"""

import sys
import os
import argparse
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR / 'label_generation_cc'))
sys.path.insert(0, str(_THIS_DIR))

import config as cfg
from train_regression import run_va_regression
from regression_shap  import run_regression_shap


# ============================================================
#                  命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Smart Traffic Emotion — V-A Regression Pipeline'
    )
    parser.add_argument('--skip-label',    action='store_true',
                        help='跳过标签生成（直接使用已有 window_labels.csv）')
    parser.add_argument('--skip-regression', action='store_true',
                        help='跳过回归训练（直接使用已保存模型，仅重新生成 SHAP）')
    return parser.parse_args()


# ============================================================
#                  分步流程
# ============================================================

def step_generate_labels():
    """Step 1: 运行标签生成流程。"""
    print(f"\n{'━' * 65}")
    print(f"  Step 1 — 标签生成（罗素情感环 V-A 映射）")
    print(f"{'━' * 65}")
    from label_generator import main as gen_main
    return gen_main()


def step_build_dataset():
    """Step 2: 构建训练数据集。"""
    print(f"\n{'━' * 65}")
    print(f"  Step 2 — 数据集构建（特征 + 标签对齐）")
    print(f"{'━' * 65}")
    from dataset_builder import main as ds_main
    return ds_main()


def step_regression(X, y_valence, y_arousal, feature_cols):
    """Step 3-4: V-A 连续值回归训练 + SHAP 敏感性分析。"""
    print(f"\n{'━' * 65}")
    print(f"  Step 3 — V-A 连续值回归训练（Valence / Arousal）")
    print(f"{'━' * 65}")

    reg_output = run_va_regression(X, y_valence, y_arousal, feature_cols)

    print(f"\n{'━' * 65}")
    print(f"  Step 4 — 回归 SHAP + 敏感性定量分析（PDP / Δŷ/σ）")
    print(f"{'━' * 65}")

    sensitivity_tables = run_regression_shap(
        reg_output      = reg_output,
        feature_cols    = feature_cols,
        best_model_name = 'xgboost',
    )
    return reg_output, sensitivity_tables


# ============================================================
#                       主入口
# ============================================================

def main():
    args = parse_args()

    print(f"\n{'═' * 65}")
    print(f"  Smart Traffic Emotion — V-A Regression Pipeline")
    print(f"  输出目录: {cfg.OUTPUT_DIR}")
    print(f"{'═' * 65}")

    # ── Step 1: 标签生成 ──
    if not args.skip_label:
        label_file = _ROOT_DIR / 'label_generation_cc' / 'results_cc' / 'window_labels.csv'
        if not label_file.exists():
            step_generate_labels()
        else:
            print(f"\n  [INFO] 已有标签文件，跳过生成步骤: {label_file}")

    # ── Step 2: 数据集构建 ──
    from dataset_builder import build_dataset
    df_dataset, X, y, feature_cols, meta_df = build_dataset()

    # 加载 V-A 连续值标签（回归目标）
    import importlib.util as _ilu
    _lcc_spec = _ilu.spec_from_file_location(
        "lcc_config",
        str(_ROOT_DIR / "label_generation_cc" / "config.py"),
    )
    _lcc_cfg = _ilu.module_from_spec(_lcc_spec)
    _lcc_spec.loader.exec_module(_lcc_cfg)

    _label_df   = pd.read_csv(_lcc_cfg.OUTPUT_DIR / 'window_labels.csv')
    _merge_keys = ['route_num', 'video_name', 'window_idx']
    _meta_with_va = meta_df.merge(
        _label_df[_merge_keys + ['valence_median', 'arousal_median']],
        on=_merge_keys, how='left',
    )
    y_valence = _meta_with_va['valence_median'].values.astype(float)
    y_arousal = _meta_with_va['arousal_median'].values.astype(float)

    print(f"\n  数据集: {X.shape[0]} 样本 × {X.shape[1]} 特征")
    print(f"  Valence 范围: [{y_valence.min():.4f}, {y_valence.max():.4f}]")
    print(f"  Arousal 范围: [{y_arousal.min():.4f}, {y_arousal.max():.4f}]")

    # ── Step 3-4: V-A 回归 + 敏感性分析 ──
    if not args.skip_regression:
        reg_output, sensitivity_tables = step_regression(
            X, y_valence, y_arousal, feature_cols
        )
    else:
        print(f"\n  [INFO] 跳过回归训练（--skip-regression）")

    print(f"\n{'━' * 65}")
    print(f"  所有结果已保存至:")
    print(f"    训练结果: {cfg.OUTPUT_DIR}")
    print(f"    图表:     {cfg.FIGURES_DIR}")
    print(f"    模型:     {cfg.MODEL_SAVE_DIR}")
    print(f"{'━' * 65}\n")


if __name__ == '__main__':
    main()

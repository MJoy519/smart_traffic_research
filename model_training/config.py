"""
模型训练全局配置
"""

import os
from pathlib import Path

# ==================== 基础路径 ====================
BASE_DIR    = Path(__file__).resolve().parent.parent   # Code_0321 根目录
LABEL_CC_DIR = BASE_DIR / 'label_generation_cc'

DATASET_FILE    = LABEL_CC_DIR / 'results_cc' / 'training_dataset.csv'
FEAT_COL_FILE   = LABEL_CC_DIR / 'results_cc' / 'feature_columns.txt'
LABEL_FILE      = LABEL_CC_DIR / 'results_cc' / 'window_labels.csv'

OUTPUT_DIR      = Path(__file__).resolve().parent / 'results'
FIGURES_DIR     = Path(__file__).resolve().parent / 'figures'
MODEL_SAVE_DIR  = Path(__file__).resolve().parent / 'saved_models'

# ==================== 训练设置 ====================
TEST_SIZE       = 0.20        # 测试集比例
RANDOM_STATE    = 42
CV_FOLDS        = 5           # K-fold 交叉验证折数

# ==================== XGBoost 超参数 ====================
XGBOOST_DEFAULT = {
    'n_estimators':     200,
    'max_depth':        5,
    'learning_rate':    0.1,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'random_state':     RANDOM_STATE,
    'n_jobs':           -1,
}

# ==================== Random Forest 超参数 ====================
RF_DEFAULT = {
    'n_estimators': 300,
    'max_depth':    None,
    'min_samples_split': 5,
    'min_samples_leaf':  2,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs':       -1,
}

# ==================== LightGBM 超参数 ====================
LGBM_DEFAULT = {
    'n_estimators':   300,
    'num_leaves':     31,
    'learning_rate':  0.05,
    'subsample':      0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 10,
    'reg_alpha':      0.1,
    'reg_lambda':     0.1,
    'random_state':   RANDOM_STATE,
    'n_jobs':         -1,
    'verbose':        -1,
}

# ==================== 可视化 ====================
SAVE_FIGURES = True
SHOW_FIGURES = False
FIGURE_DPI   = 150

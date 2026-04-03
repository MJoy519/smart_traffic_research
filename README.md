# Smart Traffic Emotion — V-A Regression Pipeline

基于驾驶视频场景特征，利用 iMotion 生理情绪数据，预测驾驶者在不同路段的情感状态（Valence / Arousal 连续值）。

---

## 项目概览

```
Code_0321/
├── videos/                   # 原始行车视频（按路线分组）
├── iMotion/                  # iMotion 情绪原始数据
├── self_report/              # 受试者自我报告 V-A 问卷数据
├── feature_extraction/       # 阶段一：视频场景特征提取
├── label_generation_cc/      # 阶段二：V-A 情感标签生成 + 数据集构建
├── model_training/           # 阶段三：V-A 回归模型训练 + SHAP 分析
└── README.md
```

完整 pipeline 分三个阶段串联执行，数据流如下：

```
videos/ + iMotion/
    │
    ├─[阶段一] feature_extraction/main.py
    │           └─→ feature_extraction/results/{route}_{extractor}_3_1.csv
    │
    ├─[阶段二] label_generation_cc/label_generator.py
    │           └─→ label_generation_cc/results_cc/window_labels.csv
    │
    │           label_generation_cc/dataset_builder.py
    │           └─→ label_generation_cc/results_cc/training_dataset.csv
    │
    └─[阶段三] model_training/run_pipeline.py  （或 run_reduced_features.py）
                ├─→ model_training/results/          # 回归性能 CSV
                ├─→ model_training/figures/          # 可视化图表
                └─→ model_training/saved_models/     # 训练好的模型 .pkl
```

---

## 目录说明

### `videos/`
行车实验原始视频，按路线编号组织：
```
videos/
├── 1/    # Route 1 的视频片段（CUT 2, 3, 4, 5）
├── 2/    # Route 2 的视频片段（CUT 1, 2, 3, 4, 5）
└── 3/    # Route 3（备用）
```
支持格式：`.mp4`、`.avi`、`.mov`、`.mkv`

---

### `iMotion/`
受试者情绪识别原始数据，由 iMotion 软件采集：
```
iMotion/
└── raw_processed/
    ├── route1/
    │   ├── P26/   p26-2.csv, p26-3.csv, ...
    │   └── P27/   ...
    └── route2/
        └── ...
```
每个 CSV 含 7 种情绪强度列（Anger, Contempt, Disgust, Fear, Joy, Sadness, Confusion）及时间戳。

---

### `self_report/`
实验后受试者填写的视频级别 Valence / Arousal 主观评分：
```
self_report/
├── self-va-processed-1.csv   # Route 1 受试者自评
└── self-va-processed-2.csv   # Route 2 受试者自评
```
用于对比验证生成的 V-A 标签的有效性。

---

### `feature_extraction/`
**阶段一：从行车视频中提取场景级交通特征。**

| 文件/目录 | 说明 |
|-----------|------|
| `main.py` | **主执行脚本**，控制提取器选择、路线、窗口参数 |
| `config.py` | 模型路径、视频路径、ROI、推理参数等全局配置 |
| `compute_composite.py` | 复合特征计算（多模型联合特征，7 维） |
| `download_models.py` | 自动下载 SegFormer 预训练权重 |
| `extractor_test.py` | 单视频快速测试脚本 |
| `roi_selector.py` | 交互式 ROI 区域选取工具 |
| `requirements.txt` | Python 依赖包列表 |
| `feature_extractor/` | 三个特征提取器的具体实现 |
| `pretrain_model/` | 预训练模型权重存放目录 |
| `results/` | 提取结果 CSV（自动生成） |

**提取的特征（共 40 维）：**

| 提取器 | 维数 | 主要特征 |
|--------|------|----------|
| YOLO BDD100K | 16 | 车辆/行人数量、速度、TTC 碰撞风险 |
| SegFormer | 8 | 建筑遮蔽、绿化覆盖、天空可见度等语义场景特征 |
| YOLOPv2 | 9 | 可驾驶区域覆盖、车道线曲率、车道偏移 |
| Composite | 7 | 综合驾驶压力、VRU 冲突、语义单调度等复合指标 |

**模型依赖（需提前准备）：**
- `pretrain_model/yolo26-bdd100k.pt` — YOLO BDD100K 检测权重
- `pretrain_model/segformer/` — SegFormer-B1 Cityscapes 权重（可运行 `download_models.py` 自动下载）
- `pretrain_model/yolopv2/` — 克隆 [YOLOPv2 仓库](https://github.com/CAIC-AD/YOLOPv2) 并放置 `data/weights/yolopv2.pt`

---

### `label_generation_cc/`
**阶段二：将 iMotion 情绪数据映射到 V-A 情感空间，生成窗口级连续标签，并构建训练数据集。**

| 文件 | 说明 |
|------|------|
| `label_generator.py` | **主流程**：按时间窗口聚合多受试者 V-A 坐标，输出 `window_labels.csv` |
| `russell_va_mapping.py` | 核心算法：罗素情感环加权质心投影（V-A 映射工具函数） |
| `dataset_builder.py` | 将特征 CSV 与 V-A 标签内连接对齐，输出 `training_dataset.csv` |
| `config.py` | 路径、窗口参数、情感环坐标、聚合方式等配置 |
| `results_cc/` | 输出目录（`window_labels.csv`、`training_dataset.csv`） |
| `figures_cc/` | 可视化输出（V-A 散点图、热力图、时序曲线等） |

**核心算法（罗素情感环加权投影）：**
```
V_window = Σ(emotion_i × V_russell_i) / Σ(emotion_i)
A_window = Σ(emotion_i × A_russell_i) / Σ(emotion_i)
```
跨受试者取中位数，窗口大小 3s / 步长 1s（与特征提取保持一致）。

---

### `model_training/`
**阶段三：以 Valence 和 Arousal 为回归目标，训练预测模型并进行可解释性分析。**

| 文件 | 说明 |
|------|------|
| `run_pipeline.py` | **主入口**：完整 V-A 回归 pipeline（标签→数据集→回归→SHAP） |
| `run_reduced_features.py` | **特征裁剪版本**：去除低贡献后 10 位特征，在 30 维上重训练 |
| `train_regression.py` | V-A 连续值回归训练（XGBoost / LightGBM / RandomForest，5-Fold KFold） |
| `regression_shap.py` | 回归 SHAP 分析、偏依赖图（PDP）、敏感性定量表（Δŷ/σ） |
| `config.py` | 训练超参数、路径配置 |
| `results/` | 全量特征（40维）训练结果 CSV |
| `results_reduced/` | 裁剪特征（30维）训练结果 CSV |
| `figures/` | 全量特征图表 |
| `figures_reduced/` | 裁剪特征图表 |
| `saved_models/` | 全量特征回归模型 `.pkl` |
| `saved_models_reduced/` | 裁剪特征回归模型 `.pkl` |

---

## 运行环境

### Python 版本
Python 3.10+

### 安装依赖

**特征提取依赖（含深度学习模型）：**
```bash
pip install -r feature_extraction/requirements.txt
```

GPU 加速（推荐，需先根据 CUDA 版本安装 PyTorch）：
```bash
# 例：CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**模型训练依赖（在特征提取依赖基础上追加）：**
```bash
pip install xgboost lightgbm scikit-learn shap
```

---

## 运行步骤

所有命令均从项目根目录 `Code_0321/` 执行。

### 阶段一：特征提取

```bash
python feature_extraction/main.py
```

在 `main.py` 顶部可调整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE` | `"auto"` | `"auto"` / `"cuda"` / `"cpu"` |
| `WINDOW_SIZE` | `3` | 时间窗口大小（秒） |
| `WINDOW_STEP` | `1` | 滑动步长（秒） |
| `USE_EXTRACTORS` | 全选 | `["segformer", "yolo", "yolopv2"]` |
| `TARGET_ROUTES` | `[3]` | 要处理的路线编号 |

输出：`feature_extraction/results/ws3_wt1_si1/{route}_{extractor}_3_1.csv`

> **注意**：处理完所有路线后，还需单独运行复合特征计算：
> ```bash
> python feature_extraction/compute_composite.py
> ```

---

### 阶段二：V-A 标签生成

```bash
python label_generation_cc/label_generator.py
```

输出：`label_generation_cc/results_cc/window_labels.csv`（每行为一个窗口的 `valence_median`、`arousal_median` 及统计量）

---

### 阶段三：模型训练与分析

**推荐方式 A — 完整一键运行（全量 40 维特征）：**
```bash
python model_training/run_pipeline.py
```

可选参数：
```bash
# 跳过标签生成（已有 window_labels.csv 时）
python model_training/run_pipeline.py --skip-label

# 跳过训练，仅重新生成 SHAP 图
python model_training/run_pipeline.py --skip-regression
```

**推荐方式 B — 特征裁剪版本（30 维，需先完成方式 A）：**
```bash
python model_training/run_reduced_features.py
```
读取方式 A 生成的 SHAP 重要性结果，自动去除贡献最低的 10 个特征后重新训练，并输出与全量结果的对比图。

---

## 输出文件说明

| 路径 | 内容 |
|------|------|
| `label_generation_cc/results_cc/window_labels.csv` | 每个时间窗口的 V-A 中位数、均值、标准差、IQR、有效受试者数 |
| `label_generation_cc/results_cc/training_dataset.csv` | 特征 + V-A 标签对齐后的完整训练集 |
| `model_training/results/regression_{valence\|arousal}_comparison.csv` | 三个模型在测试集的 R²、RMSE、MAE、Pearson r |
| `model_training/results/reg_shap_importance_{target}_{model}.csv` | SHAP 全局特征重要性排名 |
| `model_training/results/reg_sensitivity_{target}_{model}.csv` | 敏感性定量表（∂ŷ/∂x、Δŷ/σ、PDP 范围） |
| `model_training/saved_models/regressor_{target}_{model}.pkl` | 训练好的回归模型（含 imputer 和 feature_cols） |
| `model_training/figures/` | 预测值 vs 实际值散点图、残差图、SHAP Beeswarm、PDP 等 |
| `model_training/results_reduced/` | 裁剪后 30 维特征的对应结果（同上结构） |

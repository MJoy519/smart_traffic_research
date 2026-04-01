# 情绪聚类与一致性检验结果解读指南

> 适用脚本：`gmm_clustering.py` → `consistency_check.py`  
> 数据流：二值化情绪帧 → 时间窗口 → GMM 聚类标签 → 与 Self-Report Valence 对比

---

## 一、GMM 聚类结果解读

### 1.1 关键输出文件

| 文件 | 内容 |
|---|---|
| `emotion_windows_ws*_labeled.csv` | 每个时间窗口的聚类标签 |
| `figures/clustering/gmm_pca_scatter.png` | PCA 降维后的聚类散点图 |
| `figures/clustering/gmm_radar.png` | 各聚类中心的情绪激活雷达图 |
| `figures/clustering/gmm_cluster_barplot.png` | 各聚类情绪激活均值柱状图 |
| `figures/clustering/gmm_label_distribution.png` | 各类别数量分布饼图 |

---

### 1.2 `emotion_label` 与 `label_confidence`

```
route,participant,video,window_id,...,emotion_label,label_confidence
1,P26,2,0,...,Neutral,1.0
1,P26,2,3,...,Negative,1.0
```

| 字段 | 含义 |
|---|---|
| `emotion_label` | 该窗口的情绪极性：`Positive`（正面）/ `Neutral`（中性）/ `Negative`（负面） |
| `label_confidence` | GMM 后验概率中最大值，范围 [0,1]，越高表示归类越确定 |

**`label_confidence` 判读：**

| 值域 | 解读 |
|---|---|
| ≥ 0.90 | 窗口明确属于某类，分类可信 |
| 0.50 ~ 0.90 | 有一定不确定性，位于聚类边界 |
| ≈ 0.33 | 三类概率接近，该窗口处于交界模糊区 |

> 本项目中大量窗口的 `label_confidence = 1.0`，原因是 median 聚合后的二值特征（0 或 1）使各类在特征空间中分离非常清晰，GMM 可以以极高置信度区分。**这不代表结果一定可靠**，高置信度仅说明聚类模型对这些数据点非常"确定"——但若所有窗口都挤在同一类，模型也会对此给出 confidence=1.0。

---

### 1.3 聚类分布：如何判断好坏

#### ✅ 理想的聚类分布

```
Positive :  ~20–40%
Neutral  :  ~30–50%
Negative :  ~20–40%
```

三类数量**相对均衡**，雷达图中各聚类中心在情绪维度上有**显著差异**：
- Positive 中心：Joy 明显高于其他
- Negative 中心：Anger/Fear/Sadness 明显高于其他
- Neutral 中心：所有情绪激活均接近 0

#### ⚠️ 警示信号：Neutral 比例过高

```
# 本项目实际情况（仅 Route 1）：
dominant_label 全部为 neutral（60/60 个视频）
prop_positive 绝大多数为 0
```

**Neutral 占比 > 85%** 是一个重要的诊断信号，可能原因如下：

| 原因 | 说明 | 改进方向 |
|---|---|---|
| 聚合方式过严 | `median` 聚合下，只有 >50% 帧激活才得 1，否则为 0，导致绝大多数窗口全零 | 改用 `mean` 聚合，保留连续激活比例信息 |
| 二值化阈值过高 | K 值过大（如 K=1.5），激活帧本身就很少 | 适当降低 K（如 1.0~1.2），增加激活帧数量 |
| 窗口太短 | 3s 窗口内情绪激活不稳定，单帧激活难过 median 门槛 | 增大窗口（如 5~10s）或改用 mean |
| 情绪本身稀疏 | 驾驶场景中被试情绪激活本就较少，是真实规律 | 分析激活率，确认不是噪声 |

---

### 1.4 雷达图（gmm_radar.png）解读

雷达图展示 3 个聚类中心在 7 种情绪上的平均激活比例。

**好的雷达图特征：**
- 三条线形状**差异明显**（Positive 在 Joy 轴突出，Negative 在负面情绪轴突出）
- Neutral 线接近中心（各情绪激活均低）

**有问题的雷达图特征：**
- 三条线几乎**重叠**或形状相似 → 聚类没有实际区分度
- 所有聚类中心均贴近 0 → 特征稀疏，模型无法找到有意义的分组

---

## 二、一致性检验结果解读

### 2.1 关键输出文件

| 文件 | 内容 |
|---|---|
| `results/consistency_merged.csv` | 每个视频的窗口标签比例 + Self-Report Valence |
| `results/consistency_stats.csv` | 所有统计检验的数值结果 |
| `figures/consistency/consistency_scatter.png` | 正负比例 vs Valence 散点图 |
| `figures/consistency/consistency_boxplot.png` | 各主导标签组的 Valence 箱线图 |
| `figures/consistency/consistency_heatmap.png` | 按比例区间分组的平均 Valence 柱状图 |

---

### 2.2 Spearman 相关系数解读

**检验假设：**
- `prop_positive ↑` → `Valence ↑`（预期正相关，r > 0）
- `prop_negative ↑` → `Valence ↓`（预期负相关，r < 0）

**判读标准：**

| r 绝对值 | 相关强度 | 期望 |
|---|---|---|
| < 0.10 | 几乎无相关 | — |
| 0.10 ~ 0.30 | 弱相关 | 可接受但需改进 |
| 0.30 ~ 0.50 | 中等相关 | 较好 |
| > 0.50 | 强相关 | 优秀 |

**p 值判读（α = 0.05）：**
- p < 0.05（标记 `*`）：结果在统计上显著，可以拒绝"无相关"的零假设
- p < 0.01（标记 `**`）：结果高度显著
- p ≥ 0.05（标记 `ns`）：结果不显著，相关性可能由随机波动导致

**方向符号：**
- `✓`：相关方向与假设一致（如 prop_negative 与 valence 呈负相关）
- `✗`：方向与假设相反，需检查聚类标签是否被正确分配

---

#### 📌 本项目实际结果（Route 1）：

```
Spearman [ALL]  prop_positive vs valence: r=+0.082  p=0.532  ns  (预期: + ✗)
Spearman [ALL]  prop_negative vs valence: r=-0.116  p=0.376  ns  (预期: - ✓)
```

**解读：**
- 两个检验均**不显著**（p >> 0.05）
- `prop_negative` 方向正确（负相关 ✓），但效应量极小（r = -0.12）
- `prop_positive` 方向错误（✗），但原因在于：**几乎所有视频的 prop_positive = 0**（见 consistency_merged.csv），没有变异量就无法产生相关

**这是聚类质量问题，不是统计方法问题。** 需要先解决 Neutral 占比过高的问题。

---

### 2.3 线性回归结果解读

**模型：** `Valence = β₀ + β₁·prop_positive + β₂·prop_negative`

| 指标 | 含义 | 理想值 |
|---|---|---|
| **R²** | 模型解释的 Valence 方差比例 | > 0.15（较好），> 0.30（优秀） |
| **p(F)** | 整体模型显著性（F 检验） | < 0.05 |
| **coef(prop_positive)** | 正面比例每增加 1，Valence 变化量 | > 0（正向） |
| **coef(prop_negative)** | 负面比例每增加 1，Valence 变化量 | < 0（负向） |

#### 📌 本项目实际结果：

```
线性回归  Valence ~ prop_positive + prop_negative:
  n=60  R²=0.0340  F=1.003  p=0.373  ns
  coef(prop_positive)=+0.518  coef(prop_negative)=-0.763
```

**解读：**
- R² = 0.034 极低，模型仅能解释 3.4% 的 Valence 变异 → **拟合效果差**
- 回归系数**方向正确**（正面比例系数为正，负面系数为负），说明模型逻辑没有反转
- 不显著的原因：`prop_positive` 在所有样本中几乎为 0（无变异），回归无法估计其真实效应

---

### 2.4 Kruskal-Wallis 检验解读

检验各**主导标签组**（Positive / Neutral / Negative 视频）之间的 Valence 是否有显著差异。

**理想结果：**
```
Kruskal-Wallis: H=12.5  p=0.002 **
  Positive : median=+0.50, n=12
  Neutral  : median=+0.00, n=28
  Negative : median=-0.50, n=20
```
三组中位数呈现 Positive > Neutral > Negative 的梯度，且 p < 0.05 显著。

**本项目情况：**
- Kruskal-Wallis **未执行**（输出为空），因为 60 个视频的 `dominant_label` 全部为 `neutral`，只有 1 个分组，不满足检验的最低条件（需 ≥ 2 组且每组 ≥ 3 个样本）
- 根本原因同上：聚类后 Positive/Negative 比例过低，无法构成有效的分组变量

---

## 三、整体结果质量诊断表

| 诊断项 | 好的状态 | 当前状态 | 问题严重度 |
|---|---|---|---|
| 聚类分布均衡性 | Positive/Neutral/Negative 各占 20~50% | Neutral ≈ 90%+ | 🔴 严重 |
| label_confidence | 多数 > 0.7，有 0.33 ~ 0.7 的模糊样本 | 几乎全部 = 1.0 | 🟡 过于确定 |
| Spearman r 方向 | prop_positive (+)，prop_negative (-) | 负向方向正确，正向无效 | 🟡 部分正确 |
| Spearman 显著性 | p < 0.05 | p > 0.37 | 🔴 不显著 |
| 线性回归 R² | > 0.15 | 0.034 | 🔴 极低 |
| Kruskal-Wallis | 三组显著差异 | 无法执行 | 🔴 严重 |

---

## 四、改进建议与操作步骤

### 路径 A：改用 `mean` 聚合（推荐首先尝试）

在 `config.py` 中修改：
```python
AGGREGATION_METHOD = 'mean'   # 原为 'median'
```
`mean` 保留连续激活比例信息（如 0.23、0.67），GMM 能找到更细腻的分组边界，避免 Neutral 一统天下。

**重新运行顺序：**
```
windowing.py → gmm_clustering.py → consistency_check.py
```

---

### 路径 B：降低二值化阈值 K（增加激活帧）

在 `emotion_binarization.py` 中修改：
```python
K = 1.0   # 原为 1.5（降低后，更多帧会被标记为激活）
```
然后重新运行：
```
emotion_binarization.py → windowing.py → gmm_clustering.py → consistency_check.py
```

---

### 路径 C：增大时间窗口

在 `config.py` 或 `windowing.py` 中修改：
```python
WINDOW_SIZE_SEC = 10   # 原为 3
STEP_SIZE_SEC   = 5
```
更长的窗口内，情绪激活帧的比例更稳定，减少单帧噪声的影响。

---

### 路径 D：调整 GMM 协方差类型

在 `gmm_clustering.py` 中尝试：
```python
COVARIANCE_TYPE = 'diag'   # 原为 'full'
```
`diag` 协方差减少参数量，在样本有限时更稳定。也可以增加 `N_INIT`（如 30）提高稳定性。

---

## 五、快速判断流程图

```
运行完成后，先看 gmm_label_distribution.png
         │
         ├─ Neutral < 70%？
         │      ├─ YES → 聚类分布合理，继续看 consistency 结果
         │      └─ NO  → 🔴 聚类退化，先改 AGGREGATION_METHOD 或 K 值
         │
         └─ 看 consistency_scatter.png
                │
                ├─ 散点图有明显趋势线斜率？
                │      ├─ YES → 看 Spearman p 值
                │      │         ├─ p < 0.05 ✅ 一致性验证通过
                │      │         └─ p ≥ 0.05 → 样本量不足或噪声大
                │      └─ NO  → 🔴 聚类标签与主观评分无关联，需改进聚类方案
                │
                └─ 看 consistency_boxplot.png
                       ├─ Positive 组 Valence 中位数 > Neutral > Negative ✅
                       └─ 三组无梯度或顺序混乱 → 🔴 聚类极性判定有误
```

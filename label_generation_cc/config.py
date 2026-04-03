"""
标签生成模块全局配置
基于罗素情感环（Russell's Circumplex Model）将 iMotion 7 种情绪映射到 Valence-Arousal 空间
"""

import os
from pathlib import Path

# ==================== 基础路径 ====================
BASE_DIR        = Path(__file__).resolve().parent.parent   # Code_0321 根目录
IMOTION_RAW_DIR = BASE_DIR / 'iMotion' / 'raw_processed'
FEATURE_DIR     = BASE_DIR / 'feature_extraction' / 'results'
SELF_REPORT_DIR = BASE_DIR / 'self_report'
OUTPUT_DIR      = Path(__file__).resolve().parent / 'results_cc'
FIGURES_DIR     = Path(__file__).resolve().parent / 'figures_cc'

# ==================== 窗口参数（必须与特征提取保持一致）====================
WINDOW_SIZE_SEC   = 3    # 窗口大小（秒）
STEP_SIZE_SEC     = 1    # 滑动步长（秒）
FEATURE_DIR_LABEL = f'ws{WINDOW_SIZE_SEC}_wt{STEP_SIZE_SEC}_si1'

# ==================== 目标路线 ====================
# 仅处理已完成特征提取的路线
TARGET_ROUTES = [1, 2]

# ==================== 情绪列配置 ====================
EMOTION_COLUMNS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Confusion']

# ==================== 罗素情感环坐标 ====================
# 参考文献:
#   Russell, J. A. (1980). A circumplex model of affect. JPSP, 39(6), 1161–1178.
#   Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal,
#     and dominance for 13,915 English lemmas. Behavior Research Methods, 45(4), 1191–1207.
#   Posner, J., Russell, J.A., & Peterson, B.S. (2005). The circumplex model of affect.
#     Development and Psychopathology, 17(3), 715–734.
#
# Valence: [-1, +1]  ( -1 = 极度负面  |  +1 = 极度正面 )
# Arousal: [-1, +1]  ( -1 = 极度平静  |  +1 = 极度激动 )
RUSSELL_COORDINATES = {
    #  情绪名称       效价(V)    唤醒度(A)   位置描述
    'Joy':       {'valence':  0.76, 'arousal':  0.48},   # 正上方：高效价、中高唤醒
    'Anger':     {'valence': -0.51, 'arousal':  0.59},   # 左上：负效价、高唤醒
    'Fear':      {'valence': -0.64, 'arousal':  0.60},   # 左上：负效价、高唤醒（接近愤怒）
    'Disgust':   {'valence': -0.60, 'arousal':  0.35},   # 左侧：负效价、中等唤醒
    'Contempt':  {'valence': -0.51, 'arousal': -0.17},   # 左侧偏下：负效价、低唤醒
    'Sadness':   {'valence': -0.63, 'arousal': -0.27},   # 左下：负效价、低唤醒
    'Confusion': {'valence': -0.13, 'arousal':  0.28},   # 中间偏左：轻微负效价、低唤醒
}

# ==================== V-A 聚合参数 ====================

# 跨受试者 V-A 聚合方式
PARTICIPANT_AGG = 'median'   # 'median' | 'mean'

# 情绪最小激活总量阈值
# iMotion 情绪强度单位为 0-100（类概率），低于此总量视为 Neutral 状态（V=0,A=0）
# 设为 2.0：当7种情绪强度之和 < 2.0 时认为受试者处于"中性无激活"状态
MIN_EMOTION_ACTIVATION = 2.0

# 是否过滤受试者间分歧过大的窗口（跨被试IQR/range > 阈值时排除）
FILTER_LOW_AGREEMENT   = False
AGREEMENT_THRESHOLD    = 0.5   # Valence IQR 超过此值时过滤

# ==================== 视频名称与情绪文件编号的对应关系 ====================
# 特征提取中视频名 "CUT X" 对应 iMotion 文件 p{id}-X.csv
# Route 1: CUT 2, 3, 4, 5 → 文件编号 2, 3, 4, 5
# Route 2: CUT 1, 2, 3, 4, 5 → 文件编号 1, 2, 3, 4, 5
def video_name_to_num(video_name: str) -> int:
    """'CUT X' → X（整数）"""
    return int(video_name.strip().replace('CUT', '').strip())

# ==================== 自我报告数据映射 ====================
# self-va-processed-{route}.csv 的列名格式为 Valence{video_num}, Arousal{video_num}
SELF_REPORT_FILE = {
    r: SELF_REPORT_DIR / f'self-va-processed-{r}.csv'
    for r in [1, 2, 3]
}

# ==================== 可视化参数 ====================
SAVE_FIGURES = True
SHOW_FIGURES = False
FIGURE_DPI   = 150

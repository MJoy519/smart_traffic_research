"""
罗素情感环 V-A 映射工具模块

核心功能：
  1. 将 7 种 iMotion 情绪强度值通过加权质心法映射到 Valence-Arousal 空间
  2. 对单受试者的某视频窗口计算 V-A 坐标
  3. 提供数据加载、窗口截取、批量计算的工具函数

科学依据（可在论文中引用）：
  采用罗素情感环 (Russell, 1980) 作为先验知识，
  将 iMotion AU-based 情绪强度值通过加权投影映射到 V-A 空间：
    V_window = Σ(emotion_i_mean × V_russell_i) / (Σ emotion_i_mean + ε)
    A_window = Σ(emotion_i_mean × A_russell_i) / (Σ emotion_i_mean + ε)
  当所有情绪强度均低于激活阈值时认为处于中性状态，坐标为 (0, 0)。
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import sys
import os
import importlib.util as _ilu
_LCC_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_lcc_config():
    spec = _ilu.spec_from_file_location("lcc_config", os.path.join(_LCC_DIR, "config.py"))
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg = _load_lcc_config()


# ============================================================
#                       IO 工具
# ============================================================

def _read_csv_safe(path: Path) -> pd.DataFrame:
    """兼容多种编码的 CSV 读取。"""
    for enc in ('utf-8', 'gbk', 'latin1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码 CSV: {path}")


def get_participants(route: int) -> List[str]:
    """获取指定路线的所有受试者文件夹名列表（如 ['P26', 'P27', ...]）。"""
    route_dir = cfg.IMOTION_RAW_DIR / f'route{route}'
    if not route_dir.exists():
        warnings.warn(f"路线目录不存在: {route_dir}")
        return []
    return sorted([p.name for p in route_dir.iterdir() if p.is_dir()])


def load_participant_video(
    route: int,
    participant: str,
    video_num: int,
) -> Optional[pd.DataFrame]:
    """
    加载单受试者单视频的 iMotion 原始 CSV。

    Parameters
    ----------
    route       : int   路线编号
    participant : str   受试者目录名（如 'P26'）
    video_num   : int   视频编号（如 2 → p26-2.csv）

    Returns
    -------
    pd.DataFrame 或 None（文件不存在时返回 None）
    """
    p_lower = participant.lower()       # 'P26' → 'p26'
    filename = f'{p_lower}-{video_num}.csv'
    path = cfg.IMOTION_RAW_DIR / f'route{route}' / participant / filename

    if not path.exists():
        return None

    df = _read_csv_safe(path)

    # 确保 relative_time 列存在
    if 'relative_time' not in df.columns and 'Timestamp' in df.columns:
        ts = df['Timestamp'].values.astype(float)
        df['relative_time'] = (ts - ts[0]) / 1e9

    return df


# ============================================================
#                   核心：V-A 投影
# ============================================================

def project_to_va(
    emotion_means: Dict[str, float],
    epsilon: float = 1e-6,
) -> Tuple[float, float]:
    """
    将 7 种情绪强度均值通过加权质心法投影到 V-A 空间。

    算法:
      total  = Σ max(0, emotion_i)                          （总激活量）
      V      = Σ (emotion_i × V_russell_i) / (total + ε)
      A      = Σ (emotion_i × A_russell_i) / (total + ε)

    当 total < MIN_EMOTION_ACTIVATION 时，认为是 Neutral 状态，返回 (0.0, 0.0)。

    Parameters
    ----------
    emotion_means : dict  {情绪名: 该窗口内的均值强度}
    epsilon       : float 防除零项

    Returns
    -------
    (valence, arousal)  均在 [-1, 1] 范围内
    """
    total = sum(
        max(0.0, emotion_means.get(emo, 0.0))
        for emo in cfg.RUSSELL_COORDINATES
    )

    if total < cfg.MIN_EMOTION_ACTIVATION:
        return 0.0, 0.0

    valence = sum(
        max(0.0, emotion_means.get(emo, 0.0)) * coords['valence']
        for emo, coords in cfg.RUSSELL_COORDINATES.items()
    ) / (total + epsilon)

    arousal = sum(
        max(0.0, emotion_means.get(emo, 0.0)) * coords['arousal']
        for emo, coords in cfg.RUSSELL_COORDINATES.items()
    ) / (total + epsilon)

    return float(valence), float(arousal)


# ============================================================
#              单受试者单窗口 V-A 计算
# ============================================================

def compute_window_va(
    df: pd.DataFrame,
    window_start: float,
    window_end: float,
) -> Tuple[float, float, int]:
    """
    从单受试者情绪数据中截取时间窗口，计算该窗口的 V-A 坐标。

    Parameters
    ----------
    df           : pd.DataFrame  单受试者单视频的 iMotion 数据
    window_start : float         窗口起始时间（秒，闭区间）
    window_end   : float         窗口结束时间（秒，开区间）

    Returns
    -------
    (valence, arousal, n_frames)
      n_frames = 0 表示该窗口内无数据点
    """
    if 'relative_time' not in df.columns:
        return np.nan, np.nan, 0

    mask = (df['relative_time'] >= window_start) & (df['relative_time'] < window_end)
    df_win = df[mask]

    if len(df_win) == 0:
        return np.nan, np.nan, 0

    emotion_means = {
        emo: float(df_win[emo].mean()) if emo in df_win.columns else 0.0
        for emo in cfg.EMOTION_COLUMNS
    }

    v, a = project_to_va(emotion_means)
    return v, a, len(df_win)


# ============================================================
#         跨受试者聚合：单个视频窗口的群体 V-A
# ============================================================

def aggregate_window_va(
    route: int,
    video_name: str,
    window_start: float,
    window_end: float,
    participant_cache: Optional[Dict] = None,
) -> Dict:
    """
    对某路线某视频某窗口，聚合所有受试者的 V-A 坐标。

    Parameters
    ----------
    route            : int    路线编号
    video_name       : str    视频名称（如 'CUT 2'）
    window_start     : float  窗口起始秒
    window_end       : float  窗口结束秒
    participant_cache: dict   可选，{(route, participant, video_num): df} 缓存

    Returns
    -------
    dict，包含:
      valence_median, arousal_median   : 跨受试者 V-A 中位数
      valence_mean, arousal_mean       : 跨受试者 V-A 均值
      valence_std, arousal_std         : 跨受试者标准差
      valence_iqr, arousal_iqr         : 跨受试者四分位距
      n_valid                          : 有效受试者数量
      participant_va                   : {participant: (V, A)} 详情
    """
    video_num  = cfg.video_name_to_num(video_name)
    participants = get_participants(route)

    va_list = []
    participant_va = {}

    for p in participants:
        df = None
        if participant_cache is not None:
            key = (route, p, video_num)
            if key not in participant_cache:
                participant_cache[key] = load_participant_video(route, p, video_num)
            df = participant_cache[key]
        else:
            df = load_participant_video(route, p, video_num)

        if df is None:
            continue

        v, a, n = compute_window_va(df, window_start, window_end)

        if np.isnan(v) or n == 0:
            continue

        va_list.append((v, a))
        participant_va[p] = (v, a)

    if not va_list:
        return {
            'valence_median': np.nan, 'arousal_median': np.nan,
            'valence_mean':   np.nan, 'arousal_mean':   np.nan,
            'valence_std':    np.nan, 'arousal_std':    np.nan,
            'valence_iqr':    np.nan, 'arousal_iqr':    np.nan,
            'n_valid': 0,
            'participant_va': {},
        }

    va_arr  = np.array(va_list)   # (n_participants, 2)
    v_vals  = va_arr[:, 0]
    a_vals  = va_arr[:, 1]

    q25_v, q75_v = np.percentile(v_vals, [25, 75])
    q25_a, q75_a = np.percentile(a_vals, [25, 75])

    return {
        'valence_median': float(np.median(v_vals)),
        'arousal_median': float(np.median(a_vals)),
        'valence_mean':   float(np.mean(v_vals)),
        'arousal_mean':   float(np.mean(a_vals)),
        'valence_std':    float(np.std(v_vals)),
        'arousal_std':    float(np.std(a_vals)),
        'valence_iqr':    float(q75_v - q25_v),
        'arousal_iqr':    float(q75_a - q25_a),
        'n_valid':        len(va_list),
        'participant_va': participant_va,
    }



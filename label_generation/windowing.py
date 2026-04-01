"""
情绪时间窗口划分脚本
对二值化情绪数据按固定时间窗口（滑动窗口）进行划分，
并将所有文件的窗口合并为一个全局窗口数据集保存。

输出列：route, participant, video, window_id, window_start, window_end,
        Anger, Contempt, Disgust, Fear, Joy, Sadness, Confusion
每种情绪值为该窗口内的聚合统计量（由 AGGREGATION_METHOD 控制）。
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
#                       全局可控参数
# ============================================================
from config import WINDOW_SIZE, STEP_SIZE, AGGREGATION_METHOD, EMOTION_TARGETS
# -- 时间窗口参数 --
WINDOW_SIZE_SEC = WINDOW_SIZE        # 窗口大小（秒）
STEP_SIZE_SEC   = STEP_SIZE         # 滑动步长（秒）

# -- 目标情绪列（需与二值化脚本保持一致）--
EMOTION_COLUMNS = EMOTION_TARGETS

# -- 处理线路选择（可选 1, 2, 3 的任意组合）--
TARGET_ROUTES = [1, 2, 3]

# -- 窗口最小有效帧数：窗口内数据点少于此值时跳过该窗口 --
MIN_FRAMES_PER_WINDOW = 5

# -- 窗口内情绪聚合方式 --
# 'mean'  : 激活帧比例均值（默认，连续值 [0,1]）
# 'median': 激活帧比例中位数（对异常帧更鲁棒，二值数据结果为 0 或 1）
AGGREGATION_METHOD = 'median'

# ============================================================
#                       数据路径
# ============================================================

BASE_DIR    = Path(__file__).resolve().parent.parent   # Code_0321 根目录
BIN_DIR     = BASE_DIR / 'iMotion' / 'binarization'
OUTPUT_DIR  = BASE_DIR / 'label_generation' / 'results'
OUTPUT_FILE = OUTPUT_DIR / f'emotion_windows_ws{WINDOW_SIZE_SEC}_st{STEP_SIZE_SEC}_{AGGREGATION_METHOD}.csv'

# ============================================================
#                       核心函数
# ============================================================


def _read_csv_safe(path):
    """兼容不同编码的 CSV 读取。"""
    for enc in ('utf-8', 'gbk', 'latin1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode CSV: {path}")


def _aggregate(series):
    """根据 AGGREGATION_METHOD 对一列数据做聚合。"""
    if AGGREGATION_METHOD == 'median':
        return float(series.median())
    return float(series.mean())   # 默认 'mean'


def window_single_file(df_bin, route, participant, video):
    """
    对单个二值化 CSV 进行滑动时间窗口划分。

    每个窗口内各情绪的聚合值由 AGGREGATION_METHOD 决定：
      'mean'  : 激活帧比例均值（连续值 [0,1]）
      'median': 激活帧中位数（二值数据结果为 0 或 1）

    Parameters
    ----------
    df_bin      : pd.DataFrame  二值化数据（含 relative_time 列）
    route       : int           线路编号
    participant : str           受试者编号（如 'P26'）
    video       : int           视频编号

    Returns
    -------
    pd.DataFrame  包含所有窗口的 DataFrame，若无有效窗口则返回空 DataFrame
    """
    if 'relative_time' not in df_bin.columns:
        tqdm.write(f"[WARN] 缺少 relative_time 列，跳过: route{route}/{participant}/video{video}")
        return pd.DataFrame()

    t = df_bin['relative_time'].values.astype(float)
    if len(t) == 0:
        return pd.DataFrame()

    duration = t[-1]
    windows  = []
    win_id   = 0
    start    = 0.0

    while start < duration - 1e-9:
        end  = start + WINDOW_SIZE_SEC
        mask = (t >= start) & (t < end)

        if mask.sum() >= MIN_FRAMES_PER_WINDOW:
            row = {
                'route':        route,
                'participant':  participant,
                'video':        video,
                'window_id':    win_id,
                'window_start': round(start, 3),
                'window_end':   round(min(end, duration), 3),
            }
            for emo in EMOTION_COLUMNS:
                if emo in df_bin.columns:
                    row[emo] = _aggregate(df_bin.loc[mask, emo])
                else:
                    row[emo] = np.nan
            windows.append(row)
            win_id += 1

        start += STEP_SIZE_SEC

    return pd.DataFrame(windows)


def collect_tasks():
    """扫描二值化目录，返回所有待处理文件的任务列表。"""
    tasks = []
    for route in TARGET_ROUTES:
        route_dir = BIN_DIR / f'route{route}'
        if not route_dir.exists():
            tqdm.write(f"[WARN] 目录不存在（请先运行二值化脚本）: {route_dir}")
            continue
        for pdir in sorted(route_dir.iterdir()):
            if not pdir.is_dir():
                continue
            participant = pdir.name   # 如 'P26'
            for f in sorted(pdir.glob('*.csv')):
                if f.name.startswith('._'):
                    continue
                # 从文件名解析 video 编号，如 'p26-2.csv' → video=2
                stem = f.stem
                try:
                    video = int(stem.split('-')[-1])
                except ValueError:
                    tqdm.write(f"[WARN] 无法解析 video 编号，跳过: {f.name}")
                    continue
                tasks.append((route, participant, video, f))
    return tasks


def process_all():
    tasks = collect_tasks()
    if not tasks:
        print("[INFO] 未找到任何二值化 CSV 文件，请先运行 2. emotion_binarization.py。")
        return

    print(f"\n{'=' * 60}")
    print(f"  情绪时间窗口划分")
    print(f"  窗口大小: {WINDOW_SIZE_SEC}s  |  步长: {STEP_SIZE_SEC}s  |  聚合方式: {AGGREGATION_METHOD}")
    print(f"  情绪列({len(EMOTION_COLUMNS)}): {', '.join(EMOTION_COLUMNS)}")
    print(f"  线路: {TARGET_ROUTES}  |  文件总数: {len(tasks)}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")

    all_windows = []
    fail_count  = 0

    pbar = tqdm(tasks, desc="Windowing", unit="file",
                bar_format="{l_bar}{bar:30}{r_bar}")

    for route, participant, video, fpath in pbar:
        pbar.set_postfix_str(f"route{route}/{participant}/v{video}")
        try:
            df_bin = _read_csv_safe(str(fpath))
            df_win = window_single_file(df_bin, route, participant, video)
            if not df_win.empty:
                all_windows.append(df_win)
        except Exception as e:
            fail_count += 1
            tqdm.write(f"[FAIL] {fpath.name}: {e}")

    pbar.close()

    if not all_windows:
        print("[INFO] 未生成任何窗口数据，请检查二值化文件是否存在。")
        return

    df_all = pd.concat(all_windows, ignore_index=True)

    # 保证列顺序
    meta_cols = ['route', 'participant', 'video', 'window_id',
                 'window_start', 'window_end']
    df_all = df_all[meta_cols + EMOTION_COLUMNS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_FILE, index=False)

    # 统计摘要
    n_routes  = df_all['route'].nunique()
    n_parts   = df_all['participant'].nunique()
    n_videos  = df_all.groupby(['route', 'participant', 'video']).ngroups
    n_windows = len(df_all)

    print(f"\n{'=' * 60}")
    print(f"  完成！  fail={fail_count}")
    print(f"  线路数: {n_routes}  |  受试者数: {n_parts}")
    print(f"  视频段数: {n_videos}  |  总窗口数: {n_windows}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    process_all()

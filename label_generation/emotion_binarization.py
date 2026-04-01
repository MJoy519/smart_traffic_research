"""
情绪二值化脚本
使用动态阈值 + 滞后机制 + 最小持续时间过滤，
将连续的情绪强度数据转换为二值状态（0/1）。
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# ============================================================
#                       全局可控参数
# ============================================================
# -- 二值化算法参数 --
K = 1                     # 阈值系数: threshold = μ + k·σ
HYSTERESIS_DELTA = 0.6      # 滞后参数: threshold_low = μ + (k - delta)·σ
MIN_DURATION = 30           # 最小激活持续数据点数（低于此的激活段被过滤）

# -- 目标情绪列（可按需增减）--
EMOTION_COLUMNS = [
    'Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Confusion'
]

# -- 处理线路选择（可选 1, 2, 3 的任意组合）--
TARGET_ROUTES = [1, 2, 3]

# -- 可视化模式控制 --
# True : 仅进行可视化检验（使用下方 VIS_TEST_* 路径），不执行批量处理
# False: 执行批量二值化处理
VISUALIZATION_ONLY = True

# -- 可视化测试样本（VISUALIZATION_ONLY=True 时使用）--
VIS_TEST_ORIG_CSV = r'c:\Users\Lenovo\Desktop\Smart Traffic\Code_0321\iMotion\raw_processed\route1\P26\p26-2.csv'
VIS_TEST_EMOTIONS = ['Anger']   # 要可视化的情绪列表

# ============================================================
#                       数据路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent   # Code_0321 根目录
RAW_DIR  = BASE_DIR / 'iMotion' / 'raw_processed'
BIN_DIR  = BASE_DIR / 'iMotion' / 'binarization'
VIS_OUTPUT_DIR = BASE_DIR / 'label_generation' / 'figures' / 'binarization'

ROUTE_CONFIGS = [
    {
        'route':      r,
        'input_dir':  RAW_DIR / f'route{r}',
        'output_dir': BIN_DIR / f'route{r}',
    }
    for r in [1, 2, 3]
]

# ============================================================
#                       核心算法
# ============================================================


def binarize_emotion(values, k=K, delta=HYSTERESIS_DELTA, min_duration=MIN_DURATION):
    """
    对单列情绪强度序列进行二值化。

    步骤：
      1. 计算动态阈值 (μ + k·σ) 和滞后去激活阈值 (μ + (k-δ)·σ)
      2. 滞后状态机遍历
      3. 最小持续时间过滤

    Returns
    -------
    binary         : np.ndarray (int)  二值序列
    threshold_high : float             激活阈值
    threshold_low  : float             去激活阈值
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros(len(values), dtype=int), np.nan, np.nan

    mu    = valid.mean()
    sigma = valid.std()
    threshold_high = mu + k * sigma
    threshold_low  = mu + (k - delta) * sigma

    # ---- 滞后状态机 ----
    n      = len(values)
    binary = np.zeros(n, dtype=int)
    active = False

    for i in range(n):
        v = values[i]
        if np.isnan(v):
            binary[i] = int(active)
            continue
        if not active and v > threshold_high:
            active = True
        elif active and v < threshold_low:
            active = False
        binary[i] = int(active)

    # ---- 最小持续时间过滤 ----
    i = 0
    while i < n:
        if binary[i] == 1:
            start = i
            while i < n and binary[i] == 1:
                i += 1
            if (i - start) < min_duration:
                binary[start:i] = 0
        else:
            i += 1

    return binary, threshold_high, threshold_low


# ============================================================
#                       文件处理
# ============================================================


def _read_csv_safe(path):
    """兼容不同编码的 CSV 读取。"""
    for enc in ('utf-8', 'gbk', 'latin1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode CSV: {path}")


def process_single_file(input_path, output_path):
    """读取一个 CSV，对目标情绪列二值化，仅保存二值结果。"""
    df = _read_csv_safe(input_path)

    # 支持两种时间列格式
    if 'relative_time' in df.columns:
        relative_sec = df['relative_time'].values.astype(float)
    elif 'Timestamp' in df.columns:
        ts_ns = df['Timestamp'].values.astype(float)
        relative_sec = (ts_ns - ts_ns[0]) / 1e9
    else:
        relative_sec = np.arange(len(df), dtype=float)

    result = {'relative_time': relative_sec}

    for emo in EMOTION_COLUMNS:
        if emo not in df.columns:
            tqdm.write(f"    [WARN] '{emo}' 列不存在于 {Path(input_path).name}，已填零")
            result[emo] = np.zeros(len(df), dtype=int)
            continue
        vals = df[emo].values.astype(float)
        binary, _, _ = binarize_emotion(vals)
        result[emo] = binary

    df_out = pd.DataFrame(result)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)


def _collect_all_tasks():
    """预扫描所有待处理文件，返回 (route, participant, in_path, out_path) 列表。"""
    tasks = []
    for cfg in ROUTE_CONFIGS:
        if cfg['route'] not in TARGET_ROUTES:
            continue
        in_dir  = cfg['input_dir']
        out_dir = cfg['output_dir']
        if not in_dir.exists():
            print(f"[WARN] 目录不存在: {in_dir}")
            continue
        for pdir in sorted(in_dir.iterdir()):
            if not pdir.is_dir():
                continue
            out_pdir = out_dir / pdir.name
            for f in sorted(pdir.glob('*.csv')):
                if f.name.startswith('._'):
                    continue
                tasks.append((cfg['route'], pdir.name, f, out_pdir / f.name))
    return tasks


def process_all():
    """遍历所有指定线路和受试者，批量二值化（带进度条）。"""
    t_start = time.time()

    tasks = _collect_all_tasks()
    if not tasks:
        print("[INFO] 未找到任何待处理的 CSV 文件。")
        return

    routes_found = sorted(set(t[0] for t in tasks))
    participants  = sorted(set((t[0], t[1]) for t in tasks))

    print(f"\n{'=' * 60}")
    print(f"  情绪二值化  |  K={K}, delta={HYSTERESIS_DELTA}, min_dur={MIN_DURATION}")
    print(f"  情绪列({len(EMOTION_COLUMNS)}): {', '.join(EMOTION_COLUMNS)}")
    print(f"{'=' * 60}")
    print(f"  线路: {routes_found}")
    print(f"  受试者: {len(participants)} 人  |  文件总数: {len(tasks)}")
    print(f"  输出目录: {BIN_DIR}")
    print(f"{'=' * 60}\n")

    success, fail = 0, 0
    cur_route, cur_participant = None, None

    pbar = tqdm(tasks, desc="Total", unit="file",
                bar_format="{l_bar}{bar:30}{r_bar}")

    for route, participant, in_path, out_path in pbar:
        if route != cur_route:
            cur_route = route
            tqdm.write(f"\n>> Route {route}")

        if participant != cur_participant:
            cur_participant = participant
            n_files = sum(1 for t in tasks if t[0] == route and t[1] == participant)
            tqdm.write(f"   [{participant}] {n_files} files")

        pbar.set_postfix_str(f"route{route}/{participant}/{in_path.name}")

        try:
            process_single_file(str(in_path), str(out_path))
            success += 1
        except Exception as e:
            fail += 1
            tqdm.write(f"   [FAIL] {in_path.name}: {e}")

    pbar.close()

    elapsed = time.time() - t_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n{'=' * 60}")
    print(f"  Done!  success={success}  fail={fail}  "
          f"total={success + fail}  time={int(minutes)}m{seconds:.1f}s")
    print(f"  二值化结果保存至: {BIN_DIR}")
    print(f"{'=' * 60}")


# ============================================================
#                       可视化
# ============================================================


def visualize_binarization(original_csv, emotion='Anger', save_path=None):
    """
    对原始 CSV 进行实时二值化，并可视化对比结果（三子图）。

    Parameters
    ----------
    original_csv : str  原始连续值 CSV 路径
    emotion      : str  情绪名称
    save_path    : str  图片保存路径（None → plt.show()）
    """
    df_orig = _read_csv_safe(original_csv)

    if emotion not in df_orig.columns:
        print(f"[WARN] '{emotion}' 列不存在于 {Path(original_csv).name}，跳过")
        return

    values = df_orig[emotion].values.astype(float)
    binary, th_high, th_low = binarize_emotion(values)

    x = np.arange(len(values))

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 3, 1]},
    )

    # ---------- 子图 1: 连续值 & 阈值 ----------
    ax = axes[0]
    ax.plot(x, values, color='#5B9BD5', linewidth=0.7, label=f'{emotion} Value')
    ax.axhline(th_high, color='red',    linestyle='--', alpha=0.8,
               label=f'Threshold high ({th_high:.4f})')
    ax.axhline(th_low,  color='orange', linestyle=':',  alpha=0.8,
               label=f'Threshold low ({th_low:.4f})')
    ax.set_ylabel(f'{emotion} Value')
    ax.set_title(f'{emotion} — Continuous Value & Thresholds', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)

    # ---------- 子图 2: 连续值 & 激活区域 ----------
    ax = axes[1]
    first_region = True
    in_active    = False
    start_idx    = 0
    for i in range(len(binary)):
        if binary[i] == 1 and not in_active:
            start_idx, in_active = i, True
        elif binary[i] == 0 and in_active:
            ax.axvspan(start_idx, i, alpha=0.25, color='#FF6B6B',
                       label='Active Region' if first_region else None)
            first_region, in_active = False, False
    if in_active:
        ax.axvspan(start_idx, len(binary), alpha=0.25, color='#FF6B6B',
                   label='Active Region' if first_region else None)
    ax.plot(x, values, color='#5B9BD5', linewidth=0.7, label=f'{emotion} Value')
    ax.axhline(th_high, color='red', linestyle='--', alpha=0.7)
    ax.set_ylabel(f'{emotion} Value')
    ax.set_title(f'{emotion} — Value & Active Regions', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)

    # ---------- 子图 3: 二值状态 ----------
    ax = axes[2]
    active_idx = np.where(binary == 1)[0]
    if len(active_idx) > 0:
        ax.bar(active_idx, np.ones(len(active_idx)), width=1.0,
               color='#E74C3C', label=f'{emotion} Active')
        ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.15, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Inactive', 'Active'])
    ax.set_xlabel('Data Point Index')
    ax.set_title(f'{emotion} — Binary Active State', fontsize=13)

    plt.suptitle(f'Binarization Check: {Path(original_csv).name}', fontsize=11,
                 y=1.01, color='gray')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图片已保存: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


# ============================================================
#                        主入口
# ============================================================

if __name__ == '__main__':
    if VISUALIZATION_ONLY:
        # ---- 仅进行可视化检验 ----
        print(f"\n>> 可视化检验模式")
        print(f"   测试文件: {VIS_TEST_ORIG_CSV}")
        print(f"   情绪: {VIS_TEST_EMOTIONS}\n")
        for emo in tqdm(VIS_TEST_EMOTIONS, desc="Visualize", unit="emotion"):
            vis_path = VIS_OUTPUT_DIR / f'vis_{Path(VIS_TEST_ORIG_CSV).stem}_{emo}.png'
            visualize_binarization(
                VIS_TEST_ORIG_CSV,
                emotion=emo,
                save_path=str(vis_path),
            )
        print(f"\n  可视化图片已保存至: {VIS_OUTPUT_DIR}")
    else:
        # ---- 批量二值化处理 ----
        process_all()

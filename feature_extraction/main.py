"""
交通特征提取主执行脚本
========================================
逐帧提取三个路线视频的交通特征，并按滑动时间窗口聚合，保存为 CSV。

保存格式：results/{route_num}_{extractor}_{window_size}_{window_step}.csv
列说明：route_num | video_name | window_idx | window_start_sec | window_end_sec | [特征列...]
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║              全局可调参数（修改此处控制提取行为）             ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 推理设备 ──────────────────────────────────────────────────
# "cuda"  : 使用 GPU（需安装 CUDA 版 PyTorch，显存不足时自动报错）
# "cpu"   : 强制使用 CPU（速度较慢，但无需 GPU）
# "auto"  : 自动检测，有 GPU 则用 GPU，否则退回 CPU
DEVICE = "auto"

WINDOW_SIZE  = 3          # 时间窗口大小（秒）
WINDOW_STEP  = 1          # 时间窗口步长（秒）
AGGREGATION  = "median"   # 窗口内聚合方式："median" 或 "mean"

# 选择要使用的特征提取器（可单选/多选）：
#   "segformer" — SegFormer 场景语义特征
#   "yolo"      — YOLO BDD100K 目标检测 + 轨迹特征
#   "yolopv2"   — YOLOPv2 可驾驶区域 + 车道线特征
USE_EXTRACTORS = ["segformer", "yolo", "yolopv2"]   # 全选
# USE_EXTRACTORS = ["yolo"]                          # 仅 YOLO

# 要处理的路线编号（1/2/3 或子集）
TARGET_ROUTES = [2]

# 帧采样间隔（每 N 帧取 1 帧，1 = 逐帧；值越大越快但精度越低）
FRAME_SAMPLE_INTERVAL = 1

# ════════════════════════════════════════════════════════════════

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 将 feature_extraction 目录加入路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from config import (
    SEGFORMER_MODEL_DIR,
    YOLO_MODEL_PATH,
    YOLOPV2_REPO_DIR,
    YOLOPV2_WEIGHTS,
    ROUTE_VIDEO_DIRS,
    VIDEO_EXTENSIONS,
    RESULTS_DIR,
    ROI,
    YOLO_CONF_THRES,
    YOLO_IOU_THRES,
    TTC_THRESHOLD,
)
from feature_extractor.yolo_extractor import YOLOExtractor

# ── 将全局 DEVICE 参数解析为实际设备字符串 ────────────────────────────────────
def _resolve_device(device_param: str) -> str:
    """
    将 DEVICE 参数解析为 "cuda" 或 "cpu"。
    - "auto" : 有可用 GPU 则返回 "cuda"，否则返回 "cpu"
    - "cuda" : 直接返回 "cuda"（运行时若无 GPU 会由各框架报错）
    - "cpu"  : 强制返回 "cpu"
    """
    if device_param == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_param.lower()

_DEVICE = _resolve_device(DEVICE)


def _run_subdir(window_size: float, window_step: float) -> str:
    """
    构造本次运行的结果子目录路径，格式：
      results/ws{window_size}_wt{window_step}_si{frame_sample_interval}/
    相同参数组合的所有提取器结果集中在同一子目录下。
    """
    ws_str = str(window_size).replace(".", "_")
    wt_str = str(window_step).replace(".", "_")
    return os.path.join(
        RESULTS_DIR, f"ws{ws_str}_wt{wt_str}_si{FRAME_SAMPLE_INTERVAL}"
    )


# ==================== 模型加载 ====================

def load_extractors(names: List[str]) -> dict:
    """按需加载各特征提取器，返回 {name: extractor} 字典。"""
    extractors = {}

    if "segformer" in names:
        try:
            from feature_extractor.segformer_extractor import SegFormerExtractor
            extractors["segformer"] = SegFormerExtractor(
                model_dir=SEGFORMER_MODEL_DIR, device=_DEVICE
            )
        except Exception as e:
            print(f"[WARN] SegFormer 加载失败，跳过该提取器: {e}")

    if "yolo" in names:
        try:
            extractors["yolo"] = YOLOExtractor(
                model_path=YOLO_MODEL_PATH,
                device=_DEVICE,
                conf_thres=YOLO_CONF_THRES,
                iou_thres=YOLO_IOU_THRES,
                ttc_threshold=TTC_THRESHOLD,
            )
        except Exception as e:
            print(f"[WARN] YOLO 加载失败，跳过该提取器: {e}")

    if "yolopv2" in names:
        try:
            from feature_extractor.yolopv2_extractor import YOLOPv2Extractor
            extractors["yolopv2"] = YOLOPv2Extractor(
                repo_dir=YOLOPV2_REPO_DIR,
                weights_path=YOLOPV2_WEIGHTS,
                device=_DEVICE,
            )
        except Exception as e:
            print(f"[WARN] YOLOPv2 加载失败，跳过该提取器: {e}")

    return extractors


# ==================== ROI 应用 ====================

def apply_roi(frame: np.ndarray) -> np.ndarray:
    """若 ROI 已启用，裁剪图像至 ROI 区域；否则返回原帧。"""
    if not ROI.get("enabled", False):
        return frame
    h, w = frame.shape[:2]
    x1 = max(0, ROI["x1"])
    y1 = max(0, ROI["y1"])
    x2 = min(w, ROI["x2"])
    y2 = min(h, ROI["y2"])
    return frame[y1:y2, x1:x2]


# ==================== 单帧特征提取 ====================

def extract_one_frame(
    frame: np.ndarray,
    timestamp: float,
    extractors: dict,
) -> Dict[str, dict]:
    """
    对单帧运行所有已加载的特征提取器。

    Returns:
        {extractor_name: feature_dict}
    """
    roi_frame = apply_roi(frame)
    results   = {}

    for name, ext in extractors.items():
        try:
            if name == "yolo":
                results[name] = ext.extract_frame(roi_frame, timestamp)
            else:
                results[name] = ext.extract_frame(roi_frame)
        except Exception as e:
            print(f"  [WARN] {name} 提取帧特征时出错 (t={timestamp:.2f}s): {e}")
            results[name] = {}

    return results


# ==================== 滑动窗口聚合 ====================

def sliding_window_aggregate(
    frame_records: List[dict],
    extractor_name: str,
    window_size: float,
    window_step: float,
    aggregation: str,
    fps: float,
) -> List[dict]:
    """
    将帧级特征列表按滑动时间窗口聚合。

    Args:
        frame_records  : [{"timestamp": t, "features": {...}}, ...]，已按 timestamp 排序
        extractor_name : 提取器名称
        window_size    : 窗口大小（秒）
        window_step    : 步长（秒）
        aggregation    : "median" 或 "mean"
        fps            : 视频帧率（供 YOLO 轨迹计算使用）

    Returns:
        list of window feature dicts
    """
    if not frame_records:
        return []

    agg_fn = np.median if aggregation == "median" else np.mean

    timestamps = [r["timestamp"] for r in frame_records]
    t_start_all = timestamps[0]
    t_end_all   = timestamps[-1]

    windows = []
    win_idx  = 0
    t_win    = t_start_all

    while t_win < t_end_all:
        t_end_win = t_win + window_size

        in_window = [
            r for r in frame_records
            if t_win <= r["timestamp"] < t_end_win
        ]
        if not in_window:
            t_win   += window_step
            win_idx += 1
            continue

        win_feat: dict = {}

        # ── 聚合帧级特征（跳过私有字段）────────────────────────────────────
        all_keys = {k for r in in_window for k in r["features"].keys()
                    if not k.startswith("_")}

        for key in sorted(all_keys):
            vals = [
                r["features"][key]
                for r in in_window
                if key in r["features"] and r["features"][key] is not None
                and not (isinstance(r["features"][key], float) and np.isnan(r["features"][key]))
            ]
            win_feat[key] = float(agg_fn(vals)) if vals else float("nan")

        # ── YOLO 窗口级轨迹特征（在帧聚合后单独计算）───────────────────────
        if extractor_name == "yolo":
            traj_feats = YOLOExtractor.compute_window_traj_features(
                window_frames=[r["features"] for r in in_window],
                ttc_threshold=TTC_THRESHOLD,
                fps=fps,
            )
            win_feat.update(traj_feats)

        windows.append({
            "window_idx":       win_idx,
            "window_start_sec": round(t_win, 3),
            "window_end_sec":   round(min(t_end_win, t_end_all), 3),
            **win_feat,
        })

        t_win   += window_step
        win_idx += 1

    return windows


# ==================== 视频处理 ====================

def process_video(
    video_path: str,
    extractors: dict,
    window_size: float,
    window_step: float,
    aggregation: str,
) -> Dict[str, List[dict]]:
    """
    处理单个视频文件，返回各提取器的窗口特征列表。

    Returns:
        {extractor_name: [window_feat_dict, ...]}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] 无法打开视频: {video_path}")
        return {}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    video_name   = Path(video_path).stem

    print(f"  视频: {Path(video_path).name}  FPS={fps:.1f}  总帧数={total_frames}  时长={duration:.1f}s")

    # 重置 YOLO 追踪器（新视频）
    if "yolo" in extractors:
        extractors["yolo"].reset_tracker()

    # 按提取器收集逐帧记录
    per_ext_records: Dict[str, List[dict]] = {name: [] for name in extractors}

    frame_idx       = 0
    processed_count = 0
    t_process_start = time.time()

    # 进度条：以总帧数为上限，每帧更新一次
    pbar_desc = f"    {'  '.join(extractors.keys())}  |  {video_name[:22]}"
    with tqdm(total=total_frames, desc=pbar_desc, unit="帧",
              ncols=95, dynamic_ncols=True, position=1, leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 帧采样（每 N 帧处理 1 帧）
            if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
                timestamp = frame_idx / fps
                feat_map  = extract_one_frame(frame, timestamp, extractors)
                for name in extractors:
                    per_ext_records[name].append({
                        "timestamp": timestamp,
                        "features":  feat_map.get(name, {}),
                    })
                processed_count += 1

            frame_idx += 1
            pbar.update(1)

            # 在进度条后缀显示实时速度
            if frame_idx % 30 == 0:
                elapsed = time.time() - t_process_start
                fps_proc = processed_count / elapsed if elapsed > 0 else 0
                pbar.set_postfix_str(
                    f"已提取 {processed_count} 帧  {fps_proc:.1f} fps", refresh=False
                )

    cap.release()

    # 各提取器分别做滑动窗口聚合
    results = {}
    for name, records in per_ext_records.items():
        if records:
            windows = sliding_window_aggregate(
                frame_records=records,
                extractor_name=name,
                window_size=window_size,
                window_step=window_step,
                aggregation=aggregation,
                fps=fps,
            )
            results[name] = windows
        else:
            results[name] = []

    elapsed_total = time.time() - t_process_start
    print(f"  处理完成，耗时 {elapsed_total:.1f}s，共生成窗口数（YOLO）: "
          f"{len(results.get('yolo', []))}")
    return results


# ==================== CSV 保存 ====================

def save_results(
    route_num:    int,
    video_name:   str,
    ext_name:     str,
    window_rows:  List[dict],
    window_size:  float,
    window_step:  float,
    all_rows:     Dict[str, List[dict]],
):
    """
    将当前视频的窗口特征追加至对应路线/提取器的 CSV 文件中。
    文件名格式：{route_num}_{extractor}_{window_size}_{window_step}.csv
    """
    if not window_rows:
        return

    sub_dir = _run_subdir(window_size, window_step)
    os.makedirs(sub_dir, exist_ok=True)
    ws_str = str(window_size).replace(".", "_")
    wt_str = str(window_step).replace(".", "_")
    fname  = f"{route_num}_{ext_name}_{ws_str}_{wt_str}.csv"
    fpath  = os.path.join(sub_dir, fname)

    rows = []
    for w in window_rows:
        row = {
            "route_num":        route_num,
            "video_name":       video_name,
        }
        row.update(w)
        rows.append(row)

    df_new = pd.DataFrame(rows)

    # 追加到已有文件（同一路线的不同视频写入同一文件）
    if os.path.exists(fpath):
        df_old = pd.read_csv(fpath)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(fpath, index=False)


# ==================== 启动信息打印 ====================

def _print_startup_info(video_files_per_route: dict):
    """打印详细的运行配置、硬件状态和视频概览。"""
    W = 68

    # ── 设备信息 ───────────────────────────────────────────────────────────
    device_line = _DEVICE.upper()
    if _DEVICE == "cuda":
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            device_line = f"CUDA  ·  {gpu_name}  ·  显存 {gpu_mem:.1f} GB"
        except Exception:
            device_line = "CUDA  (GPU 信息读取失败)"
    if DEVICE == "auto":
        device_line += "  (auto 自动检测)"

    # ── 模型文件状态 ───────────────────────────────────────────────────────
    model_checks = [
        ("SegFormer",  os.path.join(SEGFORMER_MODEL_DIR, "pytorch_model.bin")),
        ("YOLO",       YOLO_MODEL_PATH),
        ("YOLOPv2",    YOLOPV2_WEIGHTS),
    ]

    # ── ROI 描述 ───────────────────────────────────────────────────────────
    if ROI.get("enabled"):
        roi_str = (f"启用  ({ROI['x1']}, {ROI['y1']}) → ({ROI['x2']}, {ROI['y2']})"
                   f"  [{ROI['x2']-ROI['x1']}×{ROI['y2']-ROI['y1']} px]")
    else:
        roi_str = "关闭（全图推理）"

    total_videos = sum(len(v) for v in video_files_per_route.values())

    print("\n" + "═" * W)
    print(f"{'交通特征提取工程  |  main.py':^{W}}")
    print("═" * W)
    print(f"  ┌─ 运行配置 {'─'*(W-13)}")
    print(f"  │  推理设备    : {device_line}")
    print(f"  │  提取器      : {USE_EXTRACTORS}")
    print(f"  │  目标路线    : {TARGET_ROUTES}")
    print(f"  │  帧采样间隔  : 每 {FRAME_SAMPLE_INTERVAL} 帧取 1 帧"
          f"  （约实际帧率 1/{FRAME_SAMPLE_INTERVAL}）")
    print(f"  │  窗口大小    : {WINDOW_SIZE}s  |  步长: {WINDOW_STEP}s"
          f"  |  聚合: {AGGREGATION}")
    print(f"  │  ROI         : {roi_str}")
    print(f"  ├─ 模型文件 {'─'*(W-13)}")
    for name, path in model_checks:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 ** 2
            print(f"  │  ✓  {name:<10}: {Path(path).name}  ({size_mb:.1f} MB)")
        else:
            print(f"  │  ✗  {name:<10}: 未找到 → {path}")
    print(f"  ├─ 视频概览  共 {total_videos} 个视频 {'─'*(W-22)}")
    for route_num, files in sorted(video_files_per_route.items()):
        names = "  ".join(Path(f).name for f in files[:4])
        suffix = f"  … 共{len(files)}个" if len(files) > 4 else ""
        print(f"  │  路线 {route_num} ({len(files)} 个): {names}{suffix}")
    print(f"  └{'─'*(W-3)}")
    print("═" * W + "\n")


# ==================== 主流程 ====================

def main():
    # ── 预扫描视频文件（用于启动信息展示）────────────────────────────────
    video_files_per_route: dict = {}
    for route_num in TARGET_ROUTES:
        route_dir = ROUTE_VIDEO_DIRS.get(route_num)
        if route_dir and os.path.isdir(route_dir):
            files = sorted([
                os.path.join(route_dir, f)
                for f in os.listdir(route_dir)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
            ])
            if files:
                video_files_per_route[route_num] = files

    # ── 打印启动配置信息 ─────────────────────────────────────────────────
    _print_startup_info(video_files_per_route)

    if not video_files_per_route:
        print("[ERROR] 未找到任何视频文件，请检查 TARGET_ROUTES 和视频目录")
        return

    # ── 加载模型 ─────────────────────────────────────────────────────────
    print("[1/2] 加载特征提取模型 ...")
    extractors = load_extractors(USE_EXTRACTORS)
    if not extractors:
        print("[ERROR] 没有可用的特征提取器，请先运行 download_models.py")
        return
    print(f"  已加载提取器: {list(extractors.keys())}\n")

    # ── 逐路线逐视频处理 ─────────────────────────────────────────────────
    print("[2/2] 开始视频特征提取 ...")
    total_videos = sum(len(v) for v in video_files_per_route.values())

    # 外层进度条：所有视频总进度
    with tqdm(total=total_videos, desc="总进度", unit="个视频",
              ncols=80, position=0, colour="green") as pbar_total:

        for route_num, video_files in sorted(video_files_per_route.items()):
            tqdm.write(f"\n{'─'*65}")
            tqdm.write(f"  路线 {route_num}  共 {len(video_files)} 个视频")
            tqdm.write(f"{'─'*65}")

            # 清空该路线旧结果文件（重新生成）
            _sub = _run_subdir(WINDOW_SIZE, WINDOW_STEP)
            os.makedirs(_sub, exist_ok=True)
            for ext_name in extractors:
                ws_str = str(WINDOW_SIZE).replace(".", "_")
                wt_str = str(WINDOW_STEP).replace(".", "_")
                fpath  = os.path.join(_sub,
                                      f"{route_num}_{ext_name}_{ws_str}_{wt_str}.csv")
                if os.path.exists(fpath):
                    os.remove(fpath)

            for vid_idx, video_path in enumerate(video_files):
                video_name = Path(video_path).stem
                tqdm.write(f"\n  [{vid_idx+1}/{len(video_files)}] "
                           f"处理: {Path(video_path).name}")

                try:
                    video_results = process_video(
                        video_path=video_path,
                        extractors=extractors,
                        window_size=WINDOW_SIZE,
                        window_step=WINDOW_STEP,
                        aggregation=AGGREGATION,
                    )
                except Exception as e:
                    tqdm.write(f"  [ERROR] 处理视频失败: {e}")
                    traceback.print_exc()
                    pbar_total.update(1)
                    continue

                # 保存每个提取器的结果
                for ext_name, window_rows in video_results.items():
                    save_results(
                        route_num=route_num,
                        video_name=video_name,
                        ext_name=ext_name,
                        window_rows=window_rows,
                        window_size=WINDOW_SIZE,
                        window_step=WINDOW_STEP,
                        all_rows=video_results,
                    )
                    ws_s = str(WINDOW_SIZE).replace('.','_')
                    wt_s = str(WINDOW_STEP).replace('.','_')
                    tqdm.write(
                        f"    ✓ {ext_name}: {len(window_rows)} 个窗口  → "
                        f"ws{ws_s}_wt{wt_s}_si{FRAME_SAMPLE_INTERVAL}/"
                        f"{route_num}_{ext_name}_{ws_s}_{wt_s}.csv"
                    )

                pbar_total.update(1)

    _sub = _run_subdir(WINDOW_SIZE, WINDOW_STEP)
    print("\n" + "=" * 65)
    print("  特征提取完成！结果保存至:")
    print(f"  {_sub}")
    print("=" * 65)

    # ── 打印结果摘要 ─────────────────────────────────────────────────────────
    print("\n  生成文件列表:")
    if os.path.isdir(_sub):
        for f in sorted(os.listdir(_sub)):
            if f.endswith(".csv"):
                fpath = os.path.join(_sub, f)
                try:
                    df = pd.read_csv(fpath)
                    print(f"    {f}  →  {len(df)} 行 × {len(df.columns)} 列")
                except Exception:
                    print(f"    {f}")


if __name__ == "__main__":
    main()

"""
特征提取器检查脚本 — extractor_test.py
========================================
对选定的特征提取器进行功能验证，输出：
  1. 特征 CSV 文件（所有测试窗口特征）
  2. 逐帧可视化图（原图 / ROI / 模型输出 / 特征值对比）
  3. 特征时序折线图（滑动窗口特征随时间变化）

所有结果保存至 TEST_OUTPUT_DIR 目录。
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║                  全局可调参数                                  ║
# ╚══════════════════════════════════════════════════════════════╝

# 要测试的特征提取器（单选）："segformer" / "yolo" / "yolopv2"
TEST_EXTRACTOR = "yolopv2"

# 测试视频路径（相对于 feature_extraction/ 或绝对路径）
TEST_VIDEO_PATH = r"../videos/3/CUT 1.mp4"

# 从视频中提取多少秒进行测试（0 = 全视频）
TEST_DURATION_SEC = 0

# 可视化抽取的帧数（均匀采样）
VIS_N_FRAMES = 2

# 滑动窗口参数
WINDOW_SIZE  = 3      # 窗口大小（秒）
WINDOW_STEP  = 1      # 步长（秒）
AGGREGATION  = "median"  # "median" 或 "mean"

# 推理设备："auto" / "cuda" / "cpu"
DEVICE = "auto"

# 帧采样间隔（每 N 帧取 1 帧，建议 3~10）
FRAME_SAMPLE_INTERVAL = 15

# 测试结果输出目录（在 feature_extraction/ 下）
TEST_OUTPUT_DIR = "test_output"

# ════════════════════════════════════════════════════════════════

import os
import sys
import math
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── 路径初始化 ─────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from config import (
    SEGFORMER_MODEL_DIR, YOLO_MODEL_PATH, YOLOPV2_REPO_DIR, YOLOPV2_WEIGHTS,
    ROI, YOLO_CONF_THRES, YOLO_IOU_THRES, TTC_THRESHOLD, CITYSCAPES_LABELS,
)

OUTPUT_DIR = os.path.join(_BASE_DIR, TEST_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cityscapes 标准配色（19 类）─────────────────────────────────────────────────
CITYSCAPES_COLORS = np.array([
    [128,  64, 128],  # 0  road
    [244,  35, 232],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [102, 102, 156],  # 3  wall
    [190, 153, 153],  # 4  fence
    [153, 153, 153],  # 5  pole
    [250, 170,  30],  # 6  traffic light
    [220, 220,   0],  # 7  traffic sign
    [107, 142,  35],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [ 70, 130, 180],  # 10 sky
    [220,  20,  60],  # 11 person
    [255,   0,   0],  # 12 rider
    [  0,   0, 142],  # 13 car
    [  0,   0,  70],  # 14 truck
    [  0,  60, 100],  # 15 bus
    [  0,  80, 100],  # 16 train
    [  0,   0, 230],  # 17 motorcycle
    [119,  11,  32],  # 18 bicycle
], dtype=np.uint8)

# ── BDD100K YOLO 类别配色 ────────────────────────────────────────────────────────
BDD_COLORS = [
    (255,  60,  60),  # pedestrian
    (255, 140,   0),  # rider
    ( 30, 144, 255),  # car
    (  0, 200, 100),  # truck
    (148,   0, 211),  # bus
    (255, 215,   0),  # train
    (  0, 206, 209),  # motorcycle
    (255, 105, 180),  # bicycle
    ( 50, 205,  50),  # traffic light
    (255, 165,   0),  # traffic sign
]


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device(param: str) -> str:
    if param == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return param.lower()


def load_test_video(video_path: str) -> cv2.VideoCapture:
    """加载视频，若路径为相对路径则基于 feature_extraction/ 解析。"""
    if not os.path.isabs(video_path):
        video_path = os.path.normpath(os.path.join(_BASE_DIR, video_path))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    return cap


def sample_frames(
    cap: cv2.VideoCapture,
    duration_sec: float,
    sample_interval: int,
) -> List[Tuple[int, float, np.ndarray]]:
    """按帧采样间隔顺序读取指定时长内的所有帧。返回 [(frame_idx, timestamp, frame), ...]。"""
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame    = int(duration_sec * fps) if duration_sec > 0 else total_frames
    end_frame    = min(end_frame, total_frames)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    with tqdm(total=end_frame, desc="  读取视频帧", unit="帧",
              ncols=80, leave=False) as pbar:
        while idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_interval == 0:
                frames.append((idx, idx / fps, frame))
            idx += 1
            pbar.update(1)
    return frames


def pick_vis_frames(
    all_frames: List[Tuple],
    n: int,
) -> List[Tuple]:
    """从所有帧中随机挑选 n 帧用于可视化（按时序排列）。"""
    if len(all_frames) <= n:
        return all_frames
    rng     = np.random.default_rng()
    indices = sorted(rng.choice(len(all_frames), size=n, replace=False).tolist())
    return [all_frames[i] for i in indices]


def apply_roi(frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple]]:
    """
    应用 ROI 裁剪。
    返回 (roi_frame, roi_rect_xyxy) — 若 ROI 未启用，rect 为 None（整图）。
    """
    h, w = frame.shape[:2]
    if not ROI.get("enabled", False):
        return frame.copy(), (0, 0, w, h)
    x1 = max(0, ROI["x1"])
    y1 = max(0, ROI["y1"])
    x2 = min(w, ROI["x2"])
    y2 = min(h, ROI["y2"])
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def draw_roi_overlay(frame: np.ndarray) -> np.ndarray:
    """在图上绘制 ROI 区域：ROI 内正常显示，ROI 外半透明暗化。"""
    out   = frame.copy()
    h, w  = frame.shape[:2]
    if not ROI.get("enabled", False):
        return out
    x1 = max(0, ROI["x1"]);  x2 = min(w, ROI["x2"])
    y1 = max(0, ROI["y1"]);  y2 = min(h, ROI["y2"])
    # 暗化 ROI 外区域
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    out[mask == 0] = (out[mask == 0] * 0.35).astype(np.uint8)
    # 绘制 ROI 边框
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 230, 80), 3)
    cv2.putText(out, "ROI", (x1 + 6, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 80), 2, cv2.LINE_AA)
    return out


def sliding_window_aggregate(
    frame_records: List[dict],
    window_size: float,
    window_step: float,
    agg: str,
    fps: float,
) -> List[dict]:
    """滑动窗口聚合，返回窗口特征列表。"""
    if not frame_records:
        return []
    agg_fn   = np.median if agg == "median" else np.mean
    t_start  = frame_records[0]["timestamp"]
    t_end    = frame_records[-1]["timestamp"]
    windows, win_idx, t_win = [], 0, t_start

    while t_win < t_end:
        t_win_end = t_win + window_size
        in_win    = [r for r in frame_records if t_win <= r["timestamp"] < t_win_end]
        if in_win:
            all_keys = {k for r in in_win for k in r["features"] if not k.startswith("_")}
            win_feat = {}
            for key in sorted(all_keys):
                vals = [r["features"][key] for r in in_win
                        if key in r["features"]
                        and r["features"][key] is not None
                        and not (isinstance(r["features"][key], float)
                                 and math.isnan(r["features"][key]))]
                win_feat[key] = float(agg_fn(vals)) if vals else float("nan")

            # YOLO 轨迹特征
            if TEST_EXTRACTOR == "yolo":
                from feature_extractor.yolo_extractor import YOLOExtractor
                traj = YOLOExtractor.compute_window_traj_features(
                    [r["features"] for r in in_win], ttc_threshold=TTC_THRESHOLD, fps=fps
                )
                win_feat.update(traj)

            windows.append({
                "window_idx":       win_idx,
                "window_start_sec": round(t_win, 3),
                "window_end_sec":   round(min(t_win_end, t_end), 3),
                **win_feat,
            })
        t_win   += window_step
        win_idx += 1
    return windows


# ══════════════════════════════════════════════════════════════════════════════
# SegFormer 测试 + 可视化
# ══════════════════════════════════════════════════════════════════════════════

def run_segformer_test(device: str):
    tqdm.write("\n[SegFormer] 开始测试 ...")
    from feature_extractor.segformer_extractor import SegFormerExtractor
    ext = SegFormerExtractor(model_dir=SEGFORMER_MODEL_DIR, device=device)

    cap        = load_test_video(TEST_VIDEO_PATH)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_frames = sample_frames(cap, TEST_DURATION_SEC, FRAME_SAMPLE_INTERVAL)
    cap.release()
    tqdm.write(f"  共采集 {len(all_frames)} 帧，FPS={fps:.1f}")

    # ── 逐帧提取（带 tqdm 进度条）────────────────────────────────────────────
    records = []
    for fidx, ts, frame in tqdm(all_frames, desc="  SegFormer 推理",
                                 unit="帧", ncols=80, dynamic_ncols=True):
        roi_frame, _ = apply_roi(frame)
        feats = ext.extract_frame(roi_frame)
        records.append({"timestamp": ts, "features": feats})

    # ── 滑动窗口聚合 ──────────────────────────────────────────────────────────
    windows = sliding_window_aggregate(records, WINDOW_SIZE, WINDOW_STEP, AGGREGATION, fps)
    df = pd.DataFrame(windows)
    csv_path = os.path.join(OUTPUT_DIR, "segformer_features.csv")
    df.to_csv(csv_path, index=False)
    tqdm.write(f"  特征 CSV 已保存: {csv_path}  ({len(df)} 个窗口)")

    # ── 可视化（若干帧，带进度条）────────────────────────────────────────────
    vis_frames = pick_vis_frames(all_frames, VIS_N_FRAMES)
    for fidx, ts, frame in tqdm(vis_frames, desc="  生成可视化",
                                 unit="帧", ncols=80, leave=False):
        roi_frame, _ = apply_roi(frame)
        seg_map = ext._infer_segmap(roi_frame)
        feats   = ext.extract_frame(roi_frame)
        _vis_segformer_frame(frame, roi_frame, seg_map, feats, fidx, ts)

    # ── 时序图 ────────────────────────────────────────────────────────────────
    if len(df) >= 2:
        _plot_timeseries(df, "segformer", "SegFormer 特征时序图")
    tqdm.write("[SegFormer] 测试完成")


def _vis_segformer_frame(orig, roi_frame, seg_map, feats, fidx, ts):
    """SegFormer 单帧可视化（2行×3列）。"""
    # ── 生成分割伪彩图 ─────────────────────────────────────────────────────────
    seg_rgb = CITYSCAPES_COLORS[np.clip(seg_map, 0, 18)]  # (H,W,3) uint8

    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    roi_rgb  = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

    # 分割叠加：0.55 原图 + 0.45 伪彩
    overlay_rgb = (roi_rgb * 0.55 + seg_rgb * 0.45).astype(np.uint8)

    # ROI 标注版原图
    roi_annotated = cv2.cvtColor(draw_roi_overlay(orig), cv2.COLOR_BGR2RGB)

    # ── 特征 bar chart 数据（仅 ratio 类特征）──────────────────────────────────
    bar_keys   = ["road_coverage","sidewalk_coverage","building_coverage",
                  "sky_visibility","green_coverage","wall_fence_coverage",
                  "building_oppression","openness_index"]
    bar_vals   = [feats.get(k, 0) for k in bar_keys]
    bar_labels = ["road","sidewalk","building","sky","green",
                  "wall/fence","oppression","openness"]
    bar_colors = ["#7B5EA7","#E91E8C","#546E7A","#1976D2","#388E3C",
                  "#8D6E63","#C62828","#039BE5"]

    # ── 图布局 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#F5F5F5")
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.28)

    # R0C0: 原图 + ROI 标注
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(roi_annotated)
    ax00.set_title(f"原始帧  idx={fidx}  t={ts:.2f}s\n{'ROI 已启用' if ROI['enabled'] else '全图推理'}",
                   color="#212121", fontsize=11)
    ax00.axis("off")

    # R0C1: 语义分割图
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(seg_rgb)
    ax01.set_title("语义分割图（Cityscapes）", color="#212121", fontsize=11)
    ax01.axis("off")

    # R0C2: 特征条形图
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.set_facecolor("#FFFFFF")
    bars = ax02.barh(bar_labels, bar_vals, color=bar_colors, edgecolor="#BDBDBD", linewidth=0.6)
    ax02.set_xlim(0, 1)
    ax02.set_xlabel("占比 / 指数值", color="#424242")
    ax02.set_title("SegFormer 场景特征", color="#212121", fontsize=11)
    ax02.tick_params(colors="#424242")
    for spine in ax02.spines.values():
        spine.set_edgecolor("#BDBDBD")
    for bar, val in zip(bars, bar_vals):
        ax02.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                  f"{val:.3f}", va="center", color="#212121", fontsize=8.5)

    # R1C0: 分割叠加图
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(overlay_rgb)
    ax10.set_title("分割叠加（原图 55% + 伪彩 45%）", color="#212121", fontsize=11)
    ax10.axis("off")

    # R1C1: 类别图例
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.set_facecolor("#FFFFFF")
    ax11.axis("off")
    vis_classes = [0,1,2,3,4,8,10,11,13,14,15]  # 仅展示常见类别
    patches = [
        mpatches.Patch(color=CITYSCAPES_COLORS[c] / 255.0,
                       label=f"[{c}] {CITYSCAPES_LABELS.get(c,'?')}")
        for c in vis_classes
    ]
    ax11.legend(handles=patches, loc="center", ncol=2, framealpha=0.9,
                labelcolor="#212121", fontsize=9.5,
                facecolor="#FFFFFF", edgecolor="#BDBDBD")
    ax11.set_title("类别图例", color="#212121", fontsize=11)

    # R1C2: 特征数值表
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.set_facecolor("#FFFFFF")
    ax12.axis("off")
    all_feat_keys = list(feats.keys())
    table_data = [[k, f"{feats[k]:.4f}"] for k in all_feat_keys]
    tbl = ax12.table(cellText=table_data,
                     colLabels=["特征名", "值"],
                     loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#E3F2FD" if r == 0 else "#FFFFFF")
        cell.set_text_props(color="#212121")
        cell.set_edgecolor("#BDBDBD")
    ax12.set_title("特征数值", color="#212121", fontsize=11)

    fig.suptitle(
        f"SegFormer 特征可视化  |  帧 {fidx}  t={ts:.2f}s  |  {Path(TEST_VIDEO_PATH).name}",
        color="#212121", fontsize=13, fontweight="bold",
    )
    _save_fig(fig, f"segformer_frame_{fidx:05d}.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# YOLO 测试 + 可视化
# ══════════════════════════════════════════════════════════════════════════════

def run_yolo_test(device: str):
    tqdm.write("\n[YOLO] 开始测试 ...")
    from feature_extractor.yolo_extractor import YOLOExtractor
    ext = YOLOExtractor(
        model_path=YOLO_MODEL_PATH, device=device,
        conf_thres=YOLO_CONF_THRES, iou_thres=YOLO_IOU_THRES,
        ttc_threshold=TTC_THRESHOLD,
    )

    cap        = load_test_video(TEST_VIDEO_PATH)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_frames = sample_frames(cap, TEST_DURATION_SEC, FRAME_SAMPLE_INTERVAL)
    cap.release()
    ext.reset_tracker()
    tqdm.write(f"  共采集 {len(all_frames)} 帧，FPS={fps:.1f}")

    # ── 逐帧提取（带 tqdm 进度条）────────────────────────────────────────────
    records = []
    pbar = tqdm(all_frames, desc="  YOLO 推理", unit="帧",
                ncols=80, dynamic_ncols=True)
    for fidx, ts, frame in pbar:
        roi_frame, _ = apply_roi(frame)
        feats = ext.extract_frame(roi_frame, ts)
        records.append({"timestamp": ts, "features": feats})
        n_obj = feats.get("total_object_count", 0)
        pbar.set_postfix_str(f"t={ts:.1f}s  目标={n_obj}", refresh=False)

    # ── 滑动窗口聚合 ──────────────────────────────────────────────────────────
    windows = sliding_window_aggregate(records, WINDOW_SIZE, WINDOW_STEP, AGGREGATION, fps)
    df = pd.DataFrame(windows)
    csv_path = os.path.join(OUTPUT_DIR, "yolo_features.csv")
    df.to_csv(csv_path, index=False)
    tqdm.write(f"  特征 CSV 已保存: {csv_path}  ({len(df)} 个窗口)")

    # ── 可视化（若干帧，带进度条）────────────────────────────────────────────
    vis_frames = pick_vis_frames(all_frames, VIS_N_FRAMES)
    for fidx, ts, frame in tqdm(vis_frames, desc="  生成可视化",
                                 unit="帧", ncols=80, leave=False):
        roi_frame, roi_rect = apply_roi(frame)
        results = ext.model.predict(
            source=roi_frame, conf=YOLO_CONF_THRES, iou=YOLO_IOU_THRES, verbose=False
        )
        result  = results[0]
        feats   = ext.extract_frame(roi_frame, ts)
        _vis_yolo_frame(frame, roi_frame, roi_rect, result, feats,
                        ext.model.names, fidx, ts)

    # ── 时序图 ────────────────────────────────────────────────────────────────
    if len(df) >= 2:
        _plot_timeseries(df, "yolo", "YOLO BDD100K 特征时序图")
    tqdm.write("[YOLO] 测试完成")


def _draw_yolo_detections(frame: np.ndarray, result, class_names: dict) -> np.ndarray:
    """在帧上绘制 YOLO 检测框，返回 RGB 图像。"""
    img = frame.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for box in result.boxes:
        cid   = int(box.cls[0])
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        color = BDD_COLORS[cid % len(BDD_COLORS)]
        bgr   = (color[2], color[1], color[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        label = f"{class_names.get(cid, cid)} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 3, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 2, ty + 2), bgr, -1)
        cv2.putText(img, label, (x1 + 1, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _vis_yolo_frame(orig, roi_frame, roi_rect, result, feats, class_names, fidx, ts):
    """YOLO 单帧可视化（2行×3列）。"""
    orig_rgb     = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    roi_annotated= cv2.cvtColor(draw_roi_overlay(orig), cv2.COLOR_BGR2RGB)
    det_rgb      = _draw_yolo_detections(roi_frame, result, class_names)

    # 统计各类数量
    cls_counts = {}
    if result.boxes and len(result.boxes) > 0:
        for box in result.boxes:
            cid  = int(box.cls[0])
            name = class_names.get(cid, str(cid))
            cls_counts[name] = cls_counts.get(name, 0) + 1

    # 帧级标量特征（非私有、非 NaN）
    scalar_feats = {k: v for k, v in feats.items()
                    if not k.startswith("_") and not (isinstance(v, float) and math.isnan(v))}

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#F5F5F5")
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.28)

    # R0C0: 原图 + ROI 标注
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(roi_annotated)
    ax.set_title(f"原始帧  idx={fidx}  t={ts:.2f}s\n{'ROI 已启用' if ROI['enabled'] else '全图推理'}",
                 color="#212121", fontsize=11)
    ax.axis("off")

    # R0C1: 检测结果
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(det_rgb)
    ax.set_title(f"YOLO 检测结果（共 {len(result.boxes) if result.boxes else 0} 个目标）",
                 color="#212121", fontsize=11)
    ax.axis("off")

    # R0C2: 类别计数柱状图
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor("#FFFFFF")
    if cls_counts:
        names = list(cls_counts.keys())
        cnts  = [cls_counts[n] for n in names]
        cidxs = [next((i for i, n in class_names.items() if n == nm), 0) for nm in names]
        colors= [[c/255 for c in BDD_COLORS[i % len(BDD_COLORS)]] for i in cidxs]
        ax.bar(names, cnts, color=colors, edgecolor="#BDBDBD", linewidth=0.6)
        ax.set_ylabel("数量", color="#424242")
        for i, c in enumerate(cnts):
            ax.text(i, c + 0.05, str(c), ha="center", color="#212121", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="#424242")
    else:
        ax.text(0.5, 0.5, "本帧无检测结果", ha="center", va="center",
                transform=ax.transAxes, color="#9E9E9E", fontsize=12)
    ax.set_title("各类别目标数量", color="#212121", fontsize=11)
    ax.tick_params(colors="#424242")
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values(): sp.set_edgecolor("#BDBDBD")

    # R1C0: 带 bbox 面积的散点图（动态目标位置分布）
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor("#FAFAFA")
    h, w = roi_frame.shape[:2]
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_title("目标空间分布（中心点 + 框大小）", color="#212121", fontsize=11)
    if result.boxes and len(result.boxes) > 0:
        for box in result.boxes:
            cid  = int(box.cls[0])
            x1b, y1b, x2b, y2b = box.xyxy[0].tolist()
            cx  = (x1b + x2b) / 2; cy = (y1b + y2b) / 2
            area= (x2b - x1b) * (y2b - y1b)
            s   = max(20, area / (h * w) * 2000)
            c   = [v/255 for v in BDD_COLORS[cid % len(BDD_COLORS)]]
            ax.scatter(cx, cy, s=s, color=c, alpha=0.75, edgecolors="#424242", linewidths=0.5)
    ax.set_xlabel("X 像素", color="#424242"); ax.set_ylabel("Y 像素", color="#424242")
    ax.tick_params(colors="#424242")
    for sp in ax.spines.values(): sp.set_edgecolor("#BDBDBD")

    # R1C1: 帧级特征数值表
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor("#FFFFFF"); ax.axis("off")
    count_keys  = [k for k in scalar_feats if "count" in k or "ratio" in k]
    table_items = [[k, f"{scalar_feats[k]:.4f}"] for k in sorted(count_keys)]
    if table_items:
        tbl = ax.table(cellText=table_items, colLabels=["特征名", "帧级值"],
                       loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("#E3F2FD" if r == 0 else "#FFFFFF")
            cell.set_text_props(color="#212121"); cell.set_edgecolor("#BDBDBD")
    ax.set_title("帧级检测特征", color="#212121", fontsize=11)

    # R1C2: 动态目标面积 + 大型车占比饼图
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor("#FFFFFF")
    car_c  = feats.get("car_count", 0)
    tb_c   = feats.get("truck_bus_count", 0)
    pers_c = feats.get("person_count", 0)
    cyc_c  = feats.get("cyclist_motorcycle_count", 0)
    sign_c = feats.get("traffic_sign_count", 0)
    pie_vals   = [car_c, tb_c, pers_c, cyc_c, sign_c]
    pie_labels = ["car", "truck/bus", "person", "cyclist", "sign/light"]
    pie_colors = ["#1565C0","#2E7D32","#C62828","#F9A825","#6A1B9A"]
    nonzero    = [(v, l, c) for v, l, c in zip(pie_vals, pie_labels, pie_colors) if v > 0]
    if nonzero:
        vals, labs, cols = zip(*nonzero)
        wedges, texts, autotexts = ax.pie(
            vals, labels=labs, colors=cols,
            autopct="%1.0f%%", startangle=90,
            textprops={"color": "#212121", "fontsize": 9},
        )
        for at in autotexts: at.set_color("#212121")
    else:
        ax.text(0.5, 0.5, "本帧无目标", ha="center", va="center",
                transform=ax.transAxes, color="#9E9E9E", fontsize=12)
    ax.set_title("目标类别构成", color="#212121", fontsize=11)

    fig.suptitle(
        f"YOLO BDD100K 特征可视化  |  帧 {fidx}  t={ts:.2f}s  |  {Path(TEST_VIDEO_PATH).name}",
        color="#212121", fontsize=13, fontweight="bold",
    )
    _save_fig(fig, f"yolo_frame_{fidx:05d}.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# YOLOPv2 测试 + 可视化
# ══════════════════════════════════════════════════════════════════════════════

def run_yolopv2_test(device: str):
    tqdm.write("\n[YOLOPv2] 开始测试 ...")
    from feature_extractor.yolopv2_extractor import YOLOPv2Extractor
    ext = YOLOPv2Extractor(
        repo_dir=YOLOPV2_REPO_DIR, weights_path=YOLOPV2_WEIGHTS, device=device
    )

    cap        = load_test_video(TEST_VIDEO_PATH)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_frames = sample_frames(cap, TEST_DURATION_SEC, FRAME_SAMPLE_INTERVAL)
    cap.release()
    tqdm.write(f"  共采集 {len(all_frames)} 帧，FPS={fps:.1f}")

    # ── 逐帧提取（带 tqdm 进度条）────────────────────────────────────────────
    records = []
    pbar = tqdm(all_frames, desc="  YOLOPv2 推理", unit="帧",
                ncols=80, dynamic_ncols=True)
    for fidx, ts, frame in pbar:
        roi_frame, _ = apply_roi(frame)
        feats = ext.extract_frame(roi_frame)
        records.append({"timestamp": ts, "features": feats})
        da = feats.get("drivable_coverage", float("nan"))
        if not math.isnan(da):
            pbar.set_postfix_str(f"t={ts:.1f}s  可驾驶={da:.2f}", refresh=False)

    windows = sliding_window_aggregate(records, WINDOW_SIZE, WINDOW_STEP, AGGREGATION, fps)
    df = pd.DataFrame(windows)
    csv_path = os.path.join(OUTPUT_DIR, "yolopv2_features.csv")
    df.to_csv(csv_path, index=False)
    tqdm.write(f"  特征 CSV 已保存: {csv_path}  ({len(df)} 个窗口)")

    # ── 可视化（若干帧，带进度条）────────────────────────────────────────────
    vis_frames = pick_vis_frames(all_frames, VIS_N_FRAMES)
    for fidx, ts, frame in tqdm(vis_frames, desc="  生成可视化",
                                 unit="帧", ncols=80, leave=False):
        roi_frame, _ = apply_roi(frame)
        h, w      = roi_frame.shape[:2]
        img_input = ext._preprocess(roi_frame)
        da_mask, ll_mask = ext._infer(img_input, h, w)
        feats  = ext.extract_frame(roi_frame)
        _vis_yolopv2_frame(frame, roi_frame, da_mask, ll_mask, feats, fidx, ts)

    if len(df) >= 2:
        _plot_timeseries(df, "yolopv2", "YOLOPv2 特征时序图")
    tqdm.write("[YOLOPv2] 测试完成")


def _vis_yolopv2_frame(orig, roi_frame, da_mask, ll_mask, feats, fidx, ts):
    """YOLOPv2 单帧可视化（2行×3列）。"""
    roi_rgb  = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    roi_anno = cv2.cvtColor(draw_roi_overlay(orig), cv2.COLOR_BGR2RGB)
    h, w     = roi_frame.shape[:2]

    # 可驾驶区域叠加（绿色 50%）
    da_overlay = roi_rgb.copy()
    if da_mask is not None:
        da_layer = np.zeros_like(roi_rgb)
        da_layer[da_mask > 0] = [0, 210, 80]
        da_overlay = (roi_rgb * 0.55 + da_layer * 0.45).astype(np.uint8)

    # 车道线叠加（黄色）
    ll_overlay = roi_rgb.copy()
    if ll_mask is not None:
        for ch, val in [(0, 255), (1, 215), (2, 0)]:
            ll_overlay[:, :, ch][ll_mask > 0] = val

    # 联合叠加
    combined = da_overlay.copy()
    if ll_mask is not None:
        for ch, val in [(0, 255), (1, 215), (2, 0)]:
            combined[:, :, ch][ll_mask > 0] = val

    # 可驾驶宽度曲线（按行统计）
    width_profile = []
    if da_mask is not None:
        for row in da_mask:
            nz = np.where(row > 0)[0]
            width_profile.append((nz[-1] - nz[0] + 1) / w if len(nz) >= 2 else 0.0)

    # ── 布局 ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#F5F5F5")
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.28)

    # R0C0: 原图 + ROI 标注
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(roi_anno)
    ax.set_title(f"原始帧  idx={fidx}  t={ts:.2f}s\n{'ROI 已启用' if ROI['enabled'] else '全图推理'}",
                 color="#212121", fontsize=11)
    ax.axis("off")

    # R0C1: 可驾驶区域叠加
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(da_overlay)
    da_pct = feats.get("drivable_coverage", float("nan"))
    ax.set_title(f"可驾驶区域（绿色）  占比={da_pct:.3f}" if not math.isnan(da_pct)
                 else "可驾驶区域（无数据）", color="#212121", fontsize=11)
    ax.axis("off")

    # R0C2: 车道线叠加
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(ll_overlay)
    lc  = feats.get("lane_count_visible", 0)
    lvis= feats.get("lane_marking_visibility", float("nan"))
    ax.set_title(f"车道线（黄色）  检测数={lc}  可见性={lvis:.4f}" if not math.isnan(lvis)
                 else f"车道线（黄色）  检测数={lc}", color="#212121", fontsize=11)
    ax.axis("off")

    # R1C0: 联合叠加
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(combined)
    offset = feats.get("lane_offset", float("nan"))
    ax.set_title(f"联合叠加  车道偏移={offset:.3f}" if not math.isnan(offset)
                 else "联合叠加", color="#212121", fontsize=11)
    ax.axis("off")

    # R1C1: 可驾驶宽度曲线（按图像行）
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor("#FAFAFA")
    if width_profile:
        y_vals = np.linspace(0, 1, len(width_profile))
        ax.plot(width_profile, y_vals, color="#0288D1", linewidth=2)
        mean_w = feats.get("drivable_width_mean", float("nan"))
        min_w  = feats.get("drivable_width_min",  float("nan"))
        if not math.isnan(mean_w):
            ax.axvline(mean_w, color="#F57F17", linestyle="--", linewidth=1.5, label=f"均值={mean_w:.3f}")
        if not math.isnan(min_w):
            ax.axvline(min_w,  color="#C62828", linestyle=":",  linewidth=1.5, label=f"最小={min_w:.3f}")
        ax.legend(loc="lower right", fontsize=8, labelcolor="#212121",
                  framealpha=0.9, facecolor="#FFFFFF", edgecolor="#BDBDBD")
    ax.set_xlabel("宽度（归一化）", color="#424242")
    ax.set_ylabel("图像行（0=顶 1=底）", color="#424242")
    ax.set_title("可驾驶宽度纵向分布", color="#212121", fontsize=11)
    ax.tick_params(colors="#424242")
    ax.invert_yaxis()
    for sp in ax.spines.values(): sp.set_edgecolor("#BDBDBD")

    # R1C2: 特征数值表
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor("#FFFFFF"); ax.axis("off")
    ordered_keys = [
        "drivable_coverage","drivable_width_mean","drivable_width_min",
        "road_curvature_mean","road_curvature_max","lane_count_visible",
        "lane_curvature_mean","lane_offset","lane_marking_visibility",
    ]
    table_data = [[k, f"{feats.get(k, float('nan')):.4f}"] for k in ordered_keys]
    tbl = ax.table(cellText=table_data, colLabels=["特征名", "值"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#E3F2FD" if r == 0 else "#FFFFFF")
        cell.set_text_props(color="#212121"); cell.set_edgecolor("#BDBDBD")
    ax.set_title("特征数值", color="#212121", fontsize=11)

    fig.suptitle(
        f"YOLOPv2 特征可视化  |  帧 {fidx}  t={ts:.2f}s  |  {Path(TEST_VIDEO_PATH).name}",
        color="#212121", fontsize=13, fontweight="bold",
    )
    _save_fig(fig, f"yolopv2_frame_{fidx:05d}.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# 通用时序图
# ══════════════════════════════════════════════════════════════════════════════

def _plot_timeseries(df: pd.DataFrame, extractor: str, title: str):
    """绘制所有数值特征随窗口时间变化的折线图。"""
    meta_cols  = {"window_idx", "window_start_sec", "window_end_sec"}
    feat_cols  = [c for c in df.columns
                  if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    # 过滤全 NaN 列
    feat_cols  = [c for c in feat_cols if df[c].notna().any()]
    if not feat_cols:
        return

    t_vals = df["window_start_sec"].values
    ncols  = 3
    nrows  = math.ceil(len(feat_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, max(4, 3.5 * nrows)))
    fig.patch.set_facecolor("#F5F5F5")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(feat_cols)))

    for i, col in enumerate(feat_cols):
        ax = axes_flat[i]
        ax.set_facecolor("#FFFFFF")
        vals = df[col].values
        ax.plot(t_vals, vals, color=colors[i], linewidth=1.8, marker="o",
                markersize=3, alpha=0.9)
        mean_v = np.nanmean(vals)
        ax.axhline(mean_v, color="#9E9E9E", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_title(col, color="#212121", fontsize=9)
        ax.set_xlabel("时间 (s)", color="#424242", fontsize=7)
        ax.tick_params(colors="#424242", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#BDBDBD")
        ax.text(0.98, 0.95, f"均值={mean_v:.3f}", transform=ax.transAxes,
                ha="right", va="top", color="#616161", fontsize=7, alpha=0.9)

    # 隐藏多余子图
    for j in range(len(feat_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{title}  |  {Path(TEST_VIDEO_PATH).name}\n"
        f"窗口大小={WINDOW_SIZE}s  步长={WINDOW_STEP}s  聚合={AGGREGATION}  共{len(df)}个窗口",
        color="#212121", fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, f"{extractor}_timeseries.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# 公共工具
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig: plt.Figure, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  可视化已保存: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def _print_test_startup(device: str):
    """打印详细的测试配置、硬件状态与视频信息。"""
    W = 68

    # ── GPU 信息 ───────────────────────────────────────────────────────────
    device_line = device.upper()
    if device == "cuda":
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            device_line = f"CUDA  ·  {gpu_name}  ·  显存 {gpu_mem:.1f} GB"
        except Exception:
            device_line = "CUDA  (GPU 信息读取失败)"
    if DEVICE == "auto":
        device_line += "  (auto 自动检测)"

    # ── 视频基本信息（快速读取，不加载帧）────────────────────────────────────
    vid_info = "读取失败"
    try:
        vpath = TEST_VIDEO_PATH
        if not os.path.isabs(vpath):
            vpath = os.path.normpath(os.path.join(_BASE_DIR, vpath))
        cap_tmp = cv2.VideoCapture(vpath)
        if cap_tmp.isOpened():
            _fps    = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
            _total  = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
            _dur    = _total / _fps
            _w      = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
            _h      = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_tmp.release()
            _test_dur  = min(TEST_DURATION_SEC, _dur) if TEST_DURATION_SEC > 0 else _dur
            _est_frames= int(_test_dur * _fps / FRAME_SAMPLE_INTERVAL)
            vid_info = (f"{_w}×{_h}  {_fps:.1f}fps  时长={_dur:.1f}s  "
                        f"→ 测试前 {_test_dur:.0f}s  预计处理 {_est_frames} 帧")
    except Exception:
        pass

    # ── 模型文件状态 ───────────────────────────────────────────────────────
    model_map = {
        "segformer": os.path.join(SEGFORMER_MODEL_DIR, "config.json"),
        "yolo":      YOLO_MODEL_PATH,
        "yolopv2":   YOLOPV2_WEIGHTS,
    }
    model_path = model_map.get(TEST_EXTRACTOR, "")
    model_ok   = os.path.exists(model_path)
    model_size = (os.path.getsize(model_path) / 1024 ** 2
                  if model_ok else 0)

    if ROI.get("enabled"):
        roi_str = (f"启用  ({ROI['x1']}, {ROI['y1']}) → ({ROI['x2']}, {ROI['y2']})"
                   f"  [{ROI['x2']-ROI['x1']}×{ROI['y2']-ROI['y1']} px]")
    else:
        roi_str = "关闭（全图推理）"

    print("\n" + "═" * W)
    print(f"{'特征提取器检查脚本  |  extractor_test.py':^{W}}")
    print("═" * W)
    print(f"  ┌─ 测试配置 {'─'*(W-13)}")
    print(f"  │  提取器      : {TEST_EXTRACTOR.upper()}")
    print(f"  │  推理设备    : {device_line}")
    print(f"  │  ROI         : {roi_str}")
    print(f"  │  窗口大小    : {WINDOW_SIZE}s  |  步长: {WINDOW_STEP}s"
          f"  |  聚合: {AGGREGATION}")
    print(f"  │  帧采样间隔  : 每 {FRAME_SAMPLE_INTERVAL} 帧取 1 帧（顺序）  可视化随机抽 {VIS_N_FRAMES} 帧")
    print(f"  │  可视化帧数  : {VIS_N_FRAMES} 帧")
    print(f"  ├─ 视频信息 {'─'*(W-13)}")
    print(f"  │  路径        : {TEST_VIDEO_PATH}")
    print(f"  │  属性        : {vid_info}")
    print(f"  ├─ 模型文件 {'─'*(W-13)}")
    if model_ok:
        print(f"  │  ✓  {TEST_EXTRACTOR:<10}: {Path(model_path).name}"
              f"  ({model_size:.1f} MB)")
    else:
        print(f"  │  ✗  {TEST_EXTRACTOR:<10}: 未找到 → {model_path}")
        print(f"  │      请先运行 download_models.py")
    print(f"  ├─ 输出目录 {'─'*(W-13)}")
    print(f"  │  {OUTPUT_DIR}")
    print(f"  └{'─'*(W-3)}")
    print("═" * W + "\n")


def main():
    device = resolve_device(DEVICE)
    _print_test_startup(device)

    try:
        if TEST_EXTRACTOR == "segformer":
            run_segformer_test(device)
        elif TEST_EXTRACTOR == "yolo":
            run_yolo_test(device)
        elif TEST_EXTRACTOR == "yolopv2":
            run_yolopv2_test(device)
        else:
            print(f"[ERROR] 未知提取器: {TEST_EXTRACTOR}")
            print("  可选值: segformer / yolo / yolopv2")
            return
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    except Exception:
        traceback.print_exc()
        return

    print("\n" + "=" * 65)
    print(f"  测试完成！所有结果已保存至:")
    print(f"  {OUTPUT_DIR}")
    files = sorted(os.listdir(OUTPUT_DIR))
    for f in files:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f}  ({size/1024:.0f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()

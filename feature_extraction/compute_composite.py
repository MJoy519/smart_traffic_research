"""
复合特征计算脚本 — compute_composite.py
========================================
读取 main.py 生成的基础特征 CSV，按匹配的 route_num / window_size / window_step
计算跨模型复合特征，结果保存至同一子目录下。

可计算的复合特征（共 7 项）：
  ┌──────────────────────────────────────────┬──────────────────────────────┐
  │ 特征名                                   │ 所需提取器                   │
  ├──────────────────────────────────────────┼──────────────────────────────┤
  │ drivable_occupancy_ratio                 │ YOLO + YOLOPv2               │
  │ vru_drivable_intrusion_rate              │ YOLO + YOLOPv2               │
  │ interaction_risk_integral_itcc           │ YOLO                         │
  │ enclosure_crowding_stress                │ SegFormer + YOLO + YOLOPv2   │
  │ green_buffer_under_congestion            │ SegFormer + YOLO + YOLOPv2   │
  │ exposed_vru_conflict_index               │ YOLO + YOLOPv2               │
  │ semantic_monotony_fatigue                │ SegFormer（时序）             │
  └──────────────────────────────────────────┴──────────────────────────────┘

运行方式：
    python compute_composite.py
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║                    全局可调参数                               ║
# ╚══════════════════════════════════════════════════════════════╝

# 要处理的路线编号（需与 main.py 运行时的 TARGET_ROUTES 一致）
TARGET_ROUTES = [2]

# 滑动窗口参数（需与 main.py 运行时使用的参数完全一致）
WINDOW_SIZE           = 3       # 窗口大小（秒）
WINDOW_STEP           = 1       # 步长（秒）
FRAME_SAMPLE_INTERVAL = 1       # 帧采样间隔（每 N 帧取 1 帧）

# 数值稳定性常数（防止除零）
EPS = 1e-6

# TTC 截断下限（秒）：避免 TTC 极小时风险积分爆炸
TTC_MIN = 0.01

# 语义单调指数所用的语义分量列（需 SegFormer 提供）
SEMANTIC_COLS = [
    "road_coverage",
    "building_coverage",
    "sky_visibility",
    "green_coverage",
    "sidewalk_coverage",
]

# ════════════════════════════════════════════════════════════════

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from config import RESULTS_DIR


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _run_subdir(window_size: float, window_step: float,
                frame_sample_interval: int) -> str:
    """与 main.py 保持一致的子目录命名规则。"""
    ws_str = str(window_size).replace(".", "_")
    wt_str = str(window_step).replace(".", "_")
    return os.path.join(
        RESULTS_DIR, f"ws{ws_str}_wt{wt_str}_si{frame_sample_interval}"
    )


def _csv_path(sub_dir: str, route_num: int,
              ext_name: str, window_size: float, window_step: float) -> str:
    ws_str = str(window_size).replace(".", "_")
    wt_str = str(window_step).replace(".", "_")
    return os.path.join(sub_dir, f"{route_num}_{ext_name}_{ws_str}_{wt_str}.csv")


def _load_csv(sub_dir: str, route_num: int,
              ext_name: str, window_size: float, window_step: float
              ) -> Optional[pd.DataFrame]:
    """尝试加载指定提取器的基础特征 CSV，不存在或读取失败则返回 None。"""
    fpath = _csv_path(sub_dir, route_num, ext_name, window_size, window_step)
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
        print(f"    ✓ {Path(fpath).name}  ({len(df)} 行)")
        return df
    except Exception as e:
        print(f"    ✗ 读取 {Path(fpath).name} 失败: {e}")
        return None


def _merge_base_dfs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    将多个提取器的 DataFrame 按 (video_name, window_idx) 合并为一张宽表。
    元信息列（route_num, window_start_sec, window_end_sec）取第一个可用值。
    各提取器特征列名在 BDD100K / SegFormer / YOLOPv2 间均不重复，无需加后缀。
    """
    META = ["route_num", "video_name", "window_idx",
            "window_start_sec", "window_end_sec"]

    merged: Optional[pd.DataFrame] = None

    for ext_name, df in dfs.items():
        # 取元信息 + 特征列（避免重复元信息覆盖）
        meta_present = [c for c in META if c in df.columns]
        feat_cols    = [c for c in df.columns if c not in META]

        if merged is None:
            # 第一个 DF 作为基准，保留完整元信息
            merged = df[meta_present + feat_cols].copy()
        else:
            # 后续 DF 只取 (video_name, window_idx) + 特征列
            join_cols = ["video_name", "window_idx"]
            add_cols  = [c for c in feat_cols if c not in merged.columns]
            if not add_cols:
                continue
            merged = merged.merge(
                df[join_cols + add_cols],
                on=join_cols,
                how="left",
            )

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 各复合特征计算函数
# ══════════════════════════════════════════════════════════════════════════════

def _calc_drivable_occupancy_ratio(df: pd.DataFrame) -> pd.Series:
    """
    可驾驶空间占用率
      = 动态目标面积占比 / 可驾驶区域占比
    近似解释：可驾驶区域中被动态交通参与者实际占用的比例。
    来源：YOLO(dynamic_object_area_ratio) + YOLOPv2(drivable_coverage)
    """
    veh = df["dynamic_object_area_ratio"].fillna(0.0)
    da  = df["drivable_coverage"].fillna(0.0)
    return veh / (da + EPS)


def _calc_vru_drivable_intrusion_rate(df: pd.DataFrame) -> pd.Series:
    """
    弱势参与者侵入率
      = (行人数 + 骑行/摩托数) × 可驾驶覆盖率
    当可驾驶区域大且弱势参与者多时，侵入概率更高。
    来源：YOLO(person_count, cyclist_motorcycle_count) + YOLOPv2(drivable_coverage)
    """
    vru = (df["person_count"].fillna(0.0)
           + df["cyclist_motorcycle_count"].fillna(0.0))
    da  = df["drivable_coverage"].fillna(0.0)
    return vru * da


def _calc_interaction_risk_integral_itcc(df: pd.DataFrame) -> pd.Series:
    """
    交互风险积分（逆 TTC 和）
      = risk_count / max(min_ttc, TTC_MIN)
    risk_count：时间窗内 TTC 低于阈值的高风险帧数。
    min_ttc：窗口内最小 TTC（越小风险越高）。
    来源：YOLO(risk_count, min_ttc)
    """
    rc  = df["risk_count"].fillna(0.0)
    ttc = df["min_ttc"].fillna(EPS).clip(lower=TTC_MIN)
    return rc / ttc


def _calc_enclosure_crowding_stress(df: pd.DataFrame,
                                    dor: pd.Series) -> pd.Series:
    """
    围合-拥挤压力指数
      = building_oppression × drivable_occupancy_ratio
    静态围合感 × 动态拥挤程度的乘积，反映"封闭街道中交通拥挤的叠加压力"。
    来源：SegFormer(building_oppression) + drivable_occupancy_ratio
    """
    bop = df["building_oppression"].fillna(0.0)
    return bop * dor


def _calc_green_buffer_under_congestion(df: pd.DataFrame,
                                        dor: pd.Series) -> pd.Series:
    """
    绿化缓冲拥堵指数
      = drivable_occupancy_ratio × (1 - green_coverage)
    值越高：拥堵越严重且绿化缓冲越少；值越低：绿化良好或拥堵轻。
    来源：SegFormer(green_coverage) + drivable_occupancy_ratio
    """
    gc = df["green_coverage"].fillna(0.0)
    return dor * (1.0 - gc)


def _calc_exposed_vru_conflict_index(df: pd.DataFrame,
                                     vru_intr: pd.Series) -> pd.Series:
    """
    暴露式弱势参与者冲突指数
      = vru_drivable_intrusion_rate / max(min_ttc, TTC_MIN)
    在弱势参与者侵入的基础上，叠加 TTC 风险严重程度。
    来源：YOLO(min_ttc) + vru_drivable_intrusion_rate
    """
    ttc = df["min_ttc"].fillna(EPS).clip(lower=TTC_MIN)
    return vru_intr / ttc


def _calc_semantic_monotony_fatigue(df: pd.DataFrame) -> pd.Series:
    """
    语义单调疲劳指数（需跨窗口时序差分，逐视频计算）
      帧间语义变化量 = Σ |vec_t - vec_{t-1}|（L1 范数）
      semantic_monotony_fatigue = 1 / (change + EPS)
    值越高表示场景越单调、潜在疲劳风险越高。
    首个窗口用第二窗口的变化量填充（无前序参考）。
    来源：SegFormer(road_coverage, building_coverage, sky_visibility,
                    green_coverage, sidewalk_coverage)
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)

    cols = [c for c in SEMANTIC_COLS if c in df.columns]
    if not cols:
        return result

    for video, grp in df.groupby("video_name"):
        grp_s = grp.sort_values("window_idx")
        vecs  = grp_s[cols].fillna(0.0).values   # (N, K)
        n     = len(vecs)

        if n == 1:
            changes = np.array([0.0])
        else:
            diffs   = np.sum(np.abs(np.diff(vecs, axis=0)), axis=1)  # (N-1,)
            # 首窗口：用第一个差值填充（无前序参考时与第二窗口等价）
            changes = np.concatenate([[diffs[0]], diffs])

        smf = 1.0 / (changes + EPS)
        result.loc[grp_s.index] = smf

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 单路线复合特征计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_composite_for_route(
    route_num: int,
    window_size: float,
    window_step: float,
    frame_sample_interval: int,
) -> Optional[pd.DataFrame]:
    """
    加载路线 route_num 的所有可用基础特征，计算复合特征。

    Returns:
        包含元信息列 + 所有可计算复合特征的 DataFrame；无可用数据时返回 None。
    """
    sub_dir = _run_subdir(window_size, window_step, frame_sample_interval)
    print(f"\n  路线 {route_num}  │  子目录: {sub_dir}")

    # ── 1. 加载各提取器 CSV ────────────────────────────────────────────────
    dfs: Dict[str, pd.DataFrame] = {}
    for ext in ["segformer", "yolo", "yolopv2"]:
        df = _load_csv(sub_dir, route_num, ext, window_size, window_step)
        if df is not None:
            dfs[ext] = df

    if not dfs:
        print(f"  [跳过] 路线 {route_num} 在当前参数下没有任何基础特征文件")
        return None

    avail = set(dfs.keys())
    print(f"  可用提取器: {sorted(avail)}")

    # ── 2. 合并为宽表 ─────────────────────────────────────────────────────
    merged = _merge_base_dfs(dfs)

    # ── 3. 计算复合特征 ───────────────────────────────────────────────────
    computed: List[str] = []
    skipped:  List[str] = []

    # 中间量（供多个复合特征复用）
    _dor:      Optional[pd.Series] = None
    _vru_intr: Optional[pd.Series] = None

    has_yolo_pv2 = {"yolo", "yolopv2"} <= avail
    has_all_three = {"segformer", "yolo", "yolopv2"} <= avail

    # ── drivable_occupancy_ratio ─────────────────────────────────────────
    if has_yolo_pv2:
        _dor = _calc_drivable_occupancy_ratio(merged)
        merged["drivable_occupancy_ratio"] = _dor
        computed.append("drivable_occupancy_ratio")
    else:
        skipped.append("drivable_occupancy_ratio (需 YOLO+YOLOPv2)")

    # ── vru_drivable_intrusion_rate ───────────────────────────────────────
    if has_yolo_pv2:
        _vru_intr = _calc_vru_drivable_intrusion_rate(merged)
        merged["vru_drivable_intrusion_rate"] = _vru_intr
        computed.append("vru_drivable_intrusion_rate")
    else:
        skipped.append("vru_drivable_intrusion_rate (需 YOLO+YOLOPv2)")

    # ── interaction_risk_integral_itcc ────────────────────────────────────
    if "yolo" in avail:
        merged["interaction_risk_integral_itcc"] = (
            _calc_interaction_risk_integral_itcc(merged)
        )
        computed.append("interaction_risk_integral_itcc")
    else:
        skipped.append("interaction_risk_integral_itcc (需 YOLO)")

    # ── enclosure_crowding_stress ─────────────────────────────────────────
    if has_all_three and _dor is not None:
        merged["enclosure_crowding_stress"] = (
            _calc_enclosure_crowding_stress(merged, _dor)
        )
        computed.append("enclosure_crowding_stress")
    else:
        skipped.append("enclosure_crowding_stress (需 SegFormer+YOLO+YOLOPv2)")

    # ── green_buffer_under_congestion ─────────────────────────────────────
    if has_all_three and _dor is not None:
        merged["green_buffer_under_congestion"] = (
            _calc_green_buffer_under_congestion(merged, _dor)
        )
        computed.append("green_buffer_under_congestion")
    else:
        skipped.append("green_buffer_under_congestion (需 SegFormer+YOLO+YOLOPv2)")

    # ── exposed_vru_conflict_index ────────────────────────────────────────
    if has_yolo_pv2 and _vru_intr is not None:
        merged["exposed_vru_conflict_index"] = (
            _calc_exposed_vru_conflict_index(merged, _vru_intr)
        )
        computed.append("exposed_vru_conflict_index")
    else:
        skipped.append("exposed_vru_conflict_index (需 YOLO+YOLOPv2)")

    # ── semantic_monotony_fatigue ─────────────────────────────────────────
    if "segformer" in avail:
        merged["semantic_monotony_fatigue"] = (
            _calc_semantic_monotony_fatigue(merged)
        )
        computed.append("semantic_monotony_fatigue")
    else:
        skipped.append("semantic_monotony_fatigue (需 SegFormer)")

    # ── 4. 汇报 ──────────────────────────────────────────────────────────
    print(f"  已计算 ({len(computed)}): {computed}")
    if skipped:
        print(f"  已跳过 ({len(skipped)}): {skipped}")

    if not computed:
        print(f"  [跳过] 路线 {route_num} 没有可计算的复合特征")
        return None

    # ── 5. 整理输出列：元信息 + 复合特征 ──────────────────────────────────
    META_COLS = ["route_num", "video_name", "window_idx",
                 "window_start_sec", "window_end_sec"]
    out_cols = [c for c in META_COLS if c in merged.columns] + computed
    return merged[out_cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def _print_startup():
    sub_dir = _run_subdir(WINDOW_SIZE, WINDOW_STEP, FRAME_SAMPLE_INTERVAL)
    W = 65
    print("\n" + "═" * W)
    print(f"{'复合特征计算  |  compute_composite.py':^{W}}")
    print("═" * W)
    print(f"  ┌─ 运行配置 {'─'*(W-13)}")
    print(f"  │  窗口大小        : {WINDOW_SIZE}s  步长: {WINDOW_STEP}s")
    print(f"  │  帧采样间隔      : 每 {FRAME_SAMPLE_INTERVAL} 帧取 1 帧")
    print(f"  │  目标路线        : {TARGET_ROUTES}")
    print(f"  │  数值稳定常数    : EPS={EPS}  TTC_MIN={TTC_MIN}s")
    print(f"  ├─ 路径 {'─'*(W-9)}")
    print(f"  │  基础特征目录    : {sub_dir}")
    print(f"  │  复合特征文件    : {{route}}_com_*.csv（同目录）")
    print(f"  └{'─'*(W-3)}")
    print("═" * W)


def main():
    _print_startup()

    sub_dir = _run_subdir(WINDOW_SIZE, WINDOW_STEP, FRAME_SAMPLE_INTERVAL)

    if not os.path.isdir(sub_dir):
        print(f"\n[ERROR] 基础特征目录不存在: {sub_dir}")
        print("  请先运行 main.py 生成各提取器的基础特征。")
        return

    ws_str = str(WINDOW_SIZE).replace(".", "_")
    wt_str = str(WINDOW_STEP).replace(".", "_")

    saved_files = []
    for route_num in TARGET_ROUTES:
        df_out = compute_composite_for_route(
            route_num=route_num,
            window_size=WINDOW_SIZE,
            window_step=WINDOW_STEP,
            frame_sample_interval=FRAME_SAMPLE_INTERVAL,
        )

        if df_out is None or df_out.empty:
            continue

        fname = f"{route_num}_com_{ws_str}_{wt_str}.csv"
        fpath = os.path.join(sub_dir, fname)
        df_out.to_csv(fpath, index=False)
        saved_files.append((fname, len(df_out), len(df_out.columns)))
        print(f"  ✓ 保存: {fname}  ({len(df_out)} 行 × {len(df_out.columns)} 列)")

    print("\n" + "=" * 65)
    if saved_files:
        print("  复合特征计算完成！生成文件：")
        for fname, rows, cols in saved_files:
            print(f"    {fname}  →  {rows} 行 × {cols} 列")
    else:
        print("  未生成任何复合特征文件，请检查基础特征是否存在。")
    print(f"  输出目录: {sub_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()

"""
YOLOPv2 特征提取器
使用 CAIC-AD/YOLOPv2 提取可驾驶区域和车道线特征。

可提取特征（共 9 项）：
  drivable_coverage         - 可驾驶区域占比
  drivable_width_mean       - 平均可驾驶宽度（归一化）
  drivable_width_min        - 最小可驾驶宽度（归一化）
  road_curvature_mean       - 平均道路中心线曲率
  road_curvature_max        - 最大道路曲率
  lane_count_visible        - 可见车道线数量
  lane_curvature_mean       - 平均车道线曲率
  lane_offset               - 车道中心偏移（归一化，负=偏左，正=偏右）
  lane_marking_visibility   - 车道线可见性（车道像素密度）
"""

import os
import sys
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

# YOLOPv2 标准输入尺寸
_YOLOPV2_INPUT_H = 384
_YOLOPV2_INPUT_W = 640


class YOLOPv2Extractor:
    """
    YOLOPv2 特征提取器。

    Args:
        repo_dir    (str): YOLOPv2 仓库根目录（含 models/ utils/ 等）
        weights_path(str): yolopv2.pt 模型权重路径
        device      (str): 推理设备，"cuda" 或 "cpu"
    """

    FEATURE_NAMES = [
        "drivable_coverage",
        "drivable_width_mean",
        "drivable_width_min",
        "road_curvature_mean",
        "road_curvature_max",
        "lane_count_visible",
        "lane_curvature_mean",
        "lane_offset",
        "lane_marking_visibility",
    ]

    def __init__(self, repo_dir: str, weights_path: str, device: str = "cuda"):
        import torch
        self.device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch   = torch

        print(f"[YOLOPv2] 加载模型: {weights_path}  (device={self.device})")

        load_errors = []

        # ── 方式1：TorchScript 格式（官方 Release 发布的 yolopv2.pt 均为此格式）──
        try:
            self._model = torch.jit.load(weights_path, map_location=self.device)
            self._model.eval()
            self._load_mode = "torchscript"
            print("[YOLOPv2] torch.jit.load 加载成功（TorchScript）")
        except Exception as e1:
            load_errors.append(f"torch.jit.load: {e1}")
            print(f"[YOLOPv2] TorchScript 加载失败: {e1}")

            # ── 方式2：通过仓库源码 attempt_load（需要完整克隆仓库）──────────────
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            try:
                from models.experimental import attempt_load
                self._model = attempt_load(weights_path, map_location=self.device)
                self._model.eval()
                self._load_mode = "repo"
                print("[YOLOPv2] 通过仓库代码加载模型成功")
            except Exception as e2:
                load_errors.append(f"attempt_load: {e2}")
                print(f"[YOLOPv2] 仓库加载失败: {e2}")

                # ── 方式3：torch.load 兜底（weights_only=False 兼容 PyTorch≥2.6）──
                try:
                    self._model = torch.load(
                        weights_path, map_location=self.device, weights_only=False
                    )
                    if hasattr(self._model, "model"):
                        self._model = self._model.model
                    self._model.eval()
                    self._load_mode = "direct"
                    print("[YOLOPv2] torch.load 加载成功")
                except Exception as e3:
                    load_errors.append(f"torch.load: {e3}")
                    raise RuntimeError(
                        f"YOLOPv2 模型加载失败（已尝试全部方式）。\n"
                        f"仓库路径: {repo_dir}\n"
                        f"权重路径: {weights_path}\n"
                        + "\n".join(f"  [{i+1}] {err}" for i, err in enumerate(load_errors))
                        + "\n请先运行 download_models.py 下载 YOLOPv2。"
                    )

        print("[YOLOPv2] 模型加载完成")

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────────────

    def extract_frame(self, frame: np.ndarray) -> dict:
        """
        对单帧进行 YOLOPv2 推理并提取可驾驶区域 + 车道线特征。

        Args:
            frame: BGR numpy array，shape (H, W, 3)

        Returns:
            dict: 包含 FEATURE_NAMES 中所有特征的字典
        """
        h_orig, w_orig = frame.shape[:2]

        # ── 预处理 ───────────────────────────────────────────────────────────
        img_input = self._preprocess(frame)

        # ── 模型推理 ─────────────────────────────────────────────────────────
        da_mask, ll_mask = self._infer(img_input, h_orig, w_orig)
        if da_mask is None:
            return self._nan_features()

        # ── 特征计算 ─────────────────────────────────────────────────────────
        total_pix = h_orig * w_orig

        # 可驾驶区域特征
        da_pix    = int(np.sum(da_mask))
        da_cov    = da_pix / total_pix

        row_widths = self._row_widths(da_mask)
        da_w_mean  = float(np.mean(row_widths)) / w_orig  if row_widths else 0.0
        da_w_min   = float(np.min(row_widths))  / w_orig  if row_widths else 0.0

        # 道路中心线曲率（基于可驾驶区域 mask）
        road_curve_mean, road_curve_max = self._drivable_curvature(da_mask)

        # 车道线特征
        lane_cnt, lane_curve_mean = self._lane_features(ll_mask)
        lane_offset               = self._lane_offset(ll_mask, w_orig)
        lane_vis                  = float(np.sum(ll_mask)) / total_pix

        return {
            "drivable_coverage":       float(da_cov),
            "drivable_width_mean":     float(da_w_mean),
            "drivable_width_min":      float(da_w_min),
            "road_curvature_mean":     float(road_curve_mean),
            "road_curvature_max":      float(road_curve_max),
            "lane_count_visible":      int(lane_cnt),
            "lane_curvature_mean":     float(lane_curve_mean),
            "lane_offset":             float(lane_offset),
            "lane_marking_visibility": float(lane_vis),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 推理相关
    # ──────────────────────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> "torch.Tensor":
        """BGR → 归一化 float32 Tensor，shape (1, 3, H_in, W_in)。"""
        resized = cv2.resize(frame, (_YOLOPV2_INPUT_W, _YOLOPV2_INPUT_H))
        rgb     = resized[:, :, ::-1].copy()  # BGR → RGB
        tensor  = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)  # (1,3,H,W)

    def _infer(
        self,
        img_tensor: "torch.Tensor",
        h_orig: int,
        w_orig: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        运行 YOLOPv2 推理，返回 (da_mask, ll_mask)，shape=(h_orig, w_orig)。
        da_mask: 可驾驶区域二值 mask
        ll_mask: 车道线二值 mask
        """
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            try:
                output = self._model(img_tensor)
            except Exception as e:
                print(f"[YOLOPv2] 推理错误: {e}")
                return None, None

        # YOLOPv2 输出格式：(det_out, da_seg_out, ll_seg_out)
        # 兼容不同版本的输出解析
        da_out, ll_out = self._parse_outputs(output)
        if da_out is None:
            return None, None

        # argmax 得到二值 mask
        _, da_idx = torch.max(da_out, dim=1)
        _, ll_idx = torch.max(ll_out, dim=1)

        # resize 回原始分辨率
        da_idx_r = F.interpolate(
            da_idx.unsqueeze(1).float(), size=(h_orig, w_orig), mode="nearest"
        ).squeeze().cpu().numpy().astype(np.uint8)

        ll_idx_r = F.interpolate(
            ll_idx.unsqueeze(1).float(), size=(h_orig, w_orig), mode="nearest"
        ).squeeze().cpu().numpy().astype(np.uint8)

        return da_idx_r, ll_idx_r

    @staticmethod
    def _parse_outputs(output) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """
        解析 YOLOPv2 模型输出，兼容多种格式。
        返回 (da_seg_logits, ll_seg_logits)。
        """
        if isinstance(output, (list, tuple)):
            # 典型格式：(det, da_seg, ll_seg) 或 ((det, anchor), da_seg, ll_seg)
            if len(output) == 3:
                _, da, ll = output
                return da, ll
            elif len(output) == 2:
                da, ll = output
                return da, ll
        # 无法解析
        print(f"[YOLOPv2] 无法解析输出格式，类型={type(output)}，长度={len(output) if hasattr(output,'__len__') else 'N/A'}")
        return None, None

    # ──────────────────────────────────────────────────────────────────────────
    # 特征计算方法
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_widths(da_mask: np.ndarray) -> List[float]:
        """统计可驾驶区域 mask 每行中可驾驶像素的水平跨度（像素数）。"""
        widths = []
        for row in da_mask:
            nonzero = np.where(row > 0)[0]
            if len(nonzero) >= 2:
                widths.append(float(nonzero[-1] - nonzero[0] + 1))
        return widths

    @staticmethod
    def _drivable_curvature(
        da_mask: np.ndarray,
        fit_degree: int = 2,
        sample_rows: int = 30,
    ) -> Tuple[float, float]:
        """
        基于可驾驶区域 mask 计算道路中心线曲率。
        方法：取每行可驾驶像素的横向中心 → 拟合多项式 x = f(y) → 计算曲率。
        返回：(mean_curvature, max_curvature)
        """
        h, w = da_mask.shape
        cx_list, y_list = [], []

        step = max(1, h // sample_rows)
        for r in range(0, h, step):
            nz = np.where(da_mask[r] > 0)[0]
            if len(nz) >= 2:
                cx_list.append(float((nz[0] + nz[-1]) / 2.0))
                y_list.append(float(r))

        if len(y_list) < fit_degree + 2:
            return 0.0, 0.0

        y_arr = np.array(y_list)
        x_arr = np.array(cx_list)
        try:
            coeffs = np.polyfit(y_arr, x_arr, fit_degree)  # x = a*y² + b*y + c
        except np.linalg.LinAlgError:
            return 0.0, 0.0

        # 曲率公式 κ = |x''| / (1 + x'²)^(3/2)，在各采样点计算
        poly_d1 = np.polyder(np.poly1d(coeffs), 1)
        poly_d2 = np.polyder(np.poly1d(coeffs), 2)

        curvatures = []
        for y in y_arr:
            d1 = poly_d1(y)
            d2 = poly_d2(y)
            kappa = abs(d2) / (1 + d1 ** 2) ** 1.5
            # 归一化：除以图像宽度使曲率与分辨率无关
            curvatures.append(kappa / w)

        if not curvatures:
            return 0.0, 0.0
        return float(np.mean(curvatures)), float(np.max(curvatures))

    def _lane_features(
        self,
        ll_mask: np.ndarray,
        min_pix: int = 50,
    ) -> Tuple[int, float]:
        """
        分析车道线 mask，计算可见车道线数量和平均曲率。
        使用连通域分析区分不同车道线。
        返回：(lane_count, mean_curvature)
        """
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            ll_mask, connectivity=8
        )
        h, w = ll_mask.shape
        valid_curves = []

        for label_id in range(1, labels.max() + 1 if labels.max() > 0 else 1):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_pix:
                continue

            comp_mask = (labels == label_id).astype(np.uint8)
            curve = self._fit_lane_curve(comp_mask, w)
            valid_curves.append(curve)

        lane_count = len(valid_curves)
        mean_curve = float(np.mean(valid_curves)) if valid_curves else 0.0
        return lane_count, mean_curve

    @staticmethod
    def _fit_lane_curve(lane_comp: np.ndarray, img_w: int, fit_degree: int = 2) -> float:
        """对单条车道线连通域拟合多项式并返回归一化平均曲率。"""
        ys, xs = np.where(lane_comp > 0)
        if len(ys) < fit_degree + 2:
            return 0.0
        try:
            coeffs = np.polyfit(ys.astype(float), xs.astype(float), fit_degree)
        except np.linalg.LinAlgError:
            return 0.0

        poly_d1 = np.polyder(np.poly1d(coeffs), 1)
        poly_d2 = np.polyder(np.poly1d(coeffs), 2)
        curvatures = []
        for y in ys[::max(1, len(ys) // 20)]:
            d1  = poly_d1(float(y))
            d2  = poly_d2(float(y))
            kappa = abs(d2) / (1 + d1 ** 2) ** 1.5
            curvatures.append(kappa / img_w)
        return float(np.mean(curvatures)) if curvatures else 0.0

    @staticmethod
    def _lane_offset(ll_mask: np.ndarray, img_w: int) -> float:
        """
        计算车道中心偏移：
          1. 在图像下半部分找车道线像素
          2. 将像素分为左右两组（以图像中心为界）
          3. 取左组最右端和右组最左端的平均 → 估算车道中心
          4. 偏移 = (车道中心 - 图像中心) / 图像宽度
        返回值范围约 [-0.5, 0.5]，负=偏左，正=偏右。
        """
        h = ll_mask.shape[0]
        roi = ll_mask[h // 2:]  # 下半部分
        ys, xs = np.where(roi > 0)
        if len(xs) < 4:
            return 0.0

        center = img_w / 2.0
        left_xs  = xs[xs <  center]
        right_xs = xs[xs >= center]

        left_edge  = float(np.max(left_xs))  if len(left_xs)  > 0 else 0.0
        right_edge = float(np.min(right_xs)) if len(right_xs) > 0 else float(img_w)

        lane_center = (left_edge + right_edge) / 2.0
        offset = (lane_center - center) / img_w
        return float(np.clip(offset, -0.5, 0.5))

    @staticmethod
    def _nan_features() -> dict:
        """推理完全失败时的兜底返回，所有数值特征填 0 而非 NaN。"""
        return {
            "drivable_coverage":       0.0,
            "drivable_width_mean":     0.0,
            "drivable_width_min":      0.0,
            "road_curvature_mean":     0.0,
            "road_curvature_max":      0.0,
            "lane_count_visible":      0,
            "lane_curvature_mean":     0.0,
            "lane_offset":             0.0,
            "lane_marking_visibility": 0.0,
        }

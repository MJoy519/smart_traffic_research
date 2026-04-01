"""
SegFormer 特征提取器
使用 nvidia/segformer-b1-finetuned-cityscapes-1024-1024 提取场景语义分割特征。

可提取特征（共 8 项）：
  road_coverage          - 道路覆盖率
  sidewalk_coverage      - 人行道覆盖率
  building_coverage      - 建筑覆盖率
  sky_visibility         - 天空可见度
  green_coverage         - 绿化覆盖率
  wall_fence_coverage    - 墙体/围栏覆盖率
  building_oppression    - 建筑压迫感
  openness_index         - 开敞度指数
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Cityscapes 标签索引常量
_ROAD        = 0
_SIDEWALK    = 1
_BUILDING    = 2
_WALL        = 3
_FENCE       = 4
_VEGETATION  = 8
_SKY         = 10


class SegFormerExtractor:
    """
    基于 SegFormer（Cityscapes 预训练）的场景语义特征提取器。

    Args:
        model_dir (str): SegFormer 模型本地目录路径（含 config.json + pytorch_model.bin）
        device (str):    推理设备，"cuda" 或 "cpu"
    """

    FEATURE_NAMES = [
        "road_coverage",
        "sidewalk_coverage",
        "building_coverage",
        "sky_visibility",
        "green_coverage",
        "wall_fence_coverage",
        "building_oppression",
        "openness_index",
    ]

    def __init__(self, model_dir: str, device: str = "cuda"):
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        except ImportError:
            raise ImportError(
                "缺少 transformers 库，请执行: pip install transformers"
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[SegFormer] 加载模型: {model_dir}  (device={self.device})")

        self.processor = SegformerImageProcessor.from_pretrained(model_dir)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[SegFormer] 模型加载完成")

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────────────

    def extract_frame(self, frame: np.ndarray) -> dict:
        """
        对单帧图像进行语义分割并提取场景特征。

        Args:
            frame: BGR numpy array，shape (H, W, 3)，来自 OpenCV

        Returns:
            dict: 包含 FEATURE_NAMES 中所有特征名到 float 值的映射
        """
        seg_map = self._infer_segmap(frame)
        total   = seg_map.size

        road_pix   = np.sum(seg_map == _ROAD)
        side_pix   = np.sum(seg_map == _SIDEWALK)
        bldg_pix   = np.sum(seg_map == _BUILDING)
        sky_pix    = np.sum(seg_map == _SKY)
        veg_pix    = np.sum(seg_map == _VEGETATION)
        wall_pix   = np.sum(seg_map == _WALL) + np.sum(seg_map == _FENCE)

        road_cov  = road_pix  / total
        side_cov  = side_pix  / total
        bldg_cov  = bldg_pix  / total
        sky_vis   = sky_pix   / total
        green_cov = veg_pix   / total
        wf_cov    = wall_pix  / total

        bldg_opp  = self._building_oppression(seg_map, bldg_cov)
        openness  = self._openness_index(sky_vis, road_cov, bldg_cov)

        return {
            "road_coverage":       float(road_cov),
            "sidewalk_coverage":   float(side_cov),
            "building_coverage":   float(bldg_cov),
            "sky_visibility":      float(sky_vis),
            "green_coverage":      float(green_cov),
            "wall_fence_coverage": float(wf_cov),
            "building_oppression": float(bldg_opp),
            "openness_index":      float(openness),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────────

    def _infer_segmap(self, frame: np.ndarray) -> np.ndarray:
        """BGR 帧 → Cityscapes 语义分割图（H×W int 数组）。"""
        h, w = frame.shape[:2]
        img_rgb = frame[:, :, ::-1].astype(np.uint8)
        pil_img = Image.fromarray(img_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, C, H/4, W/4)

        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        seg_map   = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        return seg_map

    @staticmethod
    def _building_oppression(seg_map: np.ndarray, bldg_cov: float) -> float:
        """
        建筑压迫感指标：
          - 建筑越多（面积占比高）、越靠近图像上方（视觉顶部）→ 压迫感越强
          - 建筑横向宽度越宽 → 压迫感越强
        取值范围约 [0, 1]。
        """
        if bldg_cov < 1e-4:
            return 0.0

        h, w = seg_map.shape
        bldg_mask = (seg_map == _BUILDING).astype(float)

        # 建筑重心纵坐标（归一化至 [0,1]，0=顶部，1=底部）
        rows, _ = np.where(bldg_mask > 0)
        vert_center = float(np.mean(rows)) / h  # 越小 → 建筑越靠上 → 压迫感越强

        # 平均横向跨度（建筑横向"堵墙感"）
        row_spans = []
        for row_pixels in bldg_mask:
            if row_pixels.any():
                nz = np.where(row_pixels > 0)[0]
                row_spans.append((nz[-1] - nz[0] + 1) / w)
        width_ratio = float(np.mean(row_spans)) if row_spans else 0.0

        oppression = (1.0 - vert_center) * bldg_cov * (0.5 + 0.5 * width_ratio)
        return float(np.clip(oppression, 0.0, 1.0))

    @staticmethod
    def _openness_index(sky_vis: float, road_cov: float, bldg_cov: float) -> float:
        """
        开敞度指数：天空 + 道路 - 建筑，线性映射至 [0, 1]。
        原始范围约 [-1, 2]，通过 (raw + 1) / 3 归一化。
        """
        raw = sky_vis + road_cov - bldg_cov
        return float(np.clip((raw + 1.0) / 3.0, 0.0, 1.0))

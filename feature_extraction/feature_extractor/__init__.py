"""
feature_extractor 包
包含三个预训练模型的特征提取器模块：
  - SegFormerExtractor  : 场景语义分割特征（SegFormer + Cityscapes）
  - YOLOExtractor       : 目标检测 + ByteTrack 轨迹特征（YOLO + BDD100K）
  - YOLOPv2Extractor    : 可驾驶区域 + 车道线特征（YOLOPv2）
"""

from .segformer_extractor import SegFormerExtractor
from .yolo_extractor import YOLOExtractor
from .yolopv2_extractor import YOLOPv2Extractor

__all__ = ["SegFormerExtractor", "YOLOExtractor", "YOLOPv2Extractor"]

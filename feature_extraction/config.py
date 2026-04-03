"""
特征提取工程全局配置文件
包含模型路径、视频路径、保存路径、ROI设置及处理参数
"""
import os

# ==================== 基础路径 ====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)          # Code_0321/

# -------------------- 模型路径 --------------------
PRETRAIN_MODEL_DIR  = os.path.join(BASE_DIR, "pretrain_model")
SEGFORMER_MODEL_DIR = os.path.join(PRETRAIN_MODEL_DIR, "segformer")
YOLOPV2_REPO_DIR    = os.path.join(PRETRAIN_MODEL_DIR, "yolopv2")
YOLOPV2_WEIGHTS     = os.path.join(YOLOPV2_REPO_DIR, "data", "weights", "yolopv2.pt")
YOLO_MODEL_PATH     = os.path.join(PRETRAIN_MODEL_DIR, "yolo26-bdd100k.pt")

# Hugging Face 仓库 ID（离线时从 SEGFORMER_MODEL_DIR 加载）
SEGFORMER_REPO_ID = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"

# -------------------- 视频路径 --------------------
VIDEO_BASE_DIR = os.path.join(ROOT_DIR, "videos")
ROUTE_VIDEO_DIRS = {
    1: os.path.join(VIDEO_BASE_DIR, "1"),
    2: os.path.join(VIDEO_BASE_DIR, "2"),
    3: os.path.join(VIDEO_BASE_DIR, "3"),
}
# 支持的视频扩展名
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# -------------------- 结果保存路径 --------------------
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ==================== Cityscapes 语义分割标签 ====================
# SegFormer + Cityscapes 标签索引（共 19 类）
CITYSCAPES_LABELS = {
    0: "road",          1: "sidewalk",      2: "building",
    3: "wall",          4: "fence",         5: "pole",
    6: "traffic light", 7: "traffic sign",  8: "vegetation",
    9: "terrain",       10: "sky",          11: "person",
    12: "rider",        13: "car",          14: "truck",
    15: "bus",          16: "train",        17: "motorcycle",
    18: "bicycle",
}

# ==================== BDD100K YOLO 类别 ====================
# yolo26-bdd100k 模型类别索引
BDD100K_CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
}

# 各类别在特征中的分组映射
BDD100K_GROUPS = {
    "car":       [2],           # car
    "truck_bus": [3, 4],        # truck, bus
    "person":    [0],           # pedestrian
    "cyclist":   [1, 6, 7],     # rider, motorcycle, bicycle
    "vehicle":   [2, 3, 4, 5],  # 全部机动车（含 train）
    "traffic_sign_light": [8, 9],  # traffic light + traffic sign
}

# ==================== ROI 设置 ====================
# 是否启用 ROI 裁剪（仅在 ROI 区域内提取特征）
# 坐标单位：像素（px），使用 roi_selector.py 工具交互式获取
ROI = {
    "enabled": True,   # True = 仅处理 ROI 区域；False = 处理整张图像
    "x1": 4,            # ROI 左边界（px）
    "y1": 372,          # ROI 上边界（px）
    "x2": 1914,         # ROI 右边界（px）
    "y2": 729,          # ROI 下边界（px）
}

# ==================== 处理参数 ====================
# 帧采样间隔：每 N 帧取 1 帧（1 = 逐帧处理，5 = 每5帧取1帧）
FRAME_SAMPLE_INTERVAL = 1

# 推理设备："cuda" 自动回退到 "cpu"
DEVICE = "cuda"

# YOLO 推理参数
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES  = 0.45

# ==================== 风险检测参数 ====================
TTC_THRESHOLD        = 3.0   # 秒：TTC 低于此值视为风险事件
MIN_BBOX_AREA_CHANGE = 5     # 像素²/帧：最小面积变化量（用于判断接近方向）

# ==================== 车道曲率参数 ====================
LANE_FIT_DEGREE    = 2   # 多项式拟合阶数（2 = 二次曲线）
LANE_MIN_PIX       = 50  # 车道线最少像素点（低于此值视为噪声）
CURVATURE_EVAL_Y   = 0.9 # 在图像 y=0.9*H 位置估计曲率（靠近车辆位置）

"""
YOLO BDD100K 特征提取器
使用 yolo26-bdd100k 模型 + ByteTrack 多目标追踪，提取交通目标检测与运动轨迹特征。

可提取特征（共 16 项）：
  每帧特征（聚合时取中位数/均值）：
    car_count                 - 小汽车数量
    truck_bus_count           - 大型车辆数量（卡车+公交）
    person_count              - 行人数量
    cyclist_motorcycle_count  - 骑行/摩托数量
    total_object_count        - 目标总数
    dynamic_object_area_ratio - 动态目标面积占比
    large_vehicle_ratio       - 大型车辆占比
    traffic_sign_count        - 交通标志+信号灯数量

  窗口级轨迹特征（在 compute_window_traj_features() 中计算）：
    car_speed_mean            - 车辆平均速度（像素/秒）
    car_accel_mean            - 车辆平均加速度
    car_jerk_mean             - 车辆平均加加速度
    person_speed_mean         - 行人平均速度
    cyclist_speed_mean        - 骑行/摩托平均速度
    pedestrian_crossing_count - 行人横穿次数
    min_ttc                   - 最小碰撞时间（秒）
    risk_count                - 风险事件次数（TTC < 阈值）
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# BDD100K 类别索引 ─────────────────────────────────────────────────────────────
_PEDESTRIAN  = 0
_RIDER       = 1
_CAR         = 2
_TRUCK       = 3
_BUS         = 4
_TRAIN       = 5
_MOTORCYCLE  = 6
_BICYCLE     = 7
_TLIGHT      = 8
_TSIGN       = 9

_CAR_IDS      = {_CAR}
_TRUCK_BUS    = {_TRUCK, _BUS}
_PERSON_IDS   = {_PEDESTRIAN}
_CYCLIST_IDS  = {_RIDER, _MOTORCYCLE, _BICYCLE}
_VEHICLE_IDS  = {_CAR, _TRUCK, _BUS, _TRAIN}
_SIGN_IDS     = {_TLIGHT, _TSIGN}
_DYNAMIC_IDS  = {_PEDESTRIAN, _RIDER, _CAR, _TRUCK, _BUS, _TRAIN, _MOTORCYCLE, _BICYCLE}


class YOLOExtractor:
    """
    YOLO BDD100K 特征提取器（含 ByteTrack 多目标追踪）。

    Args:
        model_path  (str):   yolo26-bdd100k.pt 路径
        device      (str):   推理设备，"cuda" 或 "cpu"
        conf_thres  (float): 检测置信度阈值
        iou_thres   (float): NMS IoU 阈值
        ttc_threshold (float): TTC 低于此值（秒）视为风险事件
    """

    FRAME_FEATURE_NAMES = [
        "car_count",
        "truck_bus_count",
        "person_count",
        "cyclist_motorcycle_count",
        "total_object_count",
        "dynamic_object_area_ratio",
        "large_vehicle_ratio",
        "traffic_sign_count",
    ]

    TRAJ_FEATURE_NAMES = [
        "car_speed_mean",
        "car_accel_mean",
        "car_jerk_mean",
        "person_speed_mean",
        "cyclist_speed_mean",
        "pedestrian_crossing_count",
        "min_ttc",
        "risk_count",
    ]

    def __init__(
        self,
        model_path:    str,
        device:        str   = "cuda",
        conf_thres:    float = 0.25,
        iou_thres:     float = 0.45,
        ttc_threshold: float = 3.0,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("缺少 ultralytics，请执行: pip install ultralytics")

        self.device       = device
        self.conf_thres   = conf_thres
        self.iou_thres    = iou_thres
        self.ttc_threshold = ttc_threshold

        print(f"[YOLO] 加载模型: {model_path}  (device={device})")
        self.model = YOLO(model_path)
        self.model_names: Dict[int, str] = self.model.names  # {id: class_name}

        # ByteTrack 追踪器（使用 ultralytics 内置）
        self._tracker = None
        self._use_supervision = False
        self._init_tracker()
        print("[YOLO] 模型加载完成")

    # ──────────────────────────────────────────────────────────────────────────
    # 追踪器初始化
    # ──────────────────────────────────────────────────────────────────────────

    def _init_tracker(self):
        """尝试加载 supervision ByteTrack；失败则使用内置简易追踪器。"""
        try:
            import supervision as sv
            self._sv_tracker = sv.ByteTrack()
            self._use_supervision = True
            print("[YOLO] 使用 supervision.ByteTrack 追踪器")
        except ImportError:
            self._track_state: Dict[int, dict] = {}
            self._next_tid = 0
            self._use_supervision = False
            print("[YOLO] supervision 未安装，使用内置 IoU 追踪器（pip install supervision 可启用 ByteTrack）")

    def reset_tracker(self):
        """重置追踪器状态（每条新视频开始前调用）。"""
        if self._use_supervision:
            import supervision as sv
            self._sv_tracker = sv.ByteTrack()
        else:
            self._track_state = {}
            self._next_tid = 0

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────────────

    def extract_frame(self, frame: np.ndarray, timestamp: float) -> dict:
        """
        对单帧进行 YOLO 检测 + 追踪，返回帧级特征及私有追踪快照。

        Args:
            frame     : BGR numpy array，shape (H, W, 3)
            timestamp : 当前帧时间戳（秒）

        Returns:
            dict: 帧级特征 + 私有字段 "_tracks"（供窗口轨迹计算用）
        """
        h, w   = frame.shape[:2]
        img_area = h * w

        # ── YOLO 检测 ────────────────────────────────────────────────────────
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )
        boxes_raw = results[0].boxes  # ultralytics Boxes object

        if boxes_raw is None or len(boxes_raw) == 0:
            return self._empty_frame_features(timestamp)

        xyxy      = boxes_raw.xyxy.cpu().numpy()      # (N,4)
        confs     = boxes_raw.conf.cpu().numpy()       # (N,)
        cls_ids   = boxes_raw.cls.cpu().numpy().astype(int)  # (N,)

        # ── ByteTrack 追踪 ────────────────────────────────────────────────────
        track_ids = self._update_tracker(xyxy, confs, cls_ids, frame)

        # ── 统计帧级特征 ──────────────────────────────────────────────────────
        n_car   = int(np.sum(np.isin(cls_ids, list(_CAR_IDS))))
        n_tb    = int(np.sum(np.isin(cls_ids, list(_TRUCK_BUS))))
        n_pers  = int(np.sum(np.isin(cls_ids, list(_PERSON_IDS))))
        n_cyc   = int(np.sum(np.isin(cls_ids, list(_CYCLIST_IDS))))
        n_veh   = int(np.sum(np.isin(cls_ids, list(_VEHICLE_IDS))))
        n_sign  = int(np.sum(np.isin(cls_ids, list(_SIGN_IDS))))

        dyn_mask  = np.isin(cls_ids, list(_DYNAMIC_IDS))
        dyn_areas = np.array([
            (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
            for i in range(len(cls_ids))
            if dyn_mask[i]
        ])
        dyn_area_ratio = float(dyn_areas.sum()) / img_area if len(dyn_areas) > 0 else 0.0
        large_veh_ratio = n_tb / n_veh if n_veh > 0 else 0.0

        # ── 追踪快照（供轨迹计算）─────────────────────────────────────────────
        tracks_snapshot: List[Tuple] = []  # (track_id, class_group, cx, cy, area, timestamp)
        for i, cls_id in enumerate(cls_ids):
            if track_ids[i] < 0:
                continue
            x1, y1, x2, y2 = xyxy[i]
            cx   = (x1 + x2) / 2.0
            cy   = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)

            if cls_id in _CAR_IDS:
                grp = "car"
            elif cls_id in _TRUCK_BUS:
                grp = "car"        # 统一计入车辆速度
            elif cls_id in _PERSON_IDS:
                grp = "person"
            elif cls_id in _CYCLIST_IDS:
                grp = "cyclist"
            else:
                continue

            tracks_snapshot.append((int(track_ids[i]), grp, cx, cy, area, timestamp))

        return {
            "car_count":                int(n_car),
            "truck_bus_count":          int(n_tb),
            "person_count":             int(n_pers),
            "cyclist_motorcycle_count": int(n_cyc),
            "total_object_count":       int(n_car + n_tb + n_pers + n_cyc),
            "dynamic_object_area_ratio": float(dyn_area_ratio),
            "large_vehicle_ratio":      float(large_veh_ratio),
            "traffic_sign_count":       int(n_sign),
            "_tracks":                  tracks_snapshot,
            "_img_size":                (h, w),
        }

    @staticmethod
    def compute_window_traj_features(
        window_frames: List[dict],
        ttc_threshold: float = 3.0,
        fps:           float = 30.0,
    ) -> dict:
        """
        根据时间窗内的帧追踪快照计算窗口级轨迹特征。

        Args:
            window_frames : extract_frame() 返回值列表（含 "_tracks"）
            ttc_threshold : TTC 风险判定阈值（秒）
            fps           : 视频帧率（用于时间间隔估算）

        Returns:
            dict: 轨迹特征字典
        """
        # 汇总所有追踪点：{track_id: [(t, cx, cy, area, grp), ...]}
        traj: Dict[int, List] = defaultdict(list)
        for fd in window_frames:
            for (tid, grp, cx, cy, area, ts) in fd.get("_tracks", []):
                traj[tid].append((ts, cx, cy, area, grp))

        # 对每条轨迹按时间排序
        for tid in traj:
            traj[tid].sort(key=lambda x: x[0])

        # 逐轨迹计算速度序列
        car_speeds, car_accels, car_jerks = [], [], []
        person_speeds, cyclist_speeds = [], []
        approach_speeds: List[Tuple[float, float]] = []  # (speed_px_per_sec, current_area)
        pedestrian_crossings = 0

        img_w = window_frames[0].get("_img_size", (480, 640))[1] if window_frames else 640

        for tid, pts in traj.items():
            if len(pts) < 2:
                continue

            grp = pts[0][4]
            speeds = []
            for j in range(1, len(pts)):
                t0, x0, y0, _, _ = pts[j - 1]
                t1, x1, y1, a1, _ = pts[j]
                dt = t1 - t0
                if dt < 1e-4:
                    continue
                dist = math.hypot(x1 - x0, y1 - y0)
                speeds.append(dist / dt)

                # 检测行人横穿（水平位移跨越图像中心）
                if grp == "person":
                    cx_prev = x0
                    cx_curr = x1
                    center  = img_w / 2.0
                    if (cx_prev - center) * (cx_curr - center) < 0:
                        pedestrian_crossings += 1

            if not speeds:
                continue

            mean_spd = float(np.mean(speeds))
            if grp == "car":
                car_speeds.append(mean_spd)
                # 计算加速度序列
                if len(speeds) >= 2:
                    dt_unit  = 1.0 / fps
                    accels   = [abs(speeds[k] - speeds[k-1]) / dt_unit for k in range(1, len(speeds))]
                    car_accels.append(float(np.mean(accels)))
                    if len(accels) >= 2:
                        jerks = [abs(accels[k] - accels[k-1]) / dt_unit for k in range(1, len(accels))]
                        car_jerks.append(float(np.mean(jerks)))

                # TTC（基于 bbox 面积变化率）
                area_seq = [p[3] for p in pts]
                ttc = _estimate_ttc_from_area(area_seq, fps)
                if ttc is not None:
                    approach_speeds.append((mean_spd, ttc))

            elif grp == "person":
                person_speeds.append(mean_spd)
            elif grp == "cyclist":
                cyclist_speeds.append(mean_spd)

        # TTC 风险统计
        ttc_values = [ttc for _, ttc in approach_speeds if ttc > 0]
        min_ttc    = float(min(ttc_values)) if ttc_values else float("nan")
        risk_count = int(sum(1 for t in ttc_values if t < ttc_threshold))

        return {
            "car_speed_mean":            float(np.mean(car_speeds))   if car_speeds   else float("nan"),
            "car_accel_mean":            float(np.mean(car_accels))   if car_accels   else float("nan"),
            "car_jerk_mean":             float(np.mean(car_jerks))    if car_jerks    else float("nan"),
            "person_speed_mean":         float(np.mean(person_speeds)) if person_speeds else float("nan"),
            "cyclist_speed_mean":        float(np.mean(cyclist_speeds)) if cyclist_speeds else float("nan"),
            "pedestrian_crossing_count": int(pedestrian_crossings),
            "min_ttc":                   min_ttc,
            "risk_count":                risk_count,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────────

    def _update_tracker(
        self,
        xyxy:    np.ndarray,
        confs:   np.ndarray,
        cls_ids: np.ndarray,
        frame:   np.ndarray,
    ) -> List[int]:
        """更新追踪器，返回每个检测框对应的 track_id（-1 表示未追踪）。"""
        if self._use_supervision:
            return self._sv_update(xyxy, confs, cls_ids)
        else:
            return self._simple_iou_update(xyxy, confs, cls_ids)

    def _sv_update(self, xyxy, confs, cls_ids) -> List[int]:
        """使用 supervision ByteTrack 追踪。"""
        import supervision as sv
        dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=cls_ids,
        )
        tracked = self._sv_tracker.update_with_detections(dets)
        # 将追踪 ID 匹配回原始检测顺序（按 IoU 最大匹配）
        track_ids = [-1] * len(xyxy)
        for i in range(len(tracked)):
            t_box = tracked.xyxy[i]
            best_iou, best_j = 0.0, -1
            for j, d_box in enumerate(xyxy):
                iou = _box_iou(t_box, d_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou > 0.3:
                track_ids[best_j] = int(tracked.tracker_id[i])
        return track_ids

    def _simple_iou_update(self, xyxy, confs, cls_ids) -> List[int]:
        """简易 IoU 追踪（当 supervision 不可用时的备用方案）。"""
        track_ids = []
        new_state  = {}

        for i in range(len(xyxy)):
            box     = xyxy[i]
            best_id = -1
            best_io = 0.5  # IoU 匹配阈值

            for tid, tbox in self._track_state.items():
                iou = _box_iou(box, tbox["box"])
                if iou > best_io and int(cls_ids[i]) == tbox["cls"]:
                    best_io = iou
                    best_id = tid

            if best_id >= 0:
                tid = best_id
            else:
                tid = self._next_tid
                self._next_tid += 1

            new_state[tid] = {"box": box, "cls": int(cls_ids[i])}
            track_ids.append(tid)

        self._track_state = new_state
        return track_ids

    @staticmethod
    def _empty_frame_features(timestamp: float) -> dict:
        return {
            "car_count":                0,
            "truck_bus_count":          0,
            "person_count":             0,
            "cyclist_motorcycle_count": 0,
            "total_object_count":       0,
            "dynamic_object_area_ratio": 0.0,
            "large_vehicle_ratio":      0.0,
            "traffic_sign_count":       0,
            "_tracks":                  [],
            "_img_size":                (480, 640),
        }


# ==================== 辅助函数 ====================

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个 xyxy 格式 bounding box 的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _estimate_ttc_from_area(area_seq: List[float], fps: float) -> Optional[float]:
    """
    用 bounding box 面积变化率近似估算 TTC（Time-to-Collision）。
    TTC ≈ current_area / (d(area)/dt)
    只对接近摄像机（面积增大）的目标计算。
    """
    if len(area_seq) < 2:
        return None
    dt = 1.0 / fps
    diffs = [(area_seq[k] - area_seq[k-1]) / dt for k in range(1, len(area_seq))]
    mean_rate = np.mean(diffs)
    if mean_rate <= 0:
        return None  # 目标在远离
    current_area = area_seq[-1]
    if current_area <= 0:
        return None
    return float(current_area / mean_rate)

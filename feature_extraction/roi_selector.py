"""
ROI 区域交互式选择工具

使用方法:
    python roi_selector.py <视频路径>
    python roi_selector.py  (无参数时弹出文件选择对话框)

鼠标操作:
    左键拖拽     - 绘制矩形 ROI
    右键单击     - 添加多边形顶点
    按 R         - 重置当前绘制
    按 C         - 确认 ROI 并输出坐标
    按 N         - 随机换一帧
    按 Q / ESC  - 退出
"""

import cv2
import numpy as np
import random
import sys
import os

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False


# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────
class ROIState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.drawing = False          # 正在拖拽矩形
        self.rect_start = (-1, -1)    # 矩形起点
        self.rect_end   = (-1, -1)    # 矩形终点
        self.rect_done  = False       # 矩形已确定

        self.poly_pts   = []          # 多边形顶点列表
        self.poly_done  = False       # 多边形已关闭

        self.mode = "rect"            # "rect" | "poly"


state = ROIState()
base_frame = None      # 当前帧（只读，用于刷新画布）
display_frame = None   # 实时绘制帧


# ─────────────────────────────────────────────
# 鼠标回调
# ─────────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global display_frame

    if state.mode == "rect":
        _rect_callback(event, x, y)
    else:
        _poly_callback(event, x, y)

    _redraw()


def _rect_callback(event, x, y):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.rect_start = (x, y)
        state.rect_end   = (x, y)
        state.rect_done  = False

    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        state.rect_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing   = False
        state.rect_end  = (x, y)
        state.rect_done = True


def _poly_callback(event, x, y):
    if event == cv2.EVENT_RBUTTONDOWN:
        state.poly_pts.append((x, y))
        if len(state.poly_pts) > 1:
            state.poly_done = False   # 还在添加点

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # 双击左键关闭多边形
        if len(state.poly_pts) >= 3:
            state.poly_done = True


def _redraw():
    global display_frame
    display_frame = base_frame.copy()

    # 绘制操作说明
    _draw_help(display_frame)

    if state.mode == "rect":
        if state.rect_start != (-1, -1) and state.rect_end != (-1, -1):
            color = (0, 255, 0) if state.rect_done else (0, 200, 255)
            cv2.rectangle(display_frame, state.rect_start, state.rect_end, color, 2)
            # 显示尺寸
            w = abs(state.rect_end[0] - state.rect_start[0])
            h = abs(state.rect_end[1] - state.rect_start[1])
            mid = ((state.rect_start[0] + state.rect_end[0]) // 2,
                   (state.rect_start[1] + state.rect_end[1]) // 2)
            cv2.putText(display_frame, f"{w}x{h}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    else:  # poly
        pts = state.poly_pts
        for i, pt in enumerate(pts):
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i), (pt[0]+6, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(pts) >= 2:
            arr = np.array(pts, np.int32)
            if state.poly_done:
                cv2.polylines(display_frame, [arr], True, (0, 255, 0), 2)
            else:
                cv2.polylines(display_frame, [arr], False, (0, 200, 255), 2)

    cv2.imshow("ROI Selector", display_frame)


def _draw_help(img):
    lines = [
        f"模式: {'矩形(左键拖拽)' if state.mode == 'rect' else '多边形(右键添点,左键双击闭合)'}",
        "R=重置  C=确认输出  N=换帧  M=切换模式  Q/ESC=退出",
    ]
    for i, line in enumerate(lines):
        y = 22 + i * 24
        cv2.rectangle(img, (0, y - 16), (len(line) * 11 + 8, y + 6), (0, 0, 0), -1)
        cv2.putText(img, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


# ─────────────────────────────────────────────
# 视频工具
# ─────────────────────────────────────────────
def get_video_info(cap, path):
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc  = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)])
    duration = total / fps if fps > 0 else 0

    info = {
        "路径":     os.path.abspath(path),
        "分辨率":   f"{w} x {h}",
        "帧率":     f"{fps:.2f} FPS",
        "总帧数":   total,
        "时长":     f"{duration:.2f} 秒  ({int(duration//60):02d}:{int(duration%60):02d})",
        "编码格式": fourcc.strip(),
        "文件大小": f"{os.path.getsize(path) / 1024 / 1024:.2f} MB",
    }
    return info, total, fps, w, h


def grab_random_frame(cap, total_frames):
    idx = random.randint(0, max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        idx = 0
    return frame, idx


# ─────────────────────────────────────────────
# ROI 输出
# ─────────────────────────────────────────────
def print_roi(frame_idx, frame_w, frame_h):
    sep = "─" * 52
    print(f"\n{sep}")
    print("  ROI 区域信息")
    print(sep)

    if state.mode == "rect" and state.rect_done:
        x1 = min(state.rect_start[0], state.rect_end[0])
        y1 = min(state.rect_start[1], state.rect_end[1])
        x2 = max(state.rect_start[0], state.rect_end[0])
        y2 = max(state.rect_start[1], state.rect_end[1])
        w  = x2 - x1
        h  = y2 - y1
        area = w * h
        rel_x1 = x1 / frame_w
        rel_y1 = y1 / frame_h
        rel_x2 = x2 / frame_w
        rel_y2 = y2 / frame_h

        print(f"  类型        : 矩形")
        print(f"  参考帧      : 第 {frame_idx} 帧")
        print(f"  左上角      : ({x1}, {y1})")
        print(f"  右下角      : ({x2}, {y2})")
        print(f"  宽 x 高     : {w} x {h}  px")
        print(f"  面积        : {area} px²")
        print(f"  归一化坐标  : ({rel_x1:.4f}, {rel_y1:.4f}, {rel_x2:.4f}, {rel_y2:.4f})")
        print(f"\n  Python 字典 :")
        print(f"    roi = {{'type': 'rect', 'x1': {x1}, 'y1': {y1}, 'x2': {x2}, 'y2': {y2}}}")
        print(f"\n  cv2.rectangle 参数 :")
        print(f"    cv2.rectangle(frame, ({x1}, {y1}), ({x2}, {y2}), color, thickness)")

    elif state.mode == "poly" and state.poly_done and len(state.poly_pts) >= 3:
        pts = state.poly_pts
        arr = np.array(pts)
        cx  = int(arr[:, 0].mean())
        cy  = int(arr[:, 1].mean())
        area = cv2.contourArea(arr.astype(np.float32))
        bbox = cv2.boundingRect(arr.astype(np.int32))  # x, y, w, h

        print(f"  类型        : 多边形")
        print(f"  参考帧      : 第 {frame_idx} 帧")
        print(f"  顶点数      : {len(pts)}")
        print(f"  顶点坐标    :")
        for i, pt in enumerate(pts):
            print(f"    [{i}]  ({pt[0]:4d}, {pt[1]:4d})")
        print(f"  质心        : ({cx}, {cy})")
        print(f"  面积        : {area:.1f} px²")
        print(f"  外接矩形    : x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        print(f"\n  Python 列表 :")
        print(f"    roi_pts = {pts}")
        print(f"\n  NumPy 数组  :")
        print(f"    roi_pts = np.array({pts}, np.int32)")

    else:
        print("  尚未完成 ROI 绘制，请先绘制后按 C 确认。")

    print(sep)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def select_video_path():
    """无参数时弹出文件选择对话框（可选）"""
    if HAS_TK:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.ts"), ("所有文件", "*.*")]
        )
        root.destroy()
        return path
    return None


def main():
    global base_frame, display_frame

    # ── 1. 获取视频路径 ──────────────────────────
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        video_path = select_video_path()
        if not video_path:
            print("未提供视频路径，退出。")
            sys.exit(0)

    if not os.path.isfile(video_path):
        print(f"[错误] 文件不存在: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        sys.exit(1)

    # ── 2. 输出视频信息 ──────────────────────────
    info, total_frames, fps, frame_w, frame_h = get_video_info(cap, video_path)

    sep = "═" * 52
    print(f"\n{sep}")
    print("  视频信息")
    print(sep)
    for k, v in info.items():
        print(f"  {k:<8}: {v}")
    print(sep)
    print()
    print("  操作说明:")
    print("    左键拖拽         — 绘制矩形 ROI")
    print("    右键单击         — 添加多边形顶点")
    print("    左键双击         — 闭合多边形")
    print("    R                — 重置绘制")
    print("    C                — 确认并输出 ROI 坐标")
    print("    N                — 随机换一帧")
    print("    M                — 切换矩形 / 多边形模式")
    print("    Q 或 ESC         — 退出")
    print(sep)
    print()

    # ── 3. 随机抽取一帧 ──────────────────────────
    base_frame, current_frame_idx = grab_random_frame(cap, total_frames)
    if base_frame is None:
        print("[错误] 无法读取帧，退出。")
        cap.release()
        sys.exit(1)

    print(f"  当前显示帧: 第 {current_frame_idx} 帧  "
          f"(时间戳: {current_frame_idx/fps:.2f}s)")
    print()

    # ── 4. 显示窗口并绑定鼠标 ────────────────────
    cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
    # 自适应屏幕，最大 1280×720
    max_w, max_h = 1280, 720
    scale = min(max_w / frame_w, max_h / frame_h, 1.0)
    cv2.resizeWindow("ROI Selector",
                     int(frame_w * scale), int(frame_h * scale))
    cv2.setMouseCallback("ROI Selector", mouse_callback)

    _redraw()

    # ── 5. 主循环 ────────────────────────────────
    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):   # Q / ESC
            break

        elif key in (ord('r'), ord('R')):      # 重置
            state.reset()
            _redraw()
            print("  已重置 ROI。")

        elif key in (ord('c'), ord('C')):      # 确认输出
            print_roi(current_frame_idx, frame_w, frame_h)

        elif key in (ord('n'), ord('N')):      # 换帧
            base_frame, current_frame_idx = grab_random_frame(cap, total_frames)
            state.reset()
            _redraw()
            print(f"  已切换到第 {current_frame_idx} 帧  "
                  f"(时间戳: {current_frame_idx/fps:.2f}s)")

        elif key in (ord('m'), ord('M')):      # 切换模式
            state.reset()
            state.mode = "poly" if state.mode == "rect" else "rect"
            _redraw()
            mode_name = "多边形" if state.mode == "poly" else "矩形"
            print(f"  已切换到 {mode_name} 模式。")

        # 窗口被关闭
        if cv2.getWindowProperty("ROI Selector", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n  已退出 ROI 选择工具。\n")


if __name__ == "__main__":
    main()

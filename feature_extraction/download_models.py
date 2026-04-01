"""
预训练模型下载脚本
下载并初始化以下三个特征提取模型：
  1. SegFormer (nvidia/segformer-b1-finetuned-cityscapes-1024-1024)  ← Hugging Face
  2. yolo26-bdd100k (shravanda/yolo26-bdd100k)                       ← Hugging Face
  3. YOLOPv2 (CAIC-AD/YOLOPv2)                                       ← GitHub + Release

运行方式：
    python download_models.py
"""

import os
import subprocess
import sys
import urllib.request

# ── 将 feature_extraction 加入路径以便导入 config ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PRETRAIN_MODEL_DIR,
    SEGFORMER_MODEL_DIR,
    SEGFORMER_REPO_ID,
    YOLOPV2_REPO_DIR,
    YOLOPV2_WEIGHTS,
    YOLO_MODEL_PATH,
)

# ==================== 工具函数 ====================

def _print_step(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def _download_with_progress(url: str, dest_path: str):
    """带进度条的 HTTP 下载。"""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    def _reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            bar = "#" * int(pct / 2)
            print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    print()


# ==================== 1. SegFormer ====================

def download_segformer():
    """从 Hugging Face 下载 SegFormer 模型（使用 huggingface_hub）。"""
    _print_step("1/3  SegFormer — nvidia/segformer-b1-finetuned-cityscapes-1024-1024")

    # 检查是否已下载（config.json 存在视为完整）
    config_path = os.path.join(SEGFORMER_MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        print(f"  [跳过] SegFormer 已存在: {SEGFORMER_MODEL_DIR}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [ERROR] 缺少 huggingface_hub，请执行: pip install huggingface-hub")
        sys.exit(1)

    print(f"  正在下载至: {SEGFORMER_MODEL_DIR}")
    snapshot_download(
        repo_id=SEGFORMER_REPO_ID,
        local_dir=SEGFORMER_MODEL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # 仅保留 PyTorch 权重
    )
    print(f"  [完成] SegFormer 已下载至 {SEGFORMER_MODEL_DIR}")


# ==================== 2. YOLO BDD100K ====================

def download_yolo_bdd100k():
    """从 Hugging Face 下载 yolo26-bdd100k 模型权重。"""
    _print_step("2/3  YOLO BDD100K — shravanda/yolo26-bdd100k")

    if os.path.exists(YOLO_MODEL_PATH):
        size_mb = os.path.getsize(YOLO_MODEL_PATH) / 1024 / 1024
        print(f"  [跳过] YOLO BDD100K 已存在: {YOLO_MODEL_PATH}  ({size_mb:.1f} MB)")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [ERROR] 缺少 huggingface_hub，请执行: pip install huggingface-hub")
        sys.exit(1)

    print(f"  正在下载 yolo26-bdd100k.pt ...")
    os.makedirs(PRETRAIN_MODEL_DIR, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="shravanda/yolo26-bdd100k",
        filename="yolo26-bdd100k.pt",
        local_dir=PRETRAIN_MODEL_DIR,
    )
    print(f"  [完成] YOLO BDD100K 已下载至 {downloaded}")


# ==================== 3. YOLOPv2 ====================

def download_yolopv2():
    """克隆 YOLOPv2 仓库并下载模型权重。"""
    _print_step("3/3  YOLOPv2 — CAIC-AD/YOLOPv2")

    # ── 3a. 克隆仓库代码 ──────────────────────────────────────────────────────
    # 完整仓库必须包含 models/ 目录，否则视为不完整，需要重新克隆
    _repo_complete = (
        os.path.isdir(YOLOPV2_REPO_DIR)
        and os.path.isdir(os.path.join(YOLOPV2_REPO_DIR, "models"))
        and os.path.isfile(os.path.join(YOLOPV2_REPO_DIR, "utils", "torch_utils.py"))
    )

    if _repo_complete:
        print(f"  [跳过克隆] YOLOPv2 仓库已存在且完整: {YOLOPV2_REPO_DIR}")
    else:
        if os.path.isdir(YOLOPV2_REPO_DIR) and os.listdir(YOLOPV2_REPO_DIR):
            print(f"  [警告] YOLOPv2 仓库不完整（缺少 models/ 或 utils/torch_utils.py），将删除后重新克隆 ...")
            import shutil
            import stat

            def _remove_readonly(func, path, _):
                """Windows .git 目录含只读文件，删除前先清除只读属性。"""
                os.chmod(path, stat.S_IWRITE)
                func(path)

            shutil.rmtree(YOLOPV2_REPO_DIR, onerror=_remove_readonly)

        print(f"  正在 Git clone YOLOPv2 仓库 ...")
        os.makedirs(os.path.dirname(YOLOPV2_REPO_DIR), exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/CAIC-AD/YOLOPv2.git", YOLOPV2_REPO_DIR],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [ERROR] Git clone 失败:\n{result.stderr}")
            print("  请手动执行: git clone https://github.com/CAIC-AD/YOLOPv2.git "
                  f"{YOLOPV2_REPO_DIR}")
            return
        print(f"  [完成] YOLOPv2 仓库已克隆至 {YOLOPV2_REPO_DIR}")

    # ── 3b. 下载模型权重 ──────────────────────────────────────────────────────
    # 权重文件最小应为 35 MB，否则视为下载不完整
    _MIN_WEIGHT_MB = 35
    if os.path.exists(YOLOPV2_WEIGHTS):
        size_mb = os.path.getsize(YOLOPV2_WEIGHTS) / 1024 / 1024
        if size_mb >= _MIN_WEIGHT_MB:
            print(f"  [跳过下载] YOLOPv2 权重已存在: {YOLOPV2_WEIGHTS}  ({size_mb:.1f} MB)")
            return
        else:
            print(f"  [警告] 权重文件疑似损坏（{size_mb:.1f} MB < {_MIN_WEIGHT_MB} MB），将删除后重新下载 ...")
            os.remove(YOLOPV2_WEIGHTS)

    weights_url = (
        "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
    )
    print(f"  正在下载 yolopv2.pt ...")
    try:
        _download_with_progress(weights_url, YOLOPV2_WEIGHTS)
        print(f"  [完成] YOLOPv2 权重已保存至 {YOLOPV2_WEIGHTS}")
    except Exception as e:
        print(f"  [ERROR] 下载失败: {e}")
        print(f"  请手动下载: {weights_url}")
        print(f"  并放置至:   {YOLOPV2_WEIGHTS}")


# ==================== 汇总检查 ====================

def verify_all():
    """检查所有模型文件是否存在，输出汇总状态。"""
    _print_step("模型文件检查汇总")
    checks = {
        "SegFormer config.json":  os.path.join(SEGFORMER_MODEL_DIR, "config.json"),
        "YOLO BDD100K (.pt)":     YOLO_MODEL_PATH,
        "YOLOPv2 weights (.pt)":  YOLOPV2_WEIGHTS,
    }
    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ 缺失"
        if not exists:
            all_ok = False
        print(f"  {status}  {name}")
        print(f"        路径: {path}")
    if all_ok:
        print("\n  所有模型文件就绪，可以开始特征提取！")
    else:
        print("\n  部分模型缺失，请重新运行 download_models.py 或手动下载。")


# ==================== 主入口 ====================

if __name__ == "__main__":
    os.makedirs(PRETRAIN_MODEL_DIR, exist_ok=True)
    download_segformer()
    download_yolo_bdd100k()
    download_yolopv2()
    verify_all()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kreps2clip_infer_11_22.py

功能：
- 从 KToken-Reps payload(json) 读取 reps (K, D)
- 根据 adapter ckpt 推断 d_in，构造 AdapterMLP
- 把 reps 映射成固定长度向量 x_flat（与训练时一致的展开/截断/补零）
- 用 ckpt 里的 x_mean / x_std 做标准化
- 通过 adapter 得到语义向量 z_pred（维度 D_clip）
- 载入 build_image_sem_db_11_22.py 生成的语义库 (stems + embeds)
- 计算 z_pred 与库中每个 embedding 的余弦相似度，做 top-k 检索
- 从 image_root 里找到：
    - GT: stem 对应的一张原图
    - 检索出的 top-k 图
  拼成一张横向对比图保存

用法示例：
python kreps2clip_infer_11_22.py \
  --payload_root data_rep_k32 \
  --stem coco_000001 \
  --adapter_ckpt runs/kreps2clip_exp1/adapter_best.pth \
  --sem_db runs/sem_db/image_sem_db_11_22.pt \
  --image_root data/image \
  --outdir runs/kreps2clip_infer_11_22 \
  --topk 5
"""

import argparse
import json
import zlib
import base64
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms as T


def log(*a):
    print("[kreps2clip-infer-11-22]", *a, flush=True)


# ---------- 读取 KToken-Reps payload ----------

def load_kreps_json(p: Path) -> np.ndarray:
    """
    读取 KToken-Reps JSON，反量化成 float32 reps，shape=(K, D)
    JSON 格式：见 ssss.py 生成部分
        {
          "n_tokens": K,
          "dim": D,
          "scales": [K],
          "data": base64(zlib(int8[K, D]))
        }
    """
    obj = json.loads(p.read_text())
    n = int(obj["n_tokens"])
    d = int(obj["dim"])
    scales = np.asarray(obj["scales"], dtype=np.float32)  # (K,)
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(zlib.decompress(raw), dtype=np.int8)  # (K*D,)
    arr = arr.reshape(n, d).astype(np.float32)                # (K, D)
    reps = arr * scales[:, None]
    return reps


# ---------- AdapterMLP + 从 ckpt 恢复结构 ----------

class AdapterMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, n_layers: int, p_drop: float = 0.0):
        super().__init__()
        layers = []
        last = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(p_drop)]
            last = hidden
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())


def build_adapter_from_ckpt(ckpt_path: Path, device: torch.device):
    """
    从训练好后的 ckpt 中恢复：
      - AdapterMLP 结构 (d_in, d_out, hidden, n_layers)
      - state_dict
      - 训练时记录的 x_mean / x_std（scalar）
    ckpt 兼容两种格式：
      1) {"state_dict": ..., "d_in": ..., "d_out": ..., "x_mean": ..., "x_std": ...}
      2) 直接 state_dict（则从 Linear 权重形状推断）
    """
    blob = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    else:
        sd = blob

    # 找到所有线性层权重
    lin_keys = [k for k, v in sd.items()
                if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(".weight")]
    assert lin_keys, f"no Linear weight found in ckpt: {ckpt_path}"
    lin_keys = sorted(lin_keys)

    first_w = sd[lin_keys[0]]
    last_w = sd[lin_keys[-1]]
    d_in_sd = int(first_w.shape[1])
    hidden_sd = int(first_w.shape[0]) if len(lin_keys) > 1 else int(last_w.shape[0])
    d_out_sd = int(last_w.shape[0])
    n_layers = len(lin_keys)

    # meta 中的 d_in/d_out 仅做 sanity check
    d_in_meta = int(blob.get("d_in", d_in_sd)) if isinstance(blob, dict) else d_in_sd
    d_out_meta = int(blob.get("d_out", d_out_sd)) if isinstance(blob, dict) else d_out_sd
    if d_in_meta != d_in_sd:
        log(f"⚠️ ckpt.d_in={d_in_meta} but inferred={d_in_sd}, use inferred")
    if d_out_meta != d_out_sd:
        log(f"⚠️ ckpt.d_out={d_out_meta} but inferred={d_out_sd}, use inferred")

    d_in = d_in_sd
    d_out = d_out_sd
    hidden = hidden_sd

    # 读取 scalar 标准化参数（没有就退化为 mean=0,std=1）
    x_mean = float(blob.get("x_mean", 0.0)) if isinstance(blob, dict) else 0.0
    x_std = float(blob.get("x_std", 1.0)) if isinstance(blob, dict) else 1.0

    log(f"ckpt: d_in={d_in}, d_out={d_out}, hidden={hidden}, n_layers={n_layers}")
    log(f"x_mean={x_mean:.6f}, x_std={x_std:.6f}")

    adapter = AdapterMLP(d_in=d_in, d_out=d_out, hidden=hidden, n_layers=n_layers, p_drop=0.0)
    adapter.load_state_dict(sd, strict=True)
    adapter.to(device)
    adapter.eval()

    return adapter, d_in, d_out, x_mean, x_std


# ---------- payload (K,D) → x_flat(d_in) ----------

def make_input_from_reps(reps: np.ndarray, d_in: int) -> Tuple[np.ndarray, int, int]:
    """
    reps: (K, D)
    d_in: adapter 期望的输入维度

    逻辑与训练阶段一致：
      - 令 K_eff = d_in / D，必须是整数
      - 若 K > K_eff：均匀采样到 K_eff
      - 若 K < K_eff：后面补零行
      - 展平为 (d_in,)
    """
    K, D = reps.shape
    if d_in % D != 0:
        raise RuntimeError(f"d_in={d_in} not divisible by D={D}, check max_tokens / dim")

    K_eff = d_in // D
    if K > K_eff:
        idxs = np.linspace(0, K - 1, K_eff, dtype=int)
        reps_used = reps[idxs]
    elif K < K_eff:
        pad = np.zeros((K_eff - K, D), dtype=np.float32)
        reps_used = np.concatenate([reps, pad], axis=0)
    else:
        reps_used = reps

    assert reps_used.shape == (K_eff, D)
    x_flat = reps_used.reshape(-1).astype(np.float32)  # (d_in,)
    return x_flat, K, K_eff


# ---------- 找图像 & 拼可视化 ----------

def find_one_image_for_stem(stem: str, image_root: Path) -> Path:
    """
    找一张与 stem 对应的图像：
      - {stem}_*.jpg
      - {stem}.jpg
    """
    patterns = [
        f"{stem}_*.jpg",
        f"{stem}.jpg",
        f"{stem}_*.png",
        f"{stem}.png",
    ]
    for pat in patterns:
        xs = sorted(image_root.glob(pat))
        if xs:
            return xs[0]
    return None


def make_compare_grid(gt_path: Path,
                      retrieved_paths,
                      out_path: Path,
                      size: int = 256):
    """
    横向拼一张图：
      [GT] | [Top-1] | [Top-2] | ...
    """
    tfm = T.Compose([T.Resize((size, size))])
    panels = []
    if gt_path is not None and gt_path.exists():
        panels.append(tfm(Image.open(gt_path).convert("RGB")))
    for p in retrieved_paths:
        if p is None:
            continue
        panels.append(tfm(Image.open(p).convert("RGB")))

    if not panels:
        return

    W = size * len(panels)
    H = size
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    for i, im in enumerate(panels):
        canvas.paste(im, (i * size, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------- 主逻辑 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload_root", type=str, required=True,
                    help="包含 *_payload.json 的目录，例如 data_rep_k32")
    ap.add_argument("--stem", type=str, required=True,
                    help="样本 stem，例如 coco_000001")
    ap.add_argument("--adapter_ckpt", type=str, required=True,
                    help="kreps2clip_train 训练得到的 adapter ckpt")
    ap.add_argument("--sem_db", type=str, required=True,
                    help="build_image_sem_db_11_22.py 生成的 .pt 语义库")
    ap.add_argument("--image_root", type=str, required=True,
                    help="原始图像根目录，例如 data/image")
    ap.add_argument("--outdir", type=str, required=True,
                    help="可视化输出目录")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    payload_root = Path(args.payload_root).resolve()
    payload_path = payload_root / f"{args.stem}_payload.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"payload not found: {payload_path}")

    adapter_ckpt = Path(args.adapter_ckpt).resolve()
    sem_db_path = Path(args.sem_db).resolve()
    image_root = Path(args.image_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    log(f"payload = {payload_path}")
    log(f"adapter_ckpt = {adapter_ckpt}")
    log(f"sem_db = {sem_db_path}")
    log(f"image_root = {image_root}")
    log(f"outdir = {outdir}")
    log(f"device = {device}")

    # 1) 加载 adapter
    adapter, d_in, d_out, x_mean, x_std = build_adapter_from_ckpt(adapter_ckpt, device)

    # 2) 载入语义库
    db = torch.load(str(sem_db_path), map_location="cpu")
    stems_db = db["stems"]
    embeds_db = db["embeds"].float()   # (N, D_clip)
    log(f"sem_db: N={embeds_db.shape[0]}, D={embeds_db.shape[1]}")

    if embeds_db.shape[1] != d_out:
        raise RuntimeError(f"dimension mismatch: adapter d_out={d_out} vs sem_db D={embeds_db.shape[1]}")

    # 3) 读取 payload → reps
    reps = load_kreps_json(payload_path)   # (K, D)
    log(f"payload reps shape = {reps.shape}")

    x_flat, K_raw, K_eff = make_input_from_reps(reps, d_in=d_in)
    log(f"K_raw={K_raw}, K_eff(used)={K_eff}, d_in={d_in}")

    # 标准化
    x_norm = (x_flat - x_mean) / (x_std + 1e-6)
    x = torch.from_numpy(x_norm).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, d_in)

    # 4) 通过 adapter 得到语义向量
    with torch.no_grad():
        z_pred = adapter(x)    # (1, d_out)
        z_pred = F.normalize(z_pred, dim=-1)  # (1, D_clip)

    # 5) 与库中所有 embedding 做余弦相似度
    embeds_db_n = F.normalize(embeds_db, dim=-1)  # (N, D)
    sims = (z_pred.cpu() @ embeds_db_n.T).squeeze(0)  # (N,)
    vals, idxs = torch.topk(sims, k=min(args.topk, sims.numel()))
    vals = vals.tolist()
    idxs = idxs.tolist()

    log("top-k retrieval:")
    for rank, (i, s) in enumerate(zip(idxs, vals), start=1):
        log(f"  #{rank}: stem={stems_db[i]}  sim={s:.4f}")

    # 6) 找 GT 图 + top-k 图像路径
    gt_path = find_one_image_for_stem(args.stem, image_root)

    retrieved_paths = []
    for i in idxs:
        stem_i = stems_db[i]
        p = find_one_image_for_stem(stem_i, image_root)
        retrieved_paths.append(p)

    # 7) 拼可视化
    out_img = outdir / f"{args.stem}_top{len(retrieved_paths)}.png"
    make_compare_grid(gt_path, retrieved_paths, out_img, size=256)
    log(f"saved compare image -> {out_img}")


if __name__ == "__main__":
    main()

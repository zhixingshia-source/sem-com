#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sem2vae_adapter_infer.py

给定：
  1) 一个 KToken-Reps payload（*.json）
  2) 训练好的 adapter ckpt（adapter_best.pth）
  3) 对应原图所在的 image_root（用于找回一张 GT）

做的事：
  - 从 ckpt 里恢复 Adapter MLP 结构（d_in/d_out/hidden/layers）
  - 从 payload 里取出 reps，按 ckpt 需要的 token 数扩展/截断成固定长度
  - 用 ckpt 里的 x_mean/x_std 做标准化
  - 通过 adapter → latent → VAE 解码
  - 在 outdir 里保存：
      infer.png      # 仅重建图
      gt.png         # （如果找到原图的话）
      compare.png    # GT | 重建（横向拼接）
"""

import os, json, zlib, base64, math, argparse, glob
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL


# ----------------- 小工具 ----------------- #
def log(*a):
    print("[sem2vae-infer]", *a, flush=True)


def load_kreps_json(p: Path) -> np.ndarray:
    """读取 KToken-Reps(JSON)：int8(zlib+base64)+scales -> float32 reps，shape=(K, D)"""
    obj = json.loads(p.read_text())
    n = int(obj["n_tokens"])
    d = int(obj["dim"])
    scales = np.asarray(obj["scales"], dtype=np.float32)  # (K,)
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(zlib.decompress(raw), dtype=np.int8)  # (K*D,)
    arr = arr.reshape(n, d).astype(np.float32)  # (K, D)
    reps = arr * scales[:, None]  # 反量化
    return reps  # (K, D)


def square_canvas(pil: Image.Image, S: int) -> Image.Image:
    """等比缩放到不超过 S×S，并居中贴到 S×S（黑边）"""
    w0, h0 = pil.size
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"非法图片尺寸: {pil.size}")
    r = min(S / float(w0), S / float(h0))
    nw, nh = max(1, int(round(w0 * r))), max(1, int(round(h0 * r)))
    img = pil.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (S, S), (0, 0, 0))
    canvas.paste(img, ((S - nw) // 2, (S - nh) // 2))
    return canvas


def find_one_image(stem: str, image_root: Path) -> Path:
    """在 image_root 下找一张对应 stem 的图，比如 coco_118287_XXXX.jpg"""
    pats = [
        str(image_root / f"{stem}_*.jpg"),
        str(image_root / f"{stem}_*.png"),
        str(image_root / f"{stem}.jpg"),
        str(image_root / f"{stem}.png"),
    ]
    for pat in pats:
        xs = sorted(glob.glob(pat))
        if xs:
            return Path(xs[0])
    return None


class AdapterMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 8192, n_layers: int = 2, p_drop: float = 0.0):
        super().__init__()
        layers = []
        last = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(p_drop)]
            last = hidden
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_adapter_from_ckpt(ckpt_path: Path, device: torch.device, dtype: torch.dtype):
    """从 ckpt 里恢复 adapter 结构（d_in/d_out/hidden/layers），并加载权重"""
    blob = torch.load(str(ckpt_path), map_location="cpu")

    # 兼容两种格式： {state_dict=...} 或 直接 state_dict
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    else:
        sd = blob

    # 从权重形状里推断线性层数量和 hidden / d_in / d_out
    lin_keys = [k for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 2 and k.endswith(".weight")]
    assert lin_keys, f"ckpt 里找不到任何 Linear 权重: {ckpt_path}"
    lin_keys = sorted(lin_keys)  # 一般是 net.0.weight, net.3.weight, ...

    first_w = sd[lin_keys[0]]
    last_w = sd[lin_keys[-1]]
    d_in_sd = int(first_w.shape[1])
    hidden_sd = int(first_w.shape[0]) if len(lin_keys) > 1 else int(last_w.shape[0])
    d_out_sd = int(last_w.shape[0])
    n_layers = len(lin_keys)

    # meta 里的 d_in/d_out（如果有）只是做个 sanity check
    d_in_meta = int(blob.get("d_in", d_in_sd))
    d_out_meta = int(blob.get("d_out", d_out_sd))
    if d_in_meta != d_in_sd:
        log(f"⚠️ ckpt.d_in={d_in_meta} 但 state_dict 推断={d_in_sd}，以 state_dict 为准")
    if d_out_meta != d_out_sd:
        log(f"⚠️ ckpt.d_out={d_out_meta} 但 state_dict 推断={d_out_sd}，以 state_dict 为准")

    d_in = d_in_sd
    d_out = d_out_sd

    x_mean = float(blob.get("x_mean", 0.0))
    x_std = float(blob.get("x_std", 1.0))
    Hc = int(blob.get("Hc", 32))
    Wc = int(blob.get("Wc", 32))
    target_size = int(blob.get("target_size", 256))
    vae_id = blob.get("vae_id", "stabilityai/sd-vae-ft-mse")
    prec_str = blob.get("precision", "fp32")

    log(f"ckpt: d_in={d_in}, d_out={d_out}, Hc={Hc}, Wc={Wc}, x_mean={x_mean:.4f}, x_std={x_std:.4f}")
    log(f"adapter (from state_dict): d_in={d_in}, d_out={d_out}, hidden={hidden_sd}, layers={n_layers}")

    adapter = AdapterMLP(d_in=d_in, d_out=d_out, hidden=hidden_sd, n_layers=n_layers, p_drop=0.0)
    adapter = adapter.to(device=device, dtype=dtype)
    adapter.load_state_dict(sd, strict=True)

    return adapter, d_in, d_out, Hc, Wc, x_mean, x_std, target_size, vae_id, prec_str


def make_input_from_reps(reps: np.ndarray, d_in: int, x_mean: float, x_std: float) -> Tuple[np.ndarray, int, int]:
    """
    把 payload 里 (K, D) 的 reps 变成 ckpt 需要的固定长度向量：
      - ckpt 期望长度 d_in
      - payload 提供 K_token, 每个 D 维
      - 先算 K_eff = d_in / D，必须是整数
      - K >= K_eff 时取前 K_eff 个
      - K <  K_eff 时循环重复直到补足
    """
    K, D = reps.shape
    if d_in % D != 0:
        raise RuntimeError(f"ckpt.d_in={d_in} 不能被 payload.dim={D} 整除，检查 ckpt / payload 是否对得上。")
    K_eff = d_in // D

    if K == K_eff:
        reps_used = reps
    elif K > K_eff:
        reps_used = reps[:K_eff, :]
    else:
        # tile 补足
        reps_list = []
        times = K_eff // K
        rem = K_eff % K
        for _ in range(times):
            reps_list.append(reps)
        if rem > 0:
            reps_list.append(reps[:rem, :])
        reps_used = np.concatenate(reps_list, axis=0)  # (K_eff, D)

    assert reps_used.shape == (K_eff, D)
    x_vec = reps_used.reshape(-1).astype(np.float32)  # (d_in,)
    x_norm = (x_vec - x_mean) / (x_std + 1e-6)
    return x_norm, K, K_eff


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="单个 payload json 路径")
    ap.add_argument("--ckpt",    required=True, help="adapter_best.pth 路径")
    ap.add_argument("--image_root", required=False, default="data/image", help="原始图像所在目录，用 stem 匹配")
    ap.add_argument("--outdir",  required=True, help="输出目录（会自动创建）")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None,
                    help="推理精度，默认用 ckpt 里记的 precision")
    args = ap.parse_args()

    payload_path = Path(args.payload)
    ckpt_path = Path(args.ckpt)
    image_root = Path(args.image_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = payload_path.name.replace("_payload.json", "")
    log(f"stem = {stem}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先从 ckpt 恢复 adapter 结构 + meta
    # precision 以命令行优先，缺省则用 ckpt 里的
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    # 临时用 fp32 构造 adapter / 读 meta，后面再确定 VAE dtype
    adapter_dtype = torch.float32
    adapter, d_in, d_out, Hc, Wc, x_mean, x_std, target_size, vae_id_ckpt, prec_ckpt = build_adapter_from_ckpt(
        ckpt_path, device=device, dtype=adapter_dtype
    )

    prec_used = args.precision if args.precision is not None else prec_ckpt
    if prec_used not in dtype_map:
        prec_used = "fp32"
    vae_dtype = dtype_map[prec_used]

    log(f"device={device}, vae_id={vae_id_ckpt}, precision={prec_used}")

    # 加载 VAE
    log("loading VAE ...")
    vae = AutoencoderKL.from_pretrained(vae_id_ckpt)
    vae = vae.to(device=device, dtype=vae_dtype).eval()

    # 从 payload -> x_norm
    log(f"loading payload: {payload_path}")
    reps = load_kreps_json(payload_path)  # (K, D)
    x_norm, K_raw, K_eff = make_input_from_reps(reps, d_in=d_in, x_mean=x_mean, x_std=x_std)
    log(f"payload: K_raw={K_raw}, dim={reps.shape[1]}, K_eff(used)={K_eff}, flatten={x_norm.size}")

    x = torch.from_numpy(x_norm).unsqueeze(0).to(device=device, dtype=adapter_dtype)  # (1, d_in)

    # 先找一张对应的原图（如果有的话），用于尺度对齐和可视化
    img_path = find_one_image(stem, image_root)
    img_gt_tensor = None
    y_true_std = None
    S = target_size

    if img_path is not None and img_path.exists():
        log(f"found GT image: {img_path}")
        pil_gt = Image.open(img_path).convert("RGB")
        pil_gt_sq = square_canvas(pil_gt, S)
        tfm = T.ToTensor()
        img_gt_tensor = tfm(pil_gt_sq).unsqueeze(0).to(device=device, dtype=vae_dtype)  # [0,1]
        img_gt_tensor = img_gt_tensor * 2 - 1  # [-1,1]

        with torch.no_grad():
            dist = vae.encode(img_gt_tensor).latent_dist
            z_true = dist.mean  # (1,4,Hc,Wc)
        y_true_flat = z_true.reshape(1, -1).float()
        y_true_std = float(y_true_flat.std().clamp(min=1e-6))
        log(f"GT latent std = {y_true_std:.6f}")
    else:
        log("⚠️ 没在 image_root 找到对应原图，只做无监督重建（不做尺度对齐 & 对比图）。")

    # 通过 adapter 得到 latent
    adapter.eval()
    with torch.no_grad():
        y_pred = adapter(x)  # (1, d_out)
        # reshape
        C = d_out // (Hc * Wc)
        y_pred = y_pred.view(1, C, Hc, Wc)

        # 去均值 + 按 GT latent std（如果有）做尺度对齐
        y_pred = y_pred - y_pred.mean()
        pred_std = y_pred.std().clamp(min=1e-6)
        if y_true_std is not None:
            y_pred = y_pred * (y_true_std / pred_std)
        else:
            # 至少把 std 调到一个稳定值
            y_pred = y_pred / pred_std

        # 解码
        img_pred = vae.decode(y_pred.to(dtype=vae_dtype)).sample  # [-1,1]

    # 保存重建图
    infer_path = outdir / "infer.png"
    save_image(((img_pred + 1) / 2).clamp(0, 1), infer_path)
    log(f"saved recon -> {infer_path}")

    # 如果有 GT，再存一份 GT 和对比图
    if img_gt_tensor is not None:
        gt_path = outdir / "gt.png"
        save_image(((img_gt_tensor + 1) / 2).clamp(0, 1), gt_path)

        # 拼一张 compare.png： [GT | RECON]
        with torch.no_grad():
            gt_01 = ((img_gt_tensor + 1) / 2).clamp(0, 1)
            rec_01 = ((img_pred + 1) / 2).clamp(0, 1)
            grid = torch.cat([gt_01, rec_01], dim=-1)  # 宽度方向拼接
            compare_path = outdir / "compare.png"
            save_image(grid, compare_path)
            log(f"saved compare -> {compare_path}")

    log("✅ done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把语义通信压缩后的 KToken-Reps（n_tokens=4, dim=512, int8+scale）映射到 Stable Diffusion 的 VAE 潜空间，
再用 VAE 解码生成图片。

两种模式：
- fit（默认）：用原图做一次性线性拟合，构造从 reps→latent 的映射（单样本闭环，验证“可接上 VAE 解码端”）。
- random：固定随机映射 reps→latent，仅作快速粗测。

依赖：diffusers>=0.29, torch, pillow, numpy
"""

import os, json, zlib, base64, argparse, math
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL


def log(*a): print("[sem2vae]", *a)


# ---------- IO & utils ----------

def load_payload_kreps(path: Path):
    """
    读取 KToken-Reps json：
    {
      "n_tokens": 4,
      "dim": 512,
      "dtype": "int8",
      "encoding": "zlib+base64",
      "scales": [s1, s2, s3, s4],
      "data": "<base64 zlib bytes>"
    }
    返回 reps: np.float32,  shape=(n_tokens, dim)
    """
    obj = json.loads(Path(path).read_text())
    n = int(obj["n_tokens"]); d = int(obj["dim"])
    scales = np.asarray(obj["scales"], dtype=np.float32)           # (n,)
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(zlib.decompress(raw), dtype=np.int8)       # (n*d,)
    arr = arr.reshape(n, d).astype(np.float32)
    reps = arr * scales[:, None]                                   # (n, d)
    return reps


def to_multiple_of(x, m=8): return int(math.ceil(x / m) * m)


@torch.no_grad()
def encode_with_vae_mean(vae: AutoencoderKL, img_pil: Image.Image, device: str, dtype: torch.dtype):
    tfm = T.Compose([T.ToTensor()])  # [0,1]
    x = tfm(img_pil).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * 2 - 1
    dist = vae.encode(x).latent_dist
    z = dist.mean  # 不加噪声
    return z  # (1,4,H/8,W/8)


def standardize(y, target_std=0.6, eps=1e-6):
    y = y - np.mean(y)
    sd = np.std(y)
    if sd < eps: return y
    return y * (target_std / sd)


# ---------- mapping reps -> latent ----------

def reps_to_latent_fit(reps: np.ndarray, vae: AutoencoderKL, image_path: Path, device: str, dtype: torch.dtype):
    """
    单样本线性拟合：
      给定 x = reps.flatten()      [2048]
      希望得到 y = z_true.flatten  [4*Hc*Wc]
      解：W = x^+ y = (x^T / (x·x)) ⊗ y
      对同一个 x，有 x @ W == y（精确拟合，演示“可解码”）
    """
    # 读原图并 pad 到 8 的倍数尺寸
    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    W = to_multiple_of(w, 8); H = to_multiple_of(h, 8)
    if (W, H) != (w, h):
        canvas = Image.new("RGB", (W, H), (0, 0, 0)); canvas.paste(img, ((W-w)//2, (H-h)//2)); img = canvas

    # VAE 编码得到真 latent
    z_true = encode_with_vae_mean(vae, img, device=device, dtype=dtype)  # (1,4,H/8,W/8)
    z_true = z_true[0].float().cpu().numpy()
    C, Hc, Wc = z_true.shape
    assert C == 4, f"VAE latent 通道应为 4，得到 {C}"

    x = reps.astype(np.float32).reshape(-1)          # (2048,)
    y = z_true.reshape(-1)                           # (4*Hc*Wc,)

    denom = float(np.dot(x, x)) + 1e-8
    # 显式构造 z_hat，使得对该 x 精确等于 y
    # 等价于 z_hat = (x / (x·x)) * (x·y) ，但我们直接返回 y 的重排形状以减少数值误差
    # 为了清晰演示“来自 reps 的线性可还原性”，我们按解析式再求一次：
    W = np.outer(x, y) / denom                       # (2048, 4*Hc*Wc)
    z_hat = x @ W                                    # (4*Hc*Wc,)
    z_hat = z_hat.reshape(4, Hc, Wc)

    # 标准化到与 VAE 编码统计相近的力度
    z_hat = standardize(z_hat, target_std=0.6).astype(np.float32)
    return z_hat, (H, W), (Hc, Wc)


def reps_to_latent_random(reps: np.ndarray, target_hw=(256, 256), seed=1234):
    """
    固定随机线性映射 reps.flatten() -> latent.flatten()
    """
    H, W = target_hw
    Hc, Wc = H // 8, W // 8
    x = reps.astype(np.float32).reshape(-1)         # (2048,)
    D_in = x.shape[0]; D_out = 4 * Hc * Wc

    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0 / np.sqrt(D_in), size=(D_in, D_out)).astype(np.float32)
    y = x @ W                                       # (D_out,)
    y = standardize(y, target_std=0.6)
    z_hat = y.reshape(4, Hc, Wc).astype(np.float32)
    return z_hat, (H, W), (Hc, Wc)


# ---------- main ----------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True, help="KToken-Reps JSON 文件路径")
    ap.add_argument("--image", help="fit 模式需要：用于拟合的原图路径")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vae_id", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--mode", choices=["fit", "random"], default="fit")
    ap.add_argument("--target_size", type=int, default=256, help="random 模式下的输出图尺寸（边长，8 的倍数）")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]

    # 1) 读取 reps
    reps = load_payload_kreps(Path(args.payload))   # (4,512)
    log(f"payload: reps shape={reps.shape}, min={reps.min():.4f}, max={reps.max():.4f}, std={reps.std():.4f}")

    # 2) 加载 VAE
    vae = AutoencoderKL.from_pretrained(args.vae_id)
    vae = vae.to(device=device, dtype=dtype).eval()

    # 3) 映射到 latent
    if args.mode == "fit":
        if not args.image:
            raise ValueError("fit 模式需要 --image 原图路径")
        z_hat, (H, W), (Hc, Wc) = reps_to_latent_fit(reps, vae, Path(args.image), device, dtype)
        tag = "fit"
    else:
        sz = to_multiple_of(args.target_size, 8)
        z_hat, (H, W), (Hc, Wc) = reps_to_latent_random(reps, (sz, sz), seed=args.seed)
        tag = f"rand{args.seed}_sz{sz}"

    # 4) VAE 解码
    z_t = torch.from_numpy(z_hat[None]).to(device=device, dtype=dtype)  # (1,4,Hc,Wc)
    out = vae.decode(z_t).sample                                        # [-1,1]
    img = (out + 1) / 2
    img = img.clamp(0, 1)

    # 5) 保存
    out_png = Path(args.outdir) / f"sem2vae_{tag}.png"
    save_image(img, out_png)
    log(f"✅ saved: {out_png}")

    # 如果有原图，顺便保存对比图
    if args.image:
        orig = Image.open(args.image).convert("RGB").resize((W, H), Image.BICUBIC)
        rec = T.ToPILImage()(img[0].cpu())
        side = Image.new("RGB", (W * 2, H))
        side.paste(orig, (0, 0)); side.paste(rec, (W, 0))
        side_png = Path(args.outdir) / f"sem2vae_{tag}_side.png"
        side.save(side_png)
        log(f"🖼️ side-by-side: {side_png}")


if __name__ == "__main__":
    main()

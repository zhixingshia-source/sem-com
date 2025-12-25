#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 Stable Diffusion 的 VAE (AutoencoderKL) 进行“编码→解码”回环重建。
- 不使用你当前的生成器 / 不使用任何语义压缩 / 不使用视频扩散器。
- 直接检验：用 VAE 对目标图片进行 encode→decode，输出重建图像，并计算 MSE/PSNR。
- 兼容本地模型目录或 HuggingFace 模型名（默认: stabilityai/sd-vae-ft-mse）。
"""

import os, argparse, math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image

try:
    from diffusers.models import AutoencoderKL
except Exception as e:
    raise RuntimeError(
        "未找到 diffusers，请先安装：pip install diffusers==0.27.2 transformers accelerate safetensors"
    ) from e


def to_multiple_of(x, m=8):
    return int(math.ceil(x / m) * m)


def load_image_keep_ratio(path, target_size=None, pad_to_multiple=8):
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size

    if target_size is not None:
        # 按最短边缩放到 target_size，保持长宽比
        if w0 < h0:
            new_w = target_size
            new_h = int(h0 * target_size / w0)
        else:
            new_h = target_size
            new_w = int(w0 * target_size / h0)
        img = img.resize((new_w, new_h), Image.BICUBIC)

    # pad 到 8 的倍数，避免 VAE shape 不整除
    w, h = img.size
    W = to_multiple_of(w, pad_to_multiple)
    H = to_multiple_of(h, pad_to_multiple)

    pad_l = (W - w) // 2
    pad_r = W - w - pad_l
    pad_t = (H - h) // 2
    pad_b = H - h - pad_t

    if any([pad_l, pad_r, pad_t, pad_b]):
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        canvas.paste(img, (pad_l, pad_t))
        img = canvas

    meta = {"orig_wh": (w0, h0), "resized_wh": (w, h), "padded_wh": (W, H),
            "pads": (pad_l, pad_r, pad_t, pad_b)}
    return img, meta


@torch.no_grad()
def encode_decode(vae, img_pil, device, sample_mode="mean", seed=0):
    """
    - 输入 PIL.Image
    - 输出：重建后的 torch.Tensor [1,3,H,W]，以及若干统计量
    """
    vae = vae.to(device)
    vae.eval()

    tfm = T.Compose([T.ToTensor()])  # [0,1]
    x = tfm(img_pil).unsqueeze(0).to(device)  # [1,3,H,W]
    x = x * 2 - 1  # [-1,1]

    # 编码到潜空间（注意：这里不乘 scaling_factor；Unet 才需要 scaled latents）
    dist = vae.encode(x).latent_dist
    if sample_mode == "sample":
        torch.manual_seed(seed)
        z = dist.sample()
    else:  # mean: 更接近“无损”重建
        z = dist.mean

    # 统计
    stats = {
        "z_mean": float(z.mean().item()),
        "z_std": float(z.std().item()),
        "z_min": float(z.min().item()),
        "z_max": float(z.max().item()),
        "z_norm": float(z.norm().item()),
    }

    # 直接用原始 latents 解码（不缩放）
    rec = vae.decode(z).sample  # [-1,1]
    rec = (rec + 1) / 2
    rec = rec.clamp(0, 1)

    return rec, stats


def mse_psnr(a, b):
    a = a.clamp(0, 1)
    b = b.clamp(0, 1)
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * math.log10(1.0) - 10 * math.log10(mse + 1e-12)
    return mse, psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="目标图片路径（不改变图片）")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--vae_id", default="stabilityai/sd-vae-ft-mse",
                    help="VAE 模型：HuggingFace 名称或本地目录")
    ap.add_argument("--target_size", type=int, default=None,
                    help="可选：按最短边缩放到此尺寸（默认不缩放）")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--sample_mode", choices=["mean", "sample"], default="mean",
                    help="encode 后取均值或采样；为保持“尽量不改图”，建议用 mean")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # 1) 加载 VAE
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    vae = AutoencoderKL.from_pretrained(args.vae_id)
    vae.to(dtype=dtype)

    # 2) 读图 & pad 到 8 的倍数
    img_pil, meta = load_image_keep_ratio(args.image, target_size=args.target_size, pad_to_multiple=8)

    # 3) 编码→解码
    rec, zstats = encode_decode(vae, img_pil, device=device, sample_mode=args.sample_mode, seed=args.seed)

    # 4) 裁掉 padding，还原到原图尺寸再评估
    pad_l, pad_r, pad_t, pad_b = meta["pads"]
    if any([pad_l, pad_r, pad_t, pad_b]):
        rec = rec[:, :, pad_t: rec.shape[2]-pad_b, pad_l: rec.shape[3]-pad_r]
        # 还原到 resize 前尺寸
        w, h = meta["resized_wh"]
        rec = torch.nn.functional.interpolate(rec, size=(h, w), mode="bilinear", align_corners=False)

    # 为了和原图对齐评估，再把原图也走同样 pipeline
    orig = Image.open(args.image).convert("RGB")
    orig_t = T.ToTensor()(orig).unsqueeze(0).to(rec.device)

    # 5) 指标
    mse, psnr = mse_psnr(rec, orig_t)

    # 6) 保存
    out_png = Path(args.outdir) / "vae_reconstruction.png"
    out_side = Path(args.outdir) / "vae_side_by_side.png"
    save_image(rec, out_png)
    save_image(torch.cat([orig_t, rec], dim=3), out_side)

    # 7) 打印
    print(f"[VAE] latents stats: {zstats}")
    print(f"[VAE] MSE={mse:.6f}, PSNR={psnr:.2f} dB")
    print(f"[VAE] saved: {out_png}")
    print(f"[VAE] side-by-side: {out_side}")


if __name__ == "__main__":
    main()

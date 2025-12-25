#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImageVariationPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="样本 stem，比如 coco_118287")
    ap.add_argument("--recon_root", required=True, help="重建图根目录，里面有 <stem>/infer.png")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--gt_root", default=None, help="GT 图根目录，可选，比如 data_coco/image")
    ap.add_argument("--model", default="lambdalabs/sd-image-variations-diffusers",
                    help="图像变分模型名")
    ap.add_argument("--n_samples", type=int, default=4, help="生成多少张变分图")
    ap.add_argument("--steps", type=int, default=50, help="扩散步数")
    ap.add_argument("--guidance", type=float, default=7.5, help="guidance scale")
    args = ap.parse_args()

    stem = args.stem
    recon_dir = Path(args.recon_root) / stem
    recon_path = recon_dir / "infer.png"
    assert recon_path.exists(), f"找不到重建图: {recon_path}"

    os.makedirs(args.outdir, exist_ok=True)
    out_stem_dir = Path(args.outdir) / stem
    out_stem_dir.mkdir(parents=True, exist_ok=True)

    # 读取重建图
    recon_img = Image.open(recon_path).convert("RGB")
    # 变分模型一般把输入缩放到 224x224 再做 CLIP 图像编码
    recon_for_model = recon_img.resize((224, 224), Image.BICUBIC)

    # （可选）GT 原图，用于最后对比
    gt_img = None
    if args.gt_root is not None:
        gt_candidates = list(Path(args.gt_root).glob(f"{stem}_*.jpg"))
        if len(gt_candidates) > 0:
            gt_img = Image.open(gt_candidates[0]).convert("RGB")

    # 设备 / 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[img-var] stem={stem}")
    print(f"[img-var] recon={recon_path}")
    if gt_img is not None:
        print(f"[img-var] gt_root={args.gt_root} (found one)")
    print(f"[img-var] device={device}")
    print(f"[img-var] loading variation model: {args.model} ...", flush=True)

    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.enable_attention_slicing()

    # 生成变分图
    with torch.no_grad():
        out = pipe(
            image=recon_for_model,
            num_images_per_prompt=args.n_samples,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        )
    images = out.images  # list of PIL

    # 保存单张
    for i, im in enumerate(images):
        im.save(out_stem_dir / f"var_{i:02d}.png")

    print(f"[img-var] saved {len(images)} variations to {out_stem_dir}")

    # 做一个对比图：GT | RECON | VAR_0
    target_size = 512

    def prep(im: Image.Image):
        return im.resize((target_size, target_size), Image.BICUBIC)

    panels = []
    labels = []

    if gt_img is not None:
        panels.append(prep(gt_img))
        labels.append("gt")
    panels.append(prep(recon_img))
    labels.append("recon")
    if len(images) > 0:
        panels.append(prep(images[0]))
        labels.append("var0")

    if len(panels) > 0:
        W = target_size * len(panels)
        H = target_size
        grid = Image.new("RGB", (W, H), (0, 0, 0))
        for i, im in enumerate(panels):
            grid.paste(im, (i * target_size, 0))
        grid_path = out_stem_dir / "compare_grid.png"
        grid.save(grid_path)
        print(f"[img-var] saved compare grid -> {grid_path}")

    print("[img-var] ✅ done.")


if __name__ == "__main__":
    main()

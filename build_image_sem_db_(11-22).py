#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_image_sem_db_11_22.py

功能：
- 扫描 image_root 下所有图像（默认 *.jpg）
- 约定命名：coco_000001_0001.jpg → stem = "coco_000001"
- 用指定的 CLIP 图像编码器计算 embedding
- 多帧（同一个 stem 对应多张图）则对 embedding 取平均
- 保存为一个 .pt 文件，里面包含：
    {
        "stems": [stem1, stem2, ...],          # list[str]
        "embeds": tensor(N, D),                # float32, L2-normalized
        "clip_model": "openai/clip-vit-..."    # 记录所用模型
    }

用法示例：
python build_image_sem_db_11_22.py \
  --image_root data/image \
  --out_path runs/sem_db/image_sem_db_11_22.pt \
  --clip_model openai/clip-vit-large-patch14 \
  --batch_size 64
"""

import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import CLIPModel, CLIPImageProcessor


def log(*a):
    print("[build-sem-db-11-22]", *a, flush=True)


def collect_image_paths(image_root: Path, pattern: str = "*.jpg"):
    """
    扫描 image_root 下所有图像，构造：
        stem -> [image_paths]
    约定：coco_000001_0001.jpg → stem = "coco_000001"
         x_y_z.jpg             → stem = "x_y"
         没有 '_' 的文件       → stem = 完整文件名（不含扩展名）
    """
    stem_to_paths = defaultdict(list)
    files = sorted(image_root.glob(pattern))
    for p in files:
        name = p.stem  # 不含扩展名
        parts = name.split("_")
        if len(parts) >= 2:
            stem = "_".join(parts[:-1])
        else:
            stem = name
        stem_to_paths[stem].append(p)

    return stem_to_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)

    ap.add_argument("--clip_model", type=str,
                    default="openai/clip-vit-large-patch14",
                    help="HuggingFace CLIP 模型名（必须和训练 kreps2clip 时一致）")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cpu，默认自动检测")
    ap.add_argument("--pattern", type=str, default="*.jpg")
    args = ap.parse_args()

    image_root = Path(args.image_root).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"image_root = {image_root}")
    log(f"out_path   = {out_path}")
    log(f"clip_model = {args.clip_model}")
    log(f"device     = {device}")

    # 1) 收集所有图像路径（按 stem 分组）
    stem_to_paths = collect_image_paths(image_root, pattern=args.pattern)
    stems = sorted(stem_to_paths.keys())
    log(f"found stems = {len(stems)}")

    if len(stems) == 0:
        raise RuntimeError(f"no images found under {image_root} with pattern {args.pattern}")

    log("example stems:", stems[:5])

    # 2) 加载 CLIP 模型
    log("loading CLIP model ...")
    clip_model = CLIPModel.from_pretrained(args.clip_model)
    clip_model.to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    processor = CLIPImageProcessor.from_pretrained(args.clip_model)

    # 3) 对每个 stem 计算语义向量：多张图则平均后再 L2 norm
    all_stems = []
    all_embeds = []

    for stem in tqdm(stems, desc="stems"):
        paths = stem_to_paths[stem]
        # 分批处理，避免一次性把所有帧读进来
        feats = []

        for i in range(0, len(paths), args.batch_size):
            batch_paths = paths[i: i + args.batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(img)

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)  # (B, D)
                feat = feat.to(torch.float32)
            feats.append(feat.cpu())

        feats = torch.cat(feats, dim=0)  # (n_frames, D)
        # 平均后再 normalize
        mean_feat = feats.mean(dim=0, keepdim=True)  # (1, D)
        mean_feat = F.normalize(mean_feat, dim=-1)   # (1, D)

        all_stems.append(stem)
        all_embeds.append(mean_feat.squeeze(0))      # (D,)

    embeds = torch.stack(all_embeds, dim=0)  # (N, D)
    log(f"final db: N={embeds.shape[0]}, D={embeds.shape[1]}")

    obj = {
        "stems": all_stems,
        "embeds": embeds,
        "clip_model": args.clip_model,
    }
    torch.save(obj, out_path)
    log(f"saved semantic db -> {out_path}")


if __name__ == "__main__":
    main()

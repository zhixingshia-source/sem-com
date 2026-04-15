#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kreps2clip_train.py

多样本 Adapter 训练（KToken-Reps → CLIP 图像语义向量）

输入：
  - payload_root 里的一堆 *_payload.json（KToken-Reps: int8 + scales）
  - image_root 里的 coco_xxx_0001.jpg 等，按 stem 对应

模型：
  - AdapterMLP( D_in = max_tokens * 512, D_out = D_clip )
  - CLIP: 用预训练的 CLIP image encoder 提供语义监督（冻结不训练）

损失：
  - L_mse      : 预测 embedding 与 CLIP image embedding 的 MSE
  - L_cos      : 1 - cosine_similarity(pred, gt)

把 pixel-level 重建换成 semantic-level 对齐，更符合 DeepSC 风格。
"""

import os
import json
import zlib
import base64
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

# transformers / CLIP
try:
    from transformers import CLIPModel, CLIPImageProcessor
    HAS_CLIP = True
except Exception:
    HAS_CLIP = False


# ----------------- 工具函数 -----------------
def log(*a):
    print("[kreps2clip-train]", *a, flush=True)


def load_kreps_json(p: Path) -> np.ndarray:
    """读取 KToken-Reps JSON，反量化成 float32 reps，shape=(K, D)"""
    obj = json.loads(p.read_text())
    n = int(obj["n_tokens"])
    d = int(obj["dim"])
    scales = np.asarray(obj["scales"], dtype=np.float32)  # (K,)
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(zlib.decompress(raw), dtype=np.int8)  # (K*D,)
    arr = arr.reshape(n, d).astype(np.float32)  # (K, D)
    reps = arr * scales[:, None]
    return reps  # (K, D)


def square_canvas(pil: Image.Image, S: int) -> Image.Image:
    """等比缩放到不超过 S×S，再居中贴到 S×S（黑边）"""
    w0, h0 = pil.size
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"invalid image size: {pil.size}")
    r = min(S / max(1, w0), S / max(1, h0))
    nw, nh = max(1, int(round(w0 * r))), max(1, int(round(h0 * r)))
    img = pil.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (S, S), (0, 0, 0))
    canvas.paste(img, ((S - nw) // 2, (S - nh) // 2))
    return canvas


# ----------------- Adapter -----------------
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 固定使用 FP32 计算，避免 Half/Float 冲突
        return self.net(x.float()).float()


# ----------------- Dataset -----------------
class KRepsDataset(torch.utils.data.Dataset):
    """
    每个样本： (x_flat, img_tensor, stem, img_path)
    - x_flat: float32, shape = (max_tokens * D,)  经过截断/0-padding
    - img_tensor: float32, [0,1], shape=(3,S,S)
    """

    def __init__(self, payload_root: Path, image_root: Path,
                 target_size: int, max_tokens: int, stems_file: str = None):
        super().__init__()
        self.payload_root = payload_root
        self.image_root = image_root
        self.S = target_size
        self.max_tokens = max_tokens

        payload_paths = sorted(payload_root.glob("*_payload.json"))
        if stems_file:
            keep = {
                line.strip()
                for line in Path(stems_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            before = len(payload_paths)
            payload_paths = [
                p for p in payload_paths
                if p.name.replace("_payload.json", "") in keep
            ]
            log(f"stems_file={stems_file} keep={len(keep)} payloads={len(payload_paths)}/{before}")
        if not payload_paths:
            raise RuntimeError(f"no *_payload.json in {payload_root}")

        # image index: stem -> [image paths]
        idx = defaultdict(list)
        img_files = list(image_root.glob("*.jpg"))
        for p in img_files:
            parts = p.stem.split("_")
            if len(parts) >= 2:
                # coco_118287_0001 -> coco_118287
                stem = "_".join(parts[:-1])
            else:
                stem = p.stem
            idx[stem].append(p)
        log(f"image index built: {len(idx)} keys from {image_root} (raw files={len(img_files)})")

        # pair payload & image
        samples = []
        no_img = 0
        for p in payload_paths:
            name = p.name
            if not name.endswith("_payload.json"):
                continue
            stem = name[:-len("_payload.json")]
            cand = idx.get(stem, [])
            if not cand:
                no_img += 1
                continue
            cand_sorted = sorted(cand)
            samples.append((p, cand_sorted[0], stem))
        log(f"paired samples: {len(samples)}  (payloads without image: {no_img})")
        if not samples:
            raise RuntimeError("no paired payload/image samples")

        self.samples = samples
        self.tfm = T.ToTensor()  # PIL -> [0,1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        p_payload, p_img, stem = self.samples[i]

        # reps
        reps = load_kreps_json(p_payload)  # (K,D)
        K, D = reps.shape
        if self.max_tokens is None or self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0 for fixed D_in")

        if K > self.max_tokens:
            # 均匀采样到 max_tokens
            idxs = np.linspace(0, K - 1, self.max_tokens, dtype=int)
            reps = reps[idxs]
        elif K < self.max_tokens:
            pad = np.zeros((self.max_tokens - K, D), dtype=np.float32)
            reps = np.concatenate([reps, pad], axis=0)

        x_flat = reps.reshape(-1).astype(np.float32)  # (max_tokens * D,)

        # image
        img0 = Image.open(p_img).convert("RGB")
        imgS = square_canvas(img0, self.S)
        img_tensor = self.tfm(imgS)  # [0,1], (3,S,S)

        return torch.from_numpy(x_flat), img_tensor, stem, str(p_img)


# ----------------- 主逻辑 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload_root", type=str, required=True)
    ap.add_argument("--stems_file", type=str, default=None,
                    help="optional txt file with one stem per line for train split")
    ap.add_argument("--image_root",   type=str, required=True)
    ap.add_argument("--outdir",       type=str, required=True)

    ap.add_argument("--clip_model",   type=str,
                    default="openai/clip-vit-large-patch14",
                    help="HuggingFace CLIP 模型名")
    ap.add_argument("--target_size",  type=int, default=256,
                    help="把原图等比缩放到 <=S×S 再方贴（和 Stage2 一致即可）")
    ap.add_argument("--max_tokens",   type=int, default=32,
                    help="payload 中最多使用多少个 token（不足则 0-pad）")

    ap.add_argument("--steps",        type=int, default=50000)
    ap.add_argument("--batch_size",   type=int, default=8)
    ap.add_argument("--hidden",       type=int, default=8192)
    ap.add_argument("--layers",       type=int, default=2)
    ap.add_argument("--dropout",      type=float, default=0.0)
    ap.add_argument("--lr",           type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed",         type=int, default=1234)

    ap.add_argument("--log_interval",  type=int, default=100)
    ap.add_argument("--save_interval", type=int, default=2000)

    # loss 权重
    ap.add_argument("--lambda_mse",    type=float, default=1.0)
    ap.add_argument("--lambda_cos",    type=float, default=0.5)

    args = ap.parse_args()

    if not HAS_CLIP:
        raise RuntimeError("transformers/CLIP 未安装，先 `pip install transformers` 再试")

    payload_root = Path(args.payload_root)
    image_root   = Path(args.image_root)
    outdir       = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}, batch_size={args.batch_size}")
    log(f"payload_root={payload_root}")
    log(f"image_root={image_root}")
    log(f"outdir={outdir}")

    # dataset & loader
    S = args.target_size
    ds = KRepsDataset(payload_root, image_root, S, args.max_tokens, args.stems_file)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # CLIP (frozen)
    log(f"loading CLIP model: {args.clip_model}")
    clip_model = CLIPModel.from_pretrained(args.clip_model)
    clip_model.to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    clip_proc = CLIPImageProcessor.from_pretrained(args.clip_model)
    to_pil = T.ToPILImage()

    # 用第一个样本确定 D_in / D_clip
    sample_x, sample_img, _, _ = ds[0]
    D_in = sample_x.numel()

    with torch.no_grad():
        pil = to_pil(sample_img)
        inputs = clip_proc(images=[pil], return_tensors="pt").to(device)
        feat = clip_model.get_image_features(**inputs)  # (1, D_clip)
        D_clip = feat.shape[-1]

    log(f"D_in={D_in} (max_tokens={args.max_tokens}, dim=512)")
    log(f"D_clip={D_clip} (来自 {args.clip_model})")

    # Adapter
    adapter = AdapterMLP(
        d_in=D_in,
        d_out=D_clip,
        hidden=args.hidden,
        n_layers=args.layers,
        p_drop=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    # 运行时维护输入标准化的全局统计（scalar mean/std）
    running_mean = 0.0
    running_m2 = 0.0
    global_n = 0

    best = float("inf")
    ckpt_best = outdir / "adapter_clip_best.pth"

    step = 0
    while step < args.steps:
        for x_flat, img_01, stems, img_paths in loader:
            step += 1
            if step > args.steps:
                break

            B = x_flat.size(0)

            # ==== x_flat 标准化统计（在 CPU float32 上）====
            flat_np = (
                x_flat.detach()
                .to("cpu", torch.float32)
                .reshape(-1)
                .numpy()
            )
            batch_n = flat_np.size
            if batch_n > 0:
                batch_mean = float(flat_np.mean())
                batch_var = float(flat_np.var())
            else:
                batch_mean = 0.0
                batch_var = 0.0

            if global_n == 0:
                running_mean = batch_mean
                running_m2 = batch_var * batch_n
                global_n = batch_n
            else:
                delta = batch_mean - running_mean
                new_n = global_n + batch_n
                running_mean = running_mean + delta * batch_n / new_n
                running_m2 = (
                    running_m2
                    + batch_var * batch_n
                    + delta * delta * global_n * batch_n / new_n
                )
                global_n = new_n

            running_var = running_m2 / max(global_n, 1)
            running_std = math.sqrt(running_var + 1e-6)

            # ==== 构造 x_norm ====
            x_flat = x_flat.to(device=device, dtype=torch.float32)  # (B,D_in)
            x_norm = (x_flat - running_mean) / running_std  # scalar broadcast

            # ==== CLIP image embedding (GT) ====
            # img_01: [0,1], (B,3,S,S) -> list of PIL
            pil_list = [to_pil(im) for im in img_01]
            with torch.no_grad():
                inputs = clip_proc(images=pil_list, return_tensors="pt").to(device)
                feat_gt = clip_model.get_image_features(**inputs)  # (B, D_clip)
                feat_gt = F.normalize(feat_gt.float(), dim=-1)

            # ==== Adapter forward ====
            opt.zero_grad(set_to_none=True)
            y_pred = adapter(x_norm)                    # (B, D_clip)
            y_pred = F.normalize(y_pred.float(), dim=-1)

            # ==== 损失 ====
            loss_mse = F.mse_loss(y_pred, feat_gt)
            loss_cos = 1.0 - (y_pred * feat_gt).sum(dim=-1).mean()
            loss = args.lambda_mse * loss_mse + args.lambda_cos * loss_cos

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()

            # ==== 日志 ====
            if step % args.log_interval == 0 or step == 1:
                log(
                    f"step {step}/{args.steps} | "
                    f"Lmse={float(loss_mse):.4f} "
                    f"Lcos={float(loss_cos):.44f} "
                    f"loss={float(loss):.4f} | "
                    f"x_mean={running_mean:.4f} x_std={running_std:.4f} "
                    f"global_n={global_n}"
                )

            # ==== 定期保存 ckpt ====
            if step % args.save_interval == 0:
                ckpt_path = outdir / f"adapter_clip_step_{step:06d}.pth"
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_clip,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "max_tokens": int(args.max_tokens),
                        "clip_model": args.clip_model,
                    },
                    ckpt_path,
                )
                log(f"  💾 saved ckpt -> {ckpt_path}")

            # best ckpt（按总 loss）
            if float(loss) < best:
                best = float(loss)
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_clip,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "max_tokens": int(args.max_tokens),
                        "clip_model": args.clip_model,
                    },
                    ckpt_best,
                )
                log(f"  ⭐ new best loss={best:.6f} -> {ckpt_best}")

        # end for loader

    log(f"✅ done. best={best:.6f} -> {ckpt_best}")


if __name__ == "__main__":
    main()

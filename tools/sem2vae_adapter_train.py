#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多样本 Adapter 训练（KToken-Reps → StableDiffusion VAE latent）

- 输入：payload_root 里的一堆 *_payload.json（KToken-Reps: int8 + scales）
- 配对：image_root 里的 coco_xxx_0001.jpg 等，按 stem 对应
- 模型：AdapterMLP( D_in = max_tokens * 512, D_out = 4 * Hc * Wc )
- VAE：stabilityai/sd-vae-ft-mse
- 损失：
    L_img      : L1 图像重建
    L_lat      : latent MSE
    L_cos_lat  : latent cosine (1 - cos)
    L_clip_ii  : CLIP image-image 语义损失（可关）
"""

import os, json, zlib, base64, math, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL

# CLIP 语义模型（可选）
try:
    from transformers import CLIPModel
    HAS_CLIP = True
except Exception:
    HAS_CLIP = False


# ----------------- 工具函数 -----------------
def log(*a):
    print("[sem2vae-train]", *a, flush=True)


def to_m8(x: int) -> int:
    """round up to multiple of 8"""
    return int(math.ceil(x / 8) * 8)


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
    return reps


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


def save_side(gt, rec, path_png: Path):
    """保存并排图像（[-1,1] 张量）"""
    with torch.no_grad():
        gt_01 = ((gt.clamp(-1, 1) + 1) / 2).cpu()
        rec_01 = ((rec.clamp(-1, 1) + 1) / 2).cpu()
        grid = torch.cat([gt_01, rec_01], dim=-1)  # (B,C,H,2W)
        save_image(grid, path_png)


# ----------------- Adapter -----------------
class AdapterMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=8192, n_layers=2, p_drop=0.0):
        super().__init__()
        layers, last = [], d_in
        for _ in range(n_layers-1):
            layers += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(p_drop)]
            last = hidden
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 🔥 关键: adapter 内固定使用 FP32 计算，解决 Half/Float 冲突
        return self.net(x.float()).float()



# ----------------- Dataset -----------------
class KRepsDataset(torch.utils.data.Dataset):
    """
    每个样本： (x_flat, img_tensor, stem, img_path)
    - x_flat: float32, shape = (max_tokens * D,)  经过截断/0-padding
    - img_tensor: float32, [0,1], shape=(3,S,S)
    """

    def __init__(self, payload_root: Path, image_root: Path,
                 target_size: int, max_tokens: int):
        super().__init__()
        self.payload_root = payload_root
        self.image_root = image_root
        self.S = target_size
        self.max_tokens = max_tokens

        payload_paths = sorted(payload_root.glob("*_payload.json"))
        if not payload_paths:
            raise RuntimeError(f"no *_payload.json in {payload_root}")

        # image index: stem -> [image paths]
        idx = defaultdict(list)
        img_files = list(image_root.glob("*.jpg"))
        for p in img_files:
            parts = p.stem.split("_")
            if len(parts) >= 2:
                stem = "_".join(parts[:-1])  # coco_118287_0001 -> coco_118287
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p_payload, p_img, stem = self.samples[i]

        # reps
        reps = load_kreps_json(p_payload)  # (K,D)
        K, D = reps.shape
        if self.max_tokens is None or self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0 for fixed D_in")

        if K > self.max_tokens:
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
    ap.add_argument("--image_root",   type=str, required=True)
    ap.add_argument("--outdir",       type=str, required=True)

    ap.add_argument("--vae_id",       type=str,
                    default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--target_size",  type=int, default=256)
    ap.add_argument("--steps",        type=int, default=50000)
    ap.add_argument("--batch_size",   type=int, default=8)
    ap.add_argument("--max_tokens",   type=int, default=32)
    ap.add_argument("--precision",    choices=["fp32", "fp16", "bf16"],
                    default="fp32")
    ap.add_argument("--hidden",       type=int, default=8192)
    ap.add_argument("--layers",       type=int, default=2)
    ap.add_argument("--lr",           type=float, default=2e-4)
    ap.add_argument("--dropout",      type=float, default=0.0)

    ap.add_argument("--log_interval",  type=int, default=100)
    ap.add_argument("--save_interval", type=int, default=2000)
    ap.add_argument("--seed",          type=int, default=1234)

    # loss 权重
    ap.add_argument("--lambda_img",      type=float, default=1.0)
    ap.add_argument("--lambda_lat",      type=float, default=0.1)
    ap.add_argument("--lambda_cos_lat",  type=float, default=0.1)
    ap.add_argument("--lambda_clip_ii",  type=float, default=0.2)

    args = ap.parse_args()

    payload_root = Path(args.payload_root)
    image_root   = Path(args.image_root)
    outdir       = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    log(f"device={device}, precision={args.precision}, batch_size={args.batch_size}")

    # dataset & loader
    S = to_m8(args.target_size)
    ds = KRepsDataset(payload_root, image_root, S, args.max_tokens)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # VAE
    log(f"loading VAE: {args.vae_id}")
    vae = AutoencoderKL.from_pretrained(args.vae_id).to(device=device, dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # 用第一个样本确定 D_in / D_out
    sample_x, sample_img, _, _ = ds[0]
    D_in = sample_x.numel()
    with torch.no_grad():
        img = sample_img.unsqueeze(0).to(device=device, dtype=dtype)  # (1,3,S,S)
        img = img * 2 - 1
        dist = vae.encode(img).latent_dist
        z = dist.mean  # (1,4,Hc,Wc)
        _, C, Hc, Wc = z.shape
        assert C == 4, f"VAE latent channels={C}, expected 4"
        D_out = C * Hc * Wc

    log(f"D_in={D_in} (max_tokens={args.max_tokens}, dim=512), D_out={D_out} (4×{Hc}×{Wc})")

    # Adapter
    adapter = AdapterMLP(
        d_in=D_in,
        d_out=D_out,
        hidden=args.hidden,
        n_layers=args.layers,
        p_drop=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )

    # CLIP（只做 image-image 语义，相似度）
    clip_model = None
    clip_mean = None
    clip_std = None
    use_clip = HAS_CLIP and args.lambda_clip_ii > 0.0

    if use_clip:
        log("loading CLIP model: openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_model.to(device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        # 官方 CLIP 归一化常数
        clip_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        clip_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
    else:
        if args.lambda_clip_ii > 0.0 and not HAS_CLIP:
            log("!! transformers/CLIP not available, disabling CLIP loss")
        args.lambda_clip_ii = 0.0

    def clip_encode_image(x_01: torch.Tensor, requires_grad: bool) -> torch.Tensor:
        """
        x_01: [0,1] tensor, shape (B,3,H,W)
        返回归一化后的 CLIP image features, shape (B, D_clip)
        """
        if not use_clip or clip_model is None:
            raise RuntimeError("clip_encode_image called but CLIP not initialized")

        # resize 到 224×224
        x = F.interpolate(x_01, size=(224, 224),
                          mode="bilinear", align_corners=False)
        x = x.to(torch.float32)
        x = (x - clip_mean) / clip_std
        if not requires_grad:
            with torch.no_grad():
                feat = clip_model.get_image_features(pixel_values=x)
        else:
            feat = clip_model.get_image_features(pixel_values=x)
        feat = F.normalize(feat, dim=-1)
        return feat

    # 运行时维护输入标准化的全局统计（scalar mean/std）
    running_mean = 0.0
    running_m2 = 0.0
    global_n = 0

    best = float("inf")
    ckpt_best = outdir / "adapter_best.pth"

    step = 0
    while step < args.steps:
        for x_flat, img_gt_01, stems, img_paths in loader:
            step += 1
            if step > args.steps:
                break

            B = x_flat.size(0)
            x_flat = x_flat.to(device=device, dtype=dtype)  # (B,D_in)

            # ==== 更新全局 mean/std（在 CPU float32 上）====
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

            # 用当前 global mean/std 做标准化
            x_norm = (x_flat - running_mean) / running_std  # scalar broadcast

            # ==== VAE encode 图像 ====
            img_gt = img_gt_01.to(device=device, dtype=dtype)  # [0,1]
            img_gt = img_gt * 2 - 1  # [-1,1]
            with torch.no_grad():
                dist = vae.encode(img_gt).latent_dist
                z_true = dist.mean  # (B,4,Hc,Wc)
            y_true = z_true.reshape(B, -1).float()
            y_true_std = float(y_true.std().clamp(min=1e-6))

            # ==== Adapter forward ====
            opt.zero_grad(set_to_none=True)
            y_pred = adapter(x_norm.float()).reshape(B, 4, Hc, Wc)
            # 去均值 + 对齐整体 std
            y_pred = y_pred - y_pred.mean(dim=(1, 2, 3), keepdim=True)
            pred_std = y_pred.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
            y_pred = y_pred * (y_true_std / pred_std)

            # VAE decode
            img_pred = vae.decode(y_pred.to(dtype)).sample  # [-1,1]

            # ==== 损失 ====
            # 图像 L1
            loss_img = F.l1_loss(img_pred, img_gt)

            # latent MSE + cosine
            y_pred_flat = y_pred.reshape(B, -1).float()
            loss_lat = F.mse_loss(y_pred_flat, y_true)
            cos_lat = F.cosine_similarity(y_pred_flat, y_true, dim=1).mean()
            loss_cos_lat = 1.0 - cos_lat

            # CLIP image-image 语义损失
            loss_clip_ii = torch.tensor(0.0, device=device)
            if use_clip and args.lambda_clip_ii > 0.0:
                img_pred_01 = ((img_pred.clamp(-1, 1) + 1) / 2)
                img_gt_for_clip = ((img_gt.clamp(-1, 1) + 1) / 2).detach()

                feat_pred = clip_encode_image(img_pred_01, requires_grad=True)
                feat_gt = clip_encode_image(img_gt_for_clip, requires_grad=False).detach()
                loss_clip_ii = (1.0 - (feat_pred * feat_gt).sum(dim=-1)).mean()

            loss = (
                args.lambda_img * loss_img
                + args.lambda_lat * loss_lat
                + args.lambda_cos_lat * loss_cos_lat
                + args.lambda_clip_ii * loss_clip_ii
            )

            loss.backward()
            opt.step()

            # ==== 日志 + 可视化 ====
            if step % args.log_interval == 0 or step == 1:
                log(
                    f"step {step}/{args.steps} | "
                    f"Limg={float(loss_img):.4f} "
                    f"Llat={float(loss_lat):.4f} "
                    f"Lcos_lat={float(loss_cos_lat):.4f} "
                    f"Lclip_ii={float(loss_clip_ii):.4f} "
                    f"loss={float(loss):.4f} | "
                    f"global_n={global_n}"
                )
                with torch.no_grad():
                    out_png = outdir / f"train_mid_{step:05d}.png"
                    save_side(img_gt[:1], img_pred[:1], out_png)
                    log(f"  preview saved -> {out_png}")

            # ==== 定期保存 ckpt ====
            if step % args.save_interval == 0:
                ckpt_path = outdir / f"adapter_step_{step:06d}.pth"
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_out,
                        "Hc": Hc,
                        "Wc": Wc,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "target_size": S,
                        "vae_id": args.vae_id,
                        "precision": args.precision,
                    },
                    ckpt_path,
                )
                log(
                    f"  💾 saved ckpt -> {ckpt_path} "
                    f"(x_mean={running_mean:.4f}, x_std={running_std:.4f})"
                )

            # best ckpt
            if float(loss) < best:
                best = float(loss)
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_out,
                        "Hc": Hc,
                        "Wc": Wc,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "target_size": S,
                        "vae_id": args.vae_id,
                        "precision": args.precision,
                    },
                    ckpt_best,
                )
                log(f"  ⭐ new best loss={best:.6f} -> {ckpt_best}")

        # end for loader

    log(f"✅ done. best={best:.6f} -> {ckpt_best}")


if __name__ == "__main__":
    main()

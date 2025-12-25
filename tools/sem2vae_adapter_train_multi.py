#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, zlib, base64, math, argparse, glob, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

def log(*a): print("[sem2vae-train-multi]", *a, flush=True)
def to_m8(x): return int(math.ceil(x/8)*8)

def load_kreps_json(p: Path) -> np.ndarray:
    obj = json.loads(Path(p).read_text())
    n = int(obj["n_tokens"]); d = int(obj["dim"])
    scales = np.asarray(obj["scales"], dtype=np.float32)       # (K,)
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(zlib.decompress(raw), dtype=np.int8)   # (K*D,)
    arr = arr.reshape(n, d).astype(np.float32)                 # (K, D)
    reps = arr * scales[:, None]
    return reps  # (K, D)

def square_canvas(pil: Image.Image, S: int) -> Image.Image:
    w0, h0 = pil.size
    r = min(S/max(1,w0), S/max(1,h0))
    nw, nh = max(1,int(round(w0*r))), max(1,int(round(h0*r)))
    img = pil.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (S, S), (0, 0, 0))
    canvas.paste(img, ((S-nw)//2, (S-nh)//2))
    return canvas

class AdapterMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=8192, n_layers=2, p_drop=0.0):
        super().__init__()
        layers, last = [], d_in
        for _ in range(n_layers-1):
            layers += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(p_drop)]
            last = hidden
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: (B, D_in)
        return self.net(x)

class PairSet(Dataset):
    def __init__(self, pairs: List[Tuple[str,str]], target_size=256, dtype=torch.float32, vae_id="stabilityai/sd-vae-ft-mse", device="cuda"):
        self.items = [(Path(p), Path(i)) for p,i in pairs]
        self.S = to_m8(target_size)
        self.tfm = T.Compose([T.ToTensor()])
        self.device, self.dtype = device, dtype

        # 预加载 VAE（只编码，不训）
        self.vae = AutoencoderKL.from_pretrained(vae_id).to(device=device, dtype=dtype).eval()
        # 第一次遍历：统计全局 x_mean/x_std（基于 reps 展平）
        log(f"scanning {len(self.items)} samples to estimate x_mean/x_std ...")
        m, v, n = 0.0, 0.0, 0
        for jp, _ in self.items:
            reps = load_kreps_json(jp)          # (K, 512)
            x = reps.reshape(-1).astype(np.float32)
            n_new = x.size
            mean_x = float(x.mean()); var_x = float(x.var())
            if n == 0:
                m, v, n = mean_x, var_x, n_new
            else:
                # 合并均值方差（Welford 合并）
                total = n + n_new
                delta = mean_x - m
                m = m + delta * (n_new / total)
                v = (n*v + n_new*var_x + delta*delta*(n*n_new/total)) / total
                n = total
        self.x_mean = float(m); self.x_std = float(math.sqrt(max(v, 1e-12)))
        log(f"global x_mean={self.x_mean:.4f}, x_std={self.x_std:.4f} over {len(self.items)} samples")

    def __len__(self): return len(self.items)

    @torch.no_grad()
    def _encode_img_to_latent(self, img_pil: Image.Image):
        x = self.tfm(img_pil).unsqueeze(0).to(device=self.device, dtype=self.dtype)  # [0,1]
        x = x*2 - 1
        dist = self.vae.encode(x).latent_dist
        return dist.mean  # (1,4,S/8,S/8)

    def __getitem__(self, idx):
        jp, ji = self.items[idx]
        # reps -> x_norm
        reps = load_kreps_json(jp)                # (K,512)
        x = reps.reshape(-1).astype(np.float32)   # (D_in,)
        x = (x - self.x_mean) / (self.x_std + 1e-6)
        x = torch.from_numpy(x)                   # cpu float32

        # image -> latent target + gt image
        img0 = Image.open(ji).convert("RGB")
        imgS = square_canvas(img0, self.S)
        z_true = self._encode_img_to_latent(imgS)           # (1,4,Hc,Wc)
        _, C, Hc, Wc = z_true.shape
        y_true = z_true.reshape(-1).float()                  # (D_out,)
        y_std  = float(y_true.std().clamp(min=1e-6))

        img_gt = self.tfm(imgS).unsqueeze(0).to(device=self.device, dtype=self.dtype) * 2 - 1

        sample = {
            "x": x,                       # cpu float32 (D_in,)
            "y_true": y_true,             # cpu float32 (D_out,)
            "img_gt": img_gt.squeeze(0),  # device dtype (3,S,S) in [-1,1]
            "C": C, "Hc": Hc, "Wc": Wc, "y_true_std": y_std,
            "stem": ji.stem
        }
        return sample

def build_pairs_from_manifest(manifest: Path) -> List[Tuple[str,str]]:
    pairs = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        a, b = line.split("\t")
        pairs.append((a.strip(), b.strip()))
    return pairs

def build_pairs_from_dirs(payload_dir: Path, image_dir: Path) -> List[Tuple[str,str]]:
    pairs = []
    # 期望 payload 结构：.../stem.json/stem_payload.json
    payload_jsons = glob.glob(str(payload_dir / "*.json"))
    for j in payload_jsons:
        stem = Path(j).stem
        inner = Path(j) / f"{stem}_payload.json"
        img = image_dir / f"{stem}.jpg"
        if inner.exists() and img.exists():
            pairs.append((str(inner), str(img)))
        else:
            # 容忍 png/jpeg
            for ext in [".png", ".jpeg", ".jpg"]:
                img2 = image_dir / f"{stem}{ext}"
                if inner.exists() and img2.exists():
                    pairs.append((str(inner), str(img2)))
                    break
    return pairs

def main():
    ap = argparse.ArgumentParser()
    # 数据输入（任选其一）
    ap.add_argument("--pairs_tsv", type=str, default=None, help="每行: <payload_json>\\t<image>")
    ap.add_argument("--payload_dir", type=str, default=None)
    ap.add_argument("--image_dir", type=str, default=None)

    # 训练参数
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vae_id", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--precision", choices=["fp32","fp16","bf16"], default="fp32")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--p_drop", type=float, default=0.0)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = {"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[args.precision]

    # 构建样本对
    pairs = []
    if args.pairs_tsv:
        pairs = build_pairs_from_manifest(Path(args.pairs_tsv))
    elif args.payload_dir and args.image_dir:
        pairs = build_pairs_from_dirs(Path(args.payload_dir), Path(args.image_dir))
    else:
        log("ERROR: 请提供 --pairs_tsv 或 (--payload_dir 与 --image_dir)")
        sys.exit(1)

    if len(pairs) == 0:
        log("ERROR: 找不到任何样本对")
        sys.exit(1)

    # Dataset / Loader
    ds = PairSet(pairs, target_size=args.target_size, dtype=dtype, vae_id=args.vae_id, device=device)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 建模：用第一条样本推断 D_in/D_out/Hc/Wc
    first = ds[0]
    D_in = first["x"].numel()
    C, Hc, Wc = first["C"], first["Hc"], first["Wc"]
    D_out = C*Hc*Wc
    adapter = AdapterMLP(D_in, D_out, hidden=args.hidden, n_layers=args.layers, p_drop=args.p_drop).to(device)

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, betas=(0.9,0.99), weight_decay=1e-4)

    global_best = 1e9
    ckpt_best = Path(args.outdir)/"adapter_best.pth"

    log(f"train N={len(ds)} | D_in={D_in}, D_out={D_out} | S={to_m8(args.target_size)} | precision={args.precision}")
    log(f"hidden={args.hidden}, layers={args.layers}, p_drop={args.p_drop}, bs={args.batch_size}, steps={args.steps}")

    step = 0
    y_std_ema = None

    while step < args.steps:
        for batch in dl:
            step += 1
            if step > args.steps: break

            # 准备 batch tensors
            x = torch.stack([b for b in batch["x"]], dim=0).to(device=device, dtype=dtype)         # (B, D_in)
            y_true = torch.stack([b for b in batch["y_true"]], dim=0).to(device=device, dtype=torch.float32)  # (B, D_out)
            img_gt = batch["img_gt"].to(device=device, dtype=dtype)                                 # (B=bs,3,S,S)
            y_true_std = torch.tensor(batch["y_true_std"]).mean().item()
            y_std_ema = y_true_std if y_std_ema is None else 0.95*y_std_ema + 0.05*y_true_std

            opt.zero_grad(set_to_none=True)

            # 前向 & 尺度对齐
            y_pred = adapter(x).reshape(-1, C, Hc, Wc)                   # (B,4,Hc,Wc)
            y_pred = y_pred - y_pred.mean(dim=[1,2,3], keepdim=True)
            pred_std = y_pred.flatten(1).std(dim=1).clamp(min=1e-6)      # (B,)
            y_pred = y_pred * (torch.tensor(y_true_std, device=device) / pred_std)[:,None,None,None]

            # 解码
            with torch.no_grad():
                # VAE.decode 用 dtype 与模型一致
                pass
            img_pred = ds.vae.decode(y_pred.to(dtype)).sample            # (B,3,S,S) in [-1,1]

            # 损失
            y_pred_flat = y_pred.reshape(y_pred.size(0), -1).float()
            loss_img = F.l1_loss(img_pred, img_gt)
            loss_lat = F.mse_loss(y_pred_flat, y_true.to(device=device))
            cos = F.cosine_similarity(y_pred_flat, y_true.to(device=device), dim=1).mean()
            loss_cos = 1 - cos
            loss = 1.0*loss_img + 0.1*loss_lat + 0.1*loss_cos

            loss.backward()
            opt.step()

            if step % args.log_interval == 0 or step == 1:
                # 存一张并排图
                with torch.no_grad():
                    grid = torch.cat([
                        ((img_gt[:1].clamp(-1,1)+1)/2).cpu(),
                        ((img_pred[:1].clamp(-1,1)+1)/2).cpu()
                    ], dim=-1)
                    save_image(grid, Path(args.outdir)/f"train_mid_{step:06d}.png")
                log(f"step {step}/{args.steps} | Limg={float(loss_img):.4f} Llat={float(loss_lat):.4f} Lcos={float(loss_cos):.4f} | y_std_ema={y_std_ema:.4f}")

            if step % args.save_interval == 0:
                torch.save({
                    "state_dict": adapter.state_dict(),
                    "d_in": D_in, "d_out": D_out,
                    "Hc": Hc, "Wc": Wc,
                    "x_mean": ds.x_mean, "x_std": ds.x_std,
                    "target_size": to_m8(args.target_size),
                    "vae_id": args.vae_id,
                    "precision": args.precision,
                    "hidden": args.hidden, "layers": args.layers,
                    "y_true_std": float(y_std_ema if y_std_ema is not None else y_true_std),
                }, Path(args.outdir)/f"adapter_step_{step:06d}.pth")

            cur = float(loss)
            if cur < global_best:
                global_best = cur
                torch.save({
                    "state_dict": adapter.state_dict(),
                    "d_in": D_in, "d_out": D_out,
                    "Hc": Hc, "Wc": Wc,
                    "x_mean": ds.x_mean, "x_std": ds.x_std,
                    "target_size": to_m8(args.target_size),
                    "vae_id": args.vae_id,
                    "precision": args.precision,
                    "hidden": args.hidden, "layers": args.layers,
                    "y_true_std": float(y_std_ema if y_std_ema is not None else y_true_std),
                }, ckpt_best)

    log(f"✅ done. best={global_best:.6f} -> {ckpt_best}")

if __name__ == "__main__":
    main()

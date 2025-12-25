#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_token_adapter.py
- 目标：把 payload reps 映射到 K×1024 的视觉 token（教师：HF ViT-L/14）
- 单视频/小数据也能跑：用同段的多帧做监督
"""

import argparse, json, base64, zlib, glob
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

def load_reps_from_payload(p: Path):
    obj = json.loads(Path(p).read_text(encoding="utf-8"))
    raw = zlib.decompress(base64.b64decode(obj["data"]))
    N, D = (obj["shape"] if "shape" in obj else (int(obj["n_tokens"]), int(obj["dim"])))
    q = torch.frombuffer(raw, dtype=torch.int8).clone().view(N,D).float()
    scales = torch.tensor(obj["scales"], dtype=torch.float32).view(N,1)
    reps = q * scales                          # [N,D]
    return reps.reshape(1, -1)                  # [1, N*D]

class TokenAdapter(nn.Module):
    def __init__(self, in_dim, k=16, out_dim=1024, hidden=2048):
        super().__init__()
        self.k = k; self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, k*out_dim)
        )
    def forward(self, x):                       # x: [B, in_dim]
        y = self.net(x)                         # [B, k*out_dim]
        y = y.view(x.size(0), self.k, self.out_dim)
        # 归一 + 放大，避免恒零/恒常向量
        y = F.normalize(y, dim=-1) * 10.0
        return y

def pick_frames(images_glob, max_frames=3):
    xs = sorted(glob.glob(images_glob))
    if not xs: raise FileNotFoundError(f"no frames: {images_glob}")
    if len(xs) >= max_frames:
        idxs = [0, len(xs)//2, -1][:max_frames]
        xs = [xs[i] for i in idxs]
    return xs

def teacher_tokens_vitl(frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc  = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    imgs = [Image.open(p).convert("RGB") for p in frames]
    pv = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values=pv).last_hidden_state.float()  # [B,L,1024]
    # 把多帧在 L 维拼接，然后选 top-K（按 token 范数）以压成固定 K
    toks = out.transpose(0,1).contiguous().view(-1, out.size(-1))   # [B*L,1024]
    norms = toks.norm(dim=-1)
    Ksel = min(16, toks.size(0))
    topk = norms.topk(Ksel).indices
    picked = toks[topk].unsqueeze(0)                                # [1,K,1024]
    return picked

def chamfer_l2(A,B):
    # A:[B,K,1024], B:[B,Kt,1024]
    A = F.normalize(A,dim=-1); B = F.normalize(B,dim=-1)
    d = torch.cdist(A,B,p=2)                                        # [B,K,Kt]
    return 0.5*(d.min(dim=2).values.mean() + d.min(dim=1).values.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True)
    ap.add_argument("--images_glob", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = load_reps_from_payload(Path(args.payload)).to(device)  # [1, in_dim]
    in_dim = reps.size(1)

    teacher = teacher_tokens_vitl(pick_frames(args.images_glob)).to(device)  # [1,Kt,1024]

    model = TokenAdapter(in_dim, k=args.k, out_dim=1024).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for it in range(1, args.steps+1):
        pred = model(reps)                       # [1,K,1024]
        loss = 0.7*chamfer_l2(pred, teacher) + 0.3*(1.0 - F.cosine_similarity(
            pred.mean(dim=1), teacher.mean(dim=1), dim=-1).mean())

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 100 == 0:
            print(f"[{it}/{args.steps}] loss={loss.item():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "in_dim": in_dim, "k": args.k, "out_dim": 1024},
               args.out)
    print("✅ saved:", args.out)

if __name__ == "__main__":
    main()

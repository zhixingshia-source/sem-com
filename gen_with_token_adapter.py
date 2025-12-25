#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, base64, zlib, glob, sys
from pathlib import Path
import argparse
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ====== HF CLIP（只用 Vision）======
from transformers import CLIPVisionModel, CLIPImageProcessor

def log(msg): print(f"[token-adapter-v2] {msg}")

def load_reps_from_payload(p: Path):
    obj=json.loads(Path(p).read_text(encoding="utf-8"))
    raw=zlib.decompress(base64.b64decode(obj["data"]))
    N,D = (obj["shape"] if "shape" in obj else (obj["n_tokens"], obj["dim"]))
    q=torch.frombuffer(raw,dtype=torch.int8).clone().view(N,D).float()
    scales=torch.tensor(obj["scales"],dtype=torch.float32).view(N,1)
    reps=q*scales
    return reps  # (N,D)

class AdapterMLP(nn.Module):
    def __init__(self,in_dim,out_dim,hid=1024,depth=2):
        super().__init__()
        layers=[]; d=in_dim
        for _ in range(depth-1):
            layers += [nn.Linear(d,hid), nn.GELU(), nn.LayerNorm(hid)]
            d=hid
        layers += [nn.Linear(d,out_dim)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

@torch.no_grad()
def hf_gray_tokens(K:int, device:str):
    """用 HF CLIP 把中灰图编码为 K 个 patch tokens（1024维），再投到768"""
    vis = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    gray = Image.new("RGB",(224,224),(128,128,128))
    inputs = proc(images=gray, return_tensors="pt").to(device)
    out = vis(**inputs, output_hidden_states=True)
    # last_hidden_state: [1, 257, 1024] -> 丢弃 CLS，取前 K 个 patch
    toks_1024 = out.last_hidden_state[:, 1:1+K, :]   # [1,K,1024]
    # visual_projection: 1024->768
    W = vis.visual_projection.weight    # [768,1024]
    toks_768 = F.linear(toks_1024, W)   # [1,K,768]
    return toks_768

def topk_filter(tokens: torch.Tensor, keep:int):
    # tokens: [1,K,768]
    K=tokens.shape[1]
    keep=min(max(8,keep), K)          # 至少8个，且不超过K
    scores = tokens.norm(dim=-1)[0]   # [K]
    idx = scores.topk(keep).indices
    return tokens[:, idx, :]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--pick", choices=["first","middle","last"], default="middle")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep_tokens", type=int, default=8)
    args=ap.parse_args()

    torch.manual_seed(args.seed)
    device=args.device
    # === 1) 载入 payload 并通过 token-adapter ===
    reps = load_reps_from_payload(args.payload).to(device)      # (N,1024)
    reps = reps.unsqueeze(0)                                    # [1,N,1024]

    ck = torch.load(str(args.adapter), map_location="cpu")
    in_dim, out_dim = ck["in_dim"], ck["out_dim"]
    adapter = AdapterMLP(in_dim, out_dim).to(device).eval()
    adapter.load_state_dict(ck["state_dict"])
    cond_1024 = adapter(reps)                                   # [1,N,1024]
    log(f"adapter tokens: {tuple(cond_1024.shape)}  out_dim={out_dim}")

    # 1024 -> 768 用 HF 的 visual_projection（与上面生成 uc 的保持一致）
    vis = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    W = vis.visual_projection.weight                            # [768,1024]
    cond = F.linear(cond_1024, W)                               # [1,N,768]
    log(f"projected to 768: {tuple(cond.shape)}")

    # Top-K 过滤（抑制碎片）
    cond = topk_filter(cond, args.keep_tokens)                  # [1,keep,768]

    # === 2) 真正的无条件 uc：灰图 tokens，同样 768 维 & 对齐到同一 K ===
    uc_full = hf_gray_tokens(K=cond.shape[1], device=device)    # [1,keep,768]
    # 范数标定，让 uc 与 cond 的整体尺度一致（避免 CFG 失衡）
    uc = F.normalize(uc_full, dim=-1) * cond.norm(dim=-1, keepdim=True)

    log(f"cond/uc ready: cond={tuple(cond.shape)} uc={tuple(uc.shape)}")

    # === 3) CoDi 载入 & 采样（传 Tensor 列表，不传 dict） ===
    sys.path.insert(0, str(args.checkpoints.parent))
    from core.models.model_module_infer import model_module
    mm = model_module(data_dir=str(args.checkpoints),
                      pth=["CoDi_encoders.pth","CoDi_video_diffuser_8frames.pth"],
                      fp16=False)
    net = mm.net.to(device).eval()
    sampler = mm.sampler
    try: sampler.model = net
    except: pass

    # pair: [2, K, 768]（先 uc 再 cond）
    pair = torch.cat([uc, cond], dim=0).float()

    H=Wd=256; T=8
    shape = [[1,4,T,H//8,Wd//8]]
    log(f"pair fed to sampler: {tuple(pair.shape)}  cfg={args.cfg}")
    out = sampler.sample(
        steps=args.steps,
        shape=shape,
        condition=[pair],                       # 关键：列表里放 Tensor
        unconditional_guidance_scale=args.cfg,
        eta=0.0,
        verbose=False,
    )

    # 取出 Tensor & 解码一帧
    z = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (list,tuple)) else next(iter(out.values())))
    if z.dim() == 6: z = z[0]
    frames = mm.decode(z, "video")[0]
    idx = 0 if args.pick=="first" else (len(frames)//2 if args.pick=="middle" else len(frames)-1)
    frames[idx].convert("RGB").save(args.out)
    log(f"saved: {args.out}")

if __name__ == "__main__":
    main()

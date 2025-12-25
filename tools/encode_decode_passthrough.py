#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接用 CoDi 编码器把一张图片编码为 (1,L,768) 条件，再用 CoDi 视频扩散器生成，
用于检查“生成端是否正常”。不走 adapter / 压缩流程。
"""

import os, sys, argparse
from pathlib import Path
import torch, torchvision.transforms as T
from PIL import Image
import numpy as np

def log(*a): print("[passthrough]", *a)

def add_codi_to_syspath(checkpoints_dir: Path):
    # 关键：把 i-Code-V3 根目录加进 sys.path，才能 import core.*
    root = checkpoints_dir.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

def build_mm(checkpoints_dir: Path, device: str, pth_list):
    # 对齐 gen_with_adapter.py 的加载方式
    add_codi_to_syspath(checkpoints_dir)
    from core.models.model_module_infer import model_module
    mm = model_module(data_dir=str(checkpoints_dir), pth=pth_list, fp16=False).to(device).eval()
    return mm

@torch.no_grad()
def get_image_tokens_768(checkpoints_dir: Path, device: str, img_path: Path) -> torch.Tensor:
    """
    返回 (1, L, 768)：
    - 先用 encode_vision_noproj 取得 backbone tokens（可能是 1024 宽）
    - 若宽度不是 768，则在模型里查找一个 1024→768 的线性层并应用
    逻辑参考 train_adapter_semcom.py 的实现。
    """
    mm = build_mm(checkpoints_dir, device, ["CoDi_encoders.pth"])
    net = mm.net

    # 读图 & 预处理：与项目一致
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    x = (tfm(Image.open(str(img_path)).convert("RGB")) * 2 - 1).unsqueeze(0).to(device)

    # 1) backbone tokens
    if hasattr(net.clip, "encode_vision_noproj"):
        out = net.clip.encode_vision_noproj(x)
    elif hasattr(net.clip, "encode_vision"):
        out = net.clip.encode_vision(x)
    else:
        raise RuntimeError("net.clip 缺少 encode_vision(_noproj) 接口")
    seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out  # (1,L,D_clip)
    B, L, D_clip = seq.shape
    log(f"encoded image → shape={(B,L,D_clip)}")

    if D_clip == 768:
        return seq.float().contiguous()

    # 2) 找 1024→768 的线性投影（与训练脚本思路一致）
    proj = None
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Linear) and getattr(m, "in_features", None) == D_clip and getattr(m, "out_features", None) == 768:
            proj = m
            log(f"found projector: {name} ({D_clip}→768)")
            break
    if proj is None:
        raise RuntimeError(f"未找到 {D_clip}→768 的投影层；请检查 checkpoints 版本")

    seq768 = proj(seq)
    return seq768.float().contiguous()  # (1,L,768)

def extract_tensor(x):
    # 与 gen_with_adapter.py 相同的安全取张量方法
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, (list, tuple)):
        for it in x:
            t = extract_tensor(it)
            if isinstance(t, torch.Tensor): return t
    if isinstance(x, dict):
        for v in x.values():
            t = extract_tensor(v)
            if isinstance(t, torch.Tensor): return t
    return None

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = Path(args.checkpoints)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) 得到强信息条件 (1,L,768)
    cond = get_image_tokens_768(ckpts, device, Path(args.image))
    mean, std = cond.mean().item(), cond.std().item()
    _min, _max = cond.min().item(), cond.max().item()
    log(f"cond stats: mean={mean:.4f}, std={std:.4f}, min={_min:.4f}, max={_max:.4f}, L2={cond.norm().item():.4f}")

    # 2) 加载 CoDi（编码器 + 8 帧扩散器 + sampler）
    mm = build_mm(ckpts, device, ["CoDi_encoders.pth", "CoDi_video_diffuser_8frames.pth"])
    net, sampler = mm.net, mm.sampler
    try: sampler.model = net
    except Exception: pass

    # 3) 采样（与 gen_with_adapter.py 相同接口/参数）
    L = cond.shape[1]
    uc = torch.zeros_like(cond)                      # 关键：uc=zeros，避免 uc==cond 使 CFG 失效
    pair = torch.cat([uc, cond], dim=0)             # (2, L, 768)
    shape = [[1, 4, 8, 32, 32]]                     # 8 帧 256×256
    log(f"start sampling: steps={args.steps}, cfg={args.cfg}, pair={tuple(pair.shape)}")

    out = sampler.sample(
        steps=int(args.steps),
        shape=shape,
        condition=[pair],
        unconditional_guidance_scale=float(args.cfg),
        xtype=["video"],
        condition_types=["image"],
        mix_weight={"image": 1.0},
        eta=0.0,
        verbose=False,
    )
    z = extract_tensor(out)
    if z.dim() == 6: z = z[0]
    frames = mm.decode(z, "video")[0]  # list[PIL.Image]

    # 4) 保存中间帧和 GIF
    mid = frames[len(frames)//2].convert("RGB")
    png_path = outdir / f"passthrough_mid_cfg{args.cfg}.png"
    mid.save(png_path)
    gif_path = outdir / f"passthrough_cfg{args.cfg}.gif"
    import imageio
    imageio.mimsave(gif_path, [np.array(f.convert("RGB")) for f in frames], fps=args.fps)

    log(f"✅ saved mid: {png_path}")
    log(f"✅ saved gif: {gif_path}")

if __name__ == "__main__":
    main()

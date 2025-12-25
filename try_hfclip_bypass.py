import argparse, glob, os
from pathlib import Path
from PIL import Image
import torch, torch.nn.functional as F

# -------- utils --------
def pick(paths, which="middle"):
    paths = sorted(paths)
    if not paths: raise FileNotFoundError("images_glob 匹配不到帧")
    if which=="first": return paths[0]
    if which=="last":  return paths[-1]
    return paths[len(paths)//2]

def extract_tensor(x):
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, (list,tuple)):
        for it in x:
            t = extract_tensor(it)
            if isinstance(t, torch.Tensor): return t
    if isinstance(x, dict):
        for v in x.values():
            t = extract_tensor(v)
            if isinstance(t, torch.Tensor): return t
    return None

# -------- HF-CLIP(B/16) tokens: [1,257,768], std→0.38 --------
def image_to_tokens_257x768(img: Image.Image, device: str):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    vis  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    inp = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = vis(**inp, output_hidden_states=True)
        hs  = out.last_hidden_state.float()     # [1,197,768], 1+14*14

    cls   = hs[:, :1, :]                        # [1,1,768]
    patch = hs[:, 1:, :].transpose(1,2).reshape(1,768,14,14)
    patch = F.interpolate(patch, size=(16,16), mode="bicubic", align_corners=False)
    patch = patch.reshape(1,768,256).transpose(1,2)  # [1,256,768]
    tokens = torch.cat([cls, patch], dim=1)          # [1,257,768]

    # 零均值 + std 标定到 0.38（健康注意力温度）
    tokens = tokens - tokens.mean()
    tgt = 0.38
    scale = float(tgt / (tokens.std().item() + 1e-8))
    tokens = tokens * scale
    return tokens, scale

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_glob", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--cfgs", type=float, nargs="+", default=[3.0,5.0,7.5])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pick", choices=["first","middle","last"], default="middle")
    ap.add_argument("--save_gif", action="store_true")
    args = ap.parse_args()

    # 1) 取一帧作为条件
    mid_path = pick(glob.glob(args.images_glob), which=args.pick)
    gt = Image.open(mid_path).convert("RGB")

    # 2) 载 CoDi（只用它的采样器+解码器）
    import sys
    sys.path.insert(0, str(Path(args.checkpoints).parent))
    from core.models.model_module_infer import model_module
    mm = model_module(data_dir=args.checkpoints,
                      pth=["CoDi_encoders.pth","CoDi_video_diffuser_8frames.pth"],
                      fp16=False).eval()
    net, sampler = mm.net, mm.sampler
    try: sampler.model = net
    except: pass
    dev = next(net.parameters()).device  # 识别生成器所在设备

    # 3) HF-CLIP 抽 token，并对齐到生成器设备
    tokens, scale = image_to_tokens_257x768(gt, device=dev)    # [1,257,768]
    uc = torch.zeros_like(tokens)                               # [1,257,768]
    pair = torch.cat([uc, tokens], dim=0).to(dev, dtype=torch.float32)  # [2,257,768]

    # 4) 采样 & 解码
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[cond] mean={float(tokens.mean()):.6f} std={float(tokens.std()):.6f} "
          f"min={float(tokens.min()):.3f} max={float(tokens.max()):.3f} scale_used={scale:.6f}")

    torch.manual_seed(args.seed)
    shape = [[1,4,8,32,32]]  # 256x256, 8帧
    for cfg in args.cfgs:
        outs = sampler.sample(
            steps=args.steps,
            shape=shape,
            condition=[pair],                         # 关键：传张量，不传 dict
            unconditional_guidance_scale=cfg,
            xtype=["video"],
            condition_types=["image"],
            mix_weight={"image": 1.0},
            eta=0.0,
            verbose=False,
        )
        z = extract_tensor(outs)
        if z.dim()==6: z=z[0]
        frames = mm.decode(z, "video")[0]            # list of PIL
        # 存中间帧
        mid = frames[len(frames)//2].convert("RGB")
        mid.save(outdir/f"hfclip_bypass_cfg_{cfg}.png")
        print("saved:", outdir/f"hfclip_bypass_cfg_{cfg}.png")
        # 可选 gif
        if args.save_gif:
            try:
                frames[0].save(outdir/f"hfclip_bypass_cfg_{cfg}.gif",
                               save_all=True, append_images=frames[1:], duration=80, loop=0)
                print("saved:", outdir/f"hfclip_bypass_cfg_{cfg}.gif")
            except Exception as e:
                print("[gif skipped]", e)

if __name__ == "__main__":
    main()

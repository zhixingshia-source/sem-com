#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, base64, zlib, sys, glob
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ---------- utils ----------
def log(*a): print("[adapter_gen]", *a)

def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=dim, eps=eps)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = l2n(a); b = l2n(b); return a @ b.T

def load_reps_from_payload(p: Path) -> torch.Tensor:
    obj = json.loads(p.read_text(encoding="utf-8"))
    raw = zlib.decompress(base64.b64decode(obj["data"]))
    N, D = int(obj["n_tokens"]), int(obj["dim"])
    q = np.frombuffer(raw, dtype=np.int8).reshape(N, D).astype(np.float32)
    scales = np.array(obj["scales"], dtype=np.float32)  # (N,)
    # 反量化：q * scales[:, None] 恢复原始值
    reps = q * scales[:, None]  # (N, D)
    return torch.from_numpy(reps).float()

def load_reps_from_pt(p: Path) -> torch.Tensor:
    t = torch.load(str(p), map_location="cpu")
    if isinstance(t, dict) and "reps" in t: t = t["reps"]
    assert isinstance(t, torch.Tensor) and t.ndim == 2, f"{p} 不是 (N,D) Tensor"
    return t.float()

def extract_tensor(x):
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

class AdapterMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, layers: int = 2):
        super().__init__()
        net = []
        d = in_dim
        for _ in range(layers - 1):
            net += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            d = hidden
        net += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)

# ------------- Stage-1 extractor (for third.py detection) -------------
class SemanticExtractor(nn.Module):
    def __init__(self, in_dim, out_dim=512, k_tokens=8, num_heads=8,
                 attn_tau=0.4, pool_stride=4, keep_q_residual=True, q_res_w=0.2):
        super().__init__()
        self.k_tokens=k_tokens; self.out_dim=out_dim
        self.attn_tau=attn_tau; self.pool_stride=pool_stride
        self.keep_q_residual=keep_q_residual; self.q_res_w=q_res_w
        self.queries=nn.Parameter(torch.randn(1,k_tokens,out_dim))
        with torch.no_grad(): nn.init.orthogonal_(self.queries)
        self.proj_in=nn.Linear(in_dim,out_dim); self.norm_in=nn.LayerNorm(out_dim)
        self.attention=nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.norm_out=nn.LayerNorm(out_dim)
        self.ffn=nn.Sequential(nn.Linear(out_dim,out_dim*4), nn.GELU(), nn.Linear(out_dim*4,out_dim))
        self.norm_final=nn.LayerNorm(out_dim)
    def forward(self, x: torch.Tensor):
        x_proj=self.norm_in(self.proj_in(x))
        q=self.queries.expand(x.size(0),-1,-1)/max(self.attn_tau,1e-6)
        attn_out,_=self.attention(query=q,key=x_proj,value=x_proj,need_weights=False)
        if self.keep_q_residual:
            attn_out=self.norm_out((1.0-self.q_res_w)*attn_out + self.q_res_w*q)
        else:
            attn_out=self.norm_out(attn_out)
        ffn_out=self.ffn(attn_out)
        return self.norm_final(ffn_out+attn_out)

def build_extractors_from_ckpt(ckpt_path: Path, device: str):
    blob = torch.load(str(ckpt_path), map_location="cpu")
    k_tokens = int(blob.get("k_tokens", 8)); out_dim = int(blob.get("out_dim", 512))
    mods = {"text":"EXTRACTOR_T", "image":"EXTRACTOR_I", "audio":"EXTRACTOR_A", "video":"EXTRACTOR_V"}
    ex = {m: None for m in mods}
    for m, key in mods.items():
        sd = blob.get(key, None)
        if sd is None: ex[m]=None; continue
        w = sd["proj_in.weight"]; in_dim = int(w.shape[1])
        inst = SemanticExtractor(in_dim, out_dim, k_tokens).to(device).eval()
        inst.load_state_dict(sd, strict=False)
        ex[m] = inst
    return ex, k_tokens, out_dim

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--payload", type=Path, help="Stage-2 打包的 *_payload.json")
    g.add_argument("--reps", type=Path,   help="或直接用 *_reps.pt")

    ap.add_argument("--adapter", type=Path, required=True, help="训练好的 *_adapter.pt（tokens 版，out_dim=L*768）")
    ap.add_argument("--checkpoints", type=Path, required=True, help="包含 CoDi_encoders.pth 的目录")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--cfgs", type=float, nargs="+", default=[6.0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--uc_mode", choices=["same","zeros"], default="same")
    ap.add_argument("--save_gif", action="store_true")
    ap.add_argument("--fps", type=int, default=8)

    # 调参/排错辅助
    ap.add_argument("--cond_gain", type=float, default=1.0, help="对 cond 整体乘一个系数（用于数值过小时拉升）")
    ap.add_argument("--dump_cond", type=Path, default=None, help="把 cond 存成 .pt 便于排查")
    
    # third.py 检测（可选）
    ap.add_argument("--extractor_ckpt", type=Path, default=None, help="Extractor checkpoint 用于检测 cond 质量（类似 third.py）")
    ap.add_argument("--stem", type=str, default=None, help="Stem ID 用于加载原始模态数据（可从 --reps 路径推断）")
    ap.add_argument("--data_dir", type=Path, default=Path("data"), help="数据目录（用于 third.py 检测）")

    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1) reps -> flat
    if args.payload:
        reps2d = load_reps_from_payload(args.payload); src = args.payload
    else:
        reps2d = load_reps_from_pt(args.reps);         src = args.reps
    N, D = reps2d.shape
    reps_flat = reps2d.reshape(1, -1)  # (1, N*D)

    # 2) load adapter + 清理 _orig_mod. 前缀；拿到 L/768
    ck = torch.load(str(args.adapter), map_location="cpu")
    in_dim, out_dim = int(ck["in_dim"]), int(ck["out_dim"])
    L = int(ck.get("ctx_len", out_dim // 768))
    C = int(ck.get("ctx_dim", 768))
    assert C == 768, f"ctx_dim 应为 768，实际 {C}"

    model = AdapterMLP(in_dim, out_dim).to(dev).eval()
    sd = ck["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    # 长度对齐（以防万一）
    need, have = in_dim, reps_flat.numel()
    if have != need:
        log(f"⚠️ reps_flat dim={have} 与 adapter in_dim={need} 不一致，自动适配（零填充/裁剪）。")
        if have < need:
            pad = torch.zeros(1, need - have, dtype=reps_flat.dtype)
            reps_flat = torch.cat([reps_flat, pad], dim=1)
        else:
            reps_flat = reps_flat[:, :need]
    reps_flat = reps_flat.to(dev)

    with torch.no_grad():
        vec = model(reps_flat)                 # (1, L*768)
        cond = vec.view(1, L, C).contiguous()  # (1, L, 768)

    # 统计 + 可选增益
    mean = cond.mean().item(); std = cond.std().item()
    _min = cond.min().item(); _max = cond.max().item()
    log(f"in_dim={in_dim}, out_dim={out_dim}, L={L}, C={C}, reps N×D={N}×{D}")
    log(f"cond stats: mean={mean:.4f}, std={std:.4f}, min={_min:.4f}, max={_max:.4f}, L2={cond.norm().item():.4f}")
    
    # ========== third.py 检测逻辑 ==========
    detection_gain = None
    extractors = None
    mm_temp = None
    net_temp = None
    
    if args.extractor_ckpt is not None and args.extractor_ckpt.exists():
        log("\n=== 使用 third.py 方法检测 cond 质量 ===")
        try:
            # 1) 推断 stem
            stem = args.stem
            if stem is None:
                if args.reps:
                    stem = args.reps.stem.replace("_reps", "")
                elif args.payload:
                    stem = args.payload.stem.replace("_payload", "")
                else:
                    stem = None
                if stem:
                    log(f"推断 stem: {stem}")
            
            if stem:
                # 2) 提前加载 CoDi（检测需要）
                sys.path.insert(0, str(args.checkpoints.parent))
                from core.models.model_module_infer import model_module
                mm_temp = model_module(
                    data_dir=str(args.checkpoints),
                    pth=["CoDi_encoders.pth"],
                    fp16=False
                ).to(dev).eval()
                net_temp = mm_temp.net
                for p in net_temp.parameters():
                    p.requires_grad_(False)
                
                # 3) 加载 extractors
                extractors, K, D_ext = build_extractors_from_ckpt(args.extractor_ckpt, dev)
                
                # 4) 编码函数（使用 CoDi net）
                def encode_text(list_of_str):
                    with torch.no_grad():
                        out = net_temp.clip.encode_text_noproj(list_of_str)
                        z = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                        return z.float().detach().to(dev)
                
                def encode_image(paths):
                    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
                    xs = []
                    for p in paths:
                        img = Image.open(p).convert('RGB')
                        xs.append((tfm(img).unsqueeze(0)*2-1))
                    x = torch.cat(xs,0).to(dev)
                    with torch.no_grad():
                        out = net_temp.clip.encode_vision_noproj(x)
                        f = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
                        return f.float().detach()
                
                # 5) 数据加载
                DATA = args.data_dir
                DIMG, DTXT = DATA/"image", DATA/"text"
                
                def imgs_for(st, take=8):
                    xs=sorted(glob.glob(str(DIMG/f"{st}_*.jpg")))
                    if len(xs)>take: xs=xs[:take]
                    return xs
                
                def text_for(st):
                    p = DTXT / f"{st}.txt"
                    if p.exists():
                        try: return p.read_text(errors="ignore").strip()
                        except Exception: pass
                    p_srt = DTXT / f"{st}.srt"
                    if p_srt.exists():
                        s=p_srt.read_text(errors="ignore"); lines=[]
                        for ln in s.splitlines():
                            ln2=ln.strip()
                            if not ln2 or ln2.isdigit() or "-->" in ln2: continue
                            lines.append(ln2)
                        return " ".join(lines).strip()
                    return ""
                
                # 6) 提取原始模态 tokens
                t_seq = i_seq = None
                txt = text_for(stem)
                if txt: t_seq = encode_text([txt])
                ims = imgs_for(stem, take=8)
                if len(ims) > 0: i_seq = encode_image(ims).mean(0, keepdim=True)
                
                # 7) 计算 cond 与原始 tokens 的对齐度
                cond_flat = cond.squeeze(0)  # (L, 768)
                cond_dim = cond_flat.shape[-1]
                alignments = {}
                min_alignment = 1.0
                
                for name, (mod, seq) in {
                    "text": (extractors.get("text"), t_seq),
                    "image": (extractors.get("image"), i_seq),
                }.items():
                    if mod is None or seq is None: continue
                    if float(seq.var(dim=-1).mean()) < 1e-8: continue
                    with torch.no_grad():
                        z_raw = mod(seq)  # (B, K, D_ext)
                        # 处理维度：squeeze batch 维度，处理可能的额外维度
                        if z_raw.dim() == 3:
                            z = z_raw.squeeze(0)  # (K, D_ext)
                        elif z_raw.dim() == 2:
                            z = z_raw  # (K, D_ext)
                        else:
                            log(f"  ⚠️ {name}: z 维度异常 {z_raw.shape}，跳过")
                            continue
                        
                        z_dim = z.shape[-1]
                        log(f"  {name}: cond {cond_flat.shape} vs z {z.shape}")
                        
                        # 如果维度不匹配，添加投影层对齐维度
                        if z_dim != cond_dim:
                            log(f"  {name}: 维度不匹配 (cond={cond_dim}, z={z_dim})，使用投影对齐")
                            # 创建临时投影层：将 z 投影到 cond 的维度
                            proj = nn.Linear(z_dim, cond_dim).to(z.device).to(z.dtype).eval()
                            with torch.no_grad():
                                z_proj = proj(z)  # (K, cond_dim)
                            z_use = z_proj
                        else:
                            z_use = z
                        
                        # 计算 cond 与 z 的最大余弦相似度
                        S = cosine_sim(cond_flat, z_use)  # (L, K)
                        top1 = torch.max(S, dim=1).values  # (L,)
                        align = top1.mean().item()
                        alignments[name] = align
                        min_alignment = min(min_alignment, align)
                        log(f"  {name}: top1_cos_mean={align:.4f}")
                
                # 8) 根据对齐度调整增益
                if alignments:
                    consensus = sum(alignments.values()) / len(alignments)
                    log(f"  共识对齐度: {consensus:.4f}, 最低: {min_alignment:.4f}")
                    
                    # 如果对齐度太低（< 0.70，类似 third.py 的阈值），自动调整增益
                    if consensus < 0.70 or min_alignment < 0.65:
                        # 尝试通过增益提升对齐度
                        # 如果对齐度在 0.50-0.70 之间，建议增益 1.5-2.5x
                        if consensus < 0.50:
                            detection_gain = 3.0  # 严重不足
                        elif consensus < 0.60:
                            detection_gain = 2.5
                        elif consensus < 0.70:
                            detection_gain = 2.0
                        else:
                            detection_gain = 1.5
                        log(f"  ⚠️ 对齐度不足，建议应用增益 {detection_gain:.1f}x")
                    else:
                        log(f"  ✓ 对齐度正常")
                else:
                    log(f"  ⚠️ 无法提取原始模态 tokens，跳过对齐度检测")
                    
        except Exception as e:
            log(f"  ⚠️ third.py 检测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 综合增益决策
    auto_gain = None
    if std < 0.01:
        log(f"⚠️ 警告: cond 的 std={std:.4f} 过小，可能导致生成噪声。自动应用增益。")
        auto_gain = max(1.0, 0.15 / std)
    elif std < 0.05:
        log(f"⚠️ 警告: cond 的 std={std:.4f} 较小，可能导致生成质量差。自动应用增益。")
        auto_gain = max(1.0, 0.15 / std)
    elif std > 10.0:
        log(f"⚠️ 警告: cond 的 std={std:.4f} 过大，可能导致不稳定。")
    
    # 自动应用增益（优先级：detection_gain > auto_gain > 手动指定）
    final_gain = args.cond_gain
    if detection_gain is not None and args.cond_gain == 1.0:
        final_gain = detection_gain
        log(f"💡 根据 third.py 检测结果，自动应用 --cond_gain {detection_gain:.1f} 来提升对齐度")
    elif auto_gain is not None and args.cond_gain == 1.0:
        final_gain = auto_gain
        log(f"💡 自动应用 --cond_gain {auto_gain:.2f} 来提升 cond 的强度")
    elif std < 0.05 and args.cond_gain == 1.0:
        suggested_gain = max(1.0, 0.15 / std)
        log(f"💡 建议: 使用 --cond_gain {suggested_gain:.2f} 来提升 cond 的强度")

    if final_gain != 1.0:
        cond = cond * float(final_gain)
        log(f"cond scaled by {final_gain:g} → std={cond.std().item():.4f}, L2={cond.norm().item():.4f}")

    if args.dump_cond is not None:
        torch.save({"cond": cond.cpu(), "L": L, "C": C}, str(args.dump_cond))
        log(f"cond saved to {args.dump_cond}")

    # 清理检测阶段占用的 GPU 内存
    if extractors is not None or mm_temp is not None:
        log("清理检测阶段占用的 GPU 内存...")
        if extractors is not None:
            for mod in extractors.values():
                if mod is not None:
                    del mod
            del extractors
        if mm_temp is not None:
            del mm_temp
        if net_temp is not None:
            del net_temp
        if dev == "cuda":
            torch.cuda.empty_cache()
        log("内存清理完成")

    # 3) load CoDi + sample
    sys.path.insert(0, str(args.checkpoints.parent))
    from core.models.model_module_infer import model_module
    mm = model_module(
        data_dir=str(args.checkpoints),
        pth=["CoDi_encoders.pth", "CoDi_video_diffuser_8frames.pth"],
        fp16=False
    ).to(dev).eval()
    net, sampler = mm.net, mm.sampler
    try: sampler.model = net
    except Exception: pass

    torch.manual_seed(args.seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    shape = [[1, 4, 8, 32, 32]]  # 8 帧，256x256

    # 最终条件信号检查
    log(f"最终 cond: std={cond.std().item():.4f}, mean={cond.mean().item():.4f}, norm={cond.norm().item():.4f}")
    if cond.std().item() < 0.05:
        log(f"⚠️ 最终 cond 的 std 仍然很小，生成可能仍会出现噪声。建议手动增大 --cond_gain")
    
    for cfg in args.cfgs:
        uc = cond.clone() if args.uc_mode == "same" else torch.zeros_like(cond)
        pair = torch.cat([uc, cond], dim=0)  # (2, L, 768)
        
        log(f"开始生成: cfg={cfg}, steps={args.steps}, cond shape={pair.shape}")

        out = sampler.sample(
            steps=int(args.steps),
            shape=shape,
            condition=[pair],
            unconditional_guidance_scale=float(cfg),
            xtype=["video"],
            condition_types=["image"],
            mix_weight={"image": 1.0},
            eta=0.0,
            verbose=False,
        )
        z = extract_tensor(out)
        if z.dim() == 6:
            z = z[0]
        frames = mm.decode(z, "video")[0]  # list[PIL.Image]

        # 保存中间帧
        mid = frames[len(frames)//2].convert("RGB")
        out_png = args.outdir / f"adapter_cfg_{cfg}.png"
        mid.save(out_png)

        # 保存帧序列
        seq_dir = args.outdir / f"frames_cfg_{cfg}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(frames):
            im.convert("RGB").save(seq_dir / f"{i:03d}.png")

        # 可选 GIF
        if args.save_gif:
            try:
                gif_path = args.outdir / f"adapter_cfg_{cfg}.gif"
                frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                               duration=int(1000/args.fps), loop=0)
                log(f"saved gif: {gif_path}")
            except Exception as e:
                log(f"gif 保存失败：{e}")

        log(f"✅ cfg={cfg} saved mid: {out_png} | seq: {seq_dir}")

    log(f"source reps: {src}")
    log(f"adapter: {args.adapter}")

if __name__ == "__main__":
    main()
'''/home/liz0g/multi-dann-env/bin/python /home/liz0g/semantic-communication/gen_with_adapter.py \
  --payload /home/liz0g/semantic-communication/comprehensive_output/payloads/scenery1_payload.json \
  --adapter /home/liz0g/semantic-communication/snapshots/adapter_scenery1 \
  --checkpoints /home/liz0g/semantic-communication/i-Code-V3/checkpoints \
  --outdir /home/liz0g/semantic-communication/comprehensive_output/gen_scenery1 \
  --cfgs 3.5 5.0 7.0 \
  --cond_gain 1.0 \
  --extractor_ckpt /home/liz0g/semantic-communication/snapshots/dupsafe_step_0300.pth \
  --stem scenery1 \
  --save_gif
'''
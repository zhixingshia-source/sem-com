#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-3 (Tokens-aligned): reps → AdapterMLP → CoDi Vision Tokens (1, L, 768)

- 输入：Stage-2 的 reps（.pt）或 payload（.json） + 监督帧（同 stem 任一图像帧）
- 目标：CoDi 图像条件真实 tokens：(1, L, 768)，L 由编码器实际返回决定
- 关键：out_dim = L * 768；保存 L/ctx_dim 到 ckpt，推理侧直接 reshape 即可

示例：
python /home/liz0g/semantic-communication/train_adapter_tokens.py \
  --payload /home/liz0g/semantic-communication/comprehensive_output/payloads/scenery1_payload.json \
  --image   /home/liz0g/semantic-communication/data/image/scenery1_0001.jpg \
  --checkpoints /home/liz0g/semantic-communication/i-Code-V3/checkpoints \
  --out /home/liz0g/semantic-communication/runs/adapter_scenery1_tokens.pt \
  --epochs 400 --lr 5e-4 --hidden 1024 --layers 2 --amp bf16 --compile
"""
from __future__ import annotations
import argparse, json, base64, zlib, sys
from pathlib import Path
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# -------------------- Utils --------------------
def log(s: str): print(f"[train_adapter_tokens] {s}")

def set_matmul_precision():
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)

# -------------------- I/O: reps & payload --------------------
def load_reps_pt(p: Path) -> torch.Tensor:
    t = torch.load(str(p), map_location="cpu")
    if isinstance(t, dict) and "reps" in t: t = t["reps"]
    assert isinstance(t, torch.Tensor) and t.ndim == 2, f"{p} 不是 (N,D) Tensor"
    return t.float()

def load_reps_payload(p: Path) -> torch.Tensor:
    obj = json.loads(Path(p).read_text(encoding="utf-8"))
    raw = zlib.decompress(base64.b64decode(obj["data"]))
    N, D = int(obj["n_tokens"]), int(obj["dim"])
    q = torch.frombuffer(raw, dtype=torch.int8).clone().view(N, D).to(torch.float32)
    scales = torch.tensor(obj["scales"], dtype=torch.float32).view(N, 1)
    reps = q * scales             # (N, D)
    return reps

# -------------------- Model --------------------
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -------------------- CoDi: get (1, L, 768) tokens --------------------
@torch.no_grad()
def get_image_tokens(checkpoints_dir: Path, device: str, img_path: Path) -> torch.Tensor:
    """
    返回 (1, L, 768)：使用 CoDi 的 CLIP vision 编码器 + 内部 projector（若存在）。
    - 优先调用 *_noproj 拿到 backbone token（通常宽度 1024）；
    - 若宽度不是 768，自动在编码器里查找 1024→768 的线性投影，并应用；
    - 若本来就是 768，则直接返回。
    """
    sys.path.insert(0, str(checkpoints_dir.parent))
    mm_mod = import_module("core.models.model_module_infer")
    ModelModule = getattr(mm_mod, "model_module")
    enc_pth = checkpoints_dir / "CoDi_encoders.pth"
    mm = ModelModule(data_dir=str(checkpoints_dir), pth=[enc_pth.name], fp16=False)
    net = mm.net.to(device).eval()

    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    x = (tfm(Image.open(str(img_path)).convert("RGB")) * 2 - 1).unsqueeze(0).to(device)

    # 1) 先拿到 backbone 序列
    if hasattr(net.clip, "encode_vision_noproj"):
        out = net.clip.encode_vision_noproj(x)
        seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out  # (1, L, D_clip)
    elif hasattr(net.clip, "encode_vision"):
        out = net.clip.encode_vision(x)
        seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    else:
        raise RuntimeError("net.clip 不包含 encode_vision(_noproj) 接口，请检查版本。")

    B, L, D_clip = seq.shape
    if D_clip == 768:
        return seq.float()

    # 2) 在模型内寻找 1024→768 的线性投影（项目里通常存在）
    proj_layer = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear) and getattr(m, "in_features", None) == D_clip and getattr(m, "out_features", None) == 768:
            proj_layer = m
            break
    if proj_layer is None:
        raise RuntimeError(f"未找到 {D_clip}→768 的投影层；请确认 checkpoints 版本，或在此函数中手动指定 projector。")

    seq768 = proj_layer(seq)   # (1, L, 768)
    return seq768.float()

# -------------------- Train --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=Path, default=None)
    ap.add_argument("--payload", type=Path, default=None)
    ap.add_argument("--image", type=Path, required=True, help="监督帧路径（同 stem 任意一帧）")
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_cos", type=float, default=1.0)

    ap.add_argument("--noise_std", type=float, default=0.01)
    ap.add_argument("--token_drop", type=float, default=0.10)
    ap.add_argument("--early_patience", type=int, default=80)

    ap.add_argument("--amp", choices=["off","fp16","bf16"], default="off")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_matmul_precision()
    torch.manual_seed(args.seed)

    # 1) reps -> flat
    if args.reps and args.reps.exists():
        reps = load_reps_pt(args.reps)
    elif args.payload and args.payload.exists():
        reps = load_reps_payload(args.payload)
    else:
        raise FileNotFoundError("必须提供 --reps 或 --payload")
    N, Dtok = reps.shape
    r_flat = reps.reshape(1, -1)     # (1, N*Dtok)
    D_in = r_flat.shape[1]

    # 2) 目标 tokens：(1, L, 768)
    tokens = get_image_tokens(args.checkpoints, args.device, args.image)  # (1, L, 768)
    B, L, C = tokens.shape
    assert B == 1 and C == 768, f"期望 (1, L, 768)，实际 {(B, L, C)}"
    y = tokens.to(args.device).contiguous()
    D_out = L * C

    log(f"Din={D_in} → Dout={D_out}  (L={L}, C={C}) | reps N×Dtok={N}×{Dtok}")

    # 3) 模型与优化器
    model = AdapterMLP(D_in, D_out, hidden=args.hidden, layers=args.layers).to(args.device)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            log("torch.compile: on")
        except Exception as e:
            log(f"torch.compile: 跳过（{e}）")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=40, min_lr=1e-6, verbose=False)

    def cos_loss(a_seq: torch.Tensor, b_seq: torch.Tensor) -> torch.Tensor:
        # 逐 token 的 cosine（也可改为整体 flatten 后 cosine）
        a_u = l2n(a_seq, dim=-1); b_u = l2n(b_seq, dim=-1)
        return (1 - (a_u * b_u).sum(-1)).mean()

    def augment_r(flat: torch.Tensor) -> torch.Tensor:
        r = flat.view(1, N, Dtok)
        if args.token_drop > 0:
            mask = (torch.rand((1, N, 1), device=r.device) > args.token_drop).float()
            r = r * mask
        if args.noise_std > 0:
            r = r + torch.randn_like(r) * args.noise_std
        return r.view(1, -1)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp != "off" and args.device.startswith("cuda")))
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    r_flat = r_flat.to(args.device)

    best = float("inf"); no_improve = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        r = augment_r(r_flat)
        if scaler.is_enabled():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                pred = model(r).view(1, L, C)           # (1, L, 768)
                loss_mse = F.mse_loss(pred, y)
                loss_cos = cos_loss(pred, y)
                loss = args.w_mse*loss_mse + args.w_cos*loss_cos
        else:
            pred = model(r).view(1, L, C)
            loss_mse = F.mse_loss(pred, y)
            loss_cos = cos_loss(pred, y)
            loss = args.w_mse*loss_mse + args.w_cos*loss_cos

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()

        # “验证”：同样本
        with torch.no_grad():
            pred_eval = model(r_flat).view(1, L, C)
            val_mse = F.mse_loss(pred_eval, y).item()
            val_cos = cos_loss(pred_eval, y).item()
            valL = args.w_mse*val_mse + args.w_cos*val_cos
        sched.step(valL)

        log(f"ep {ep:04d} | L={loss.item():.6f} (mse={loss_mse.item():.5f}, 1-c={loss_cos.item():.5f})  "
            f"||  val={valL:.6f}  lr={opt.param_groups[0]['lr']:.2e}")

        # 保存 best
        if valL + 1e-6 < best:
            best = float(valL); no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "in_dim": D_in,
                "out_dim": D_out,          # 重要：L*768
                "ctx_len": L,              # 重要：保存 L
                "ctx_dim": C,              # 重要：应为 768
                "metrics": {"best_loss": best},
                "train_args": vars(args),
            }, str(args.out))
            log(f"[best↓] ep {ep:04d}  val={best:.6f} -> saved {args.out.name}")
        else:
            no_improve += 1
            if no_improve >= args.early_patience:
                log(f"early stop at ep {ep} (no improve {args.early_patience})")
                break

    log(f"✅ done. best={best:.6f} -> {args.out}")

if __name__ == "__main__":
    main()

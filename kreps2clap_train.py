#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kreps2clap_train.py

多样本 Adapter 训练（KToken-Reps → CoDi CLAP 音频语义向量）

输入：
  - payload_root 里的一堆 *_payload.json（由 second_ssss.py 生成的 KToken-Reps）
    命名假设为：safe_stem = 原始 stem.replace("/", "_")
      例如：原始 stem = "dev/clotho_dev_00001"
           payload 文件名 = "dev_clotho_dev_00001_payload.json"

  - audio_root 里的 Clotho wav：
      data_clotho/
        audio/
          dev/  clotho_dev_00001.wav
          eval/ clotho_eval_00001.wav

    匹配规则：
      safe_stem = "{split}_{rest}"
      还原 stem_for_audio = "{split}/{rest}"
      wav 路径 = audio_root / stem_for_audio.with_suffix(".wav")

模型：
  - AdapterMLP( D_in = max_tokens * 512, D_out = D_clap )
  - CLAP: 用 CoDi encoders 里的 net.clap 提供语义监督（冻结）

损失：
  - L_mse      : 预测 embedding 与 CLAP embedding 的 MSE
  - L_cos      : 1 - cosine_similarity(pred, gt)

用法示例（照着改路径就行）：
  python kreps2clap_train.py \
    --payload_root data_clotho_rep_k16 \
    --audio_root   data_clotho/audio \
    --outdir       runs/adapter_audio \
    --codi_root    /home/liz0g/semantic-communication/i-Code-V3 \
    --ckpt_name    CoDi_encoders.pth \
    --max_tokens   16 \
    --steps        50000 \
    --batch_size   16 \
    --seconds      8.0
"""

import os
import json
import zlib
import base64
import math
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except Exception:
    HAS_TORCHAUDIO = False


# ----------------- 小工具 -----------------
def log(*a):
    print("[kreps2clap-train]", *a, flush=True)


def load_kreps_json(p: Path) -> np.ndarray:
    """读取 KToken-Reps JSON，反量化成 float32 reps，shape=(K, D)"""
    obj = json.loads(p.read_text())
    n = int(obj["n_tokens"])
    d = int(obj["dim"])
    scales = np.asarray(obj["scales"], dtype=np.float32)  # (K,)
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(zlib.decompress(raw), dtype=np.int8)  # (K*D,)
    arr = arr.reshape(n, d).astype(np.float32)               # (K, D)
    reps = arr * scales[:, None]
    return reps


# ----------------- CoDi / CLAP 加载 + patch CLIPTokenizer -----------------
def _patch_codi_clip_tokenizer():
    """
    修掉 CoDi 自带 CLIPTokenizer 在新 transformers 上没有 encoder 属性的报错：
    CLIPTokenizer has no attribute encoder
    """
    try:
        from core.models.encoders.clip_modules import tokenization_clip
        CTok = tokenization_clip.CLIPTokenizer
    except Exception as e:
        print(f"[kreps2clap-train] ⚠️ cannot import CoDi CLIPTokenizer for patch: {e}", flush=True)
        return

    if getattr(CTok, "_patched_missing_encoder", False):
        return

    orig_get_vocab = CTok.get_vocab

    def safe_get_vocab(self):
        # CoDi 里有些实例没有 self.encoder，直接返回 added_tokens_encoder 即可，
        # CLIP 本身不会用到这些额外 token。
        if not hasattr(self, "encoder"):
            extra = getattr(self, "added_tokens_encoder", {})
            return dict(extra)
        return orig_get_vocab(self)

    CTok.get_vocab = safe_get_vocab
    CTok._patched_missing_encoder = True
    print("[kreps2clap-train] ⚙️ patched CoDi CLIPTokenizer.get_vocab", flush=True)


def load_codi(codi_root: Path, ckpt_name: str, device: str):
    """
    加载 i-Code-V3 的 encoders（clip + clap），返回 net, net.clap
    """
    codi_root = codi_root.resolve()
    ckpt_dir = codi_root / "checkpoints"

    assert (codi_root / "core/models/model_module_infer.py").exists(), \
        f"找不到 {codi_root}/core/models/model_module_infer.py"

    if str(codi_root) not in sys.path:
        sys.path.insert(0, str(codi_root))

    from core.models.model_module_infer import model_module

    _patch_codi_clip_tokenizer()

    tester = model_module(
        data_dir=str(ckpt_dir),
        pth=[ckpt_name],
        fp16=True,
    )
    net = tester.net if hasattr(tester, "net") else tester
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    clap = getattr(net, "clap", None)
    if clap is None:
        raise RuntimeError("CoDi net.clap 不存在，检查 encoders checkpoint 是否包含 clap 模块")

    print(f"[kreps2clap-train] ✅ loaded CoDi encoders from {ckpt_dir / ckpt_name}", flush=True)
    return net, clap


@torch.no_grad()
def encode_audio(clap, wav_path: Path, device: str, seconds: float = 8.0) -> torch.Tensor:
    """
    用 CoDi 的 clap 编码一个 wav，输出 L2-normalized 向量 (D_a,)
    """
    if not HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio is required for audio encoding")

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"audio file not found: {wav_path}")

    wav, sr = torchaudio.load(str(wav_path))  # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)   # 单声道

    target_sr = getattr(clap, "sample_rate", 48000)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    max_len = int(seconds * target_sr)
    if wav.size(1) >= max_len:
        wav = wav[:, :max_len]
    else:
        pad_len = max_len - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, pad_len))

    batch = wav.to(device)  # (1, T)

    if hasattr(clap, "encode_audio_noproj"):
        out = clap.encode_audio_noproj(batch)
        z = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    else:
        z = clap(batch)

    z = z.to(device=device, dtype=torch.float32)  # 可能是 (1,L,D) 或 (1,D)
    if z.ndim == 3:
        z = z.mean(dim=1)  # (1,D)
    z = z.squeeze(0)       # (D,)
    z = torch.nn.functional.normalize(z, dim=-1)
    return z


# ----------------- Adapter MLP -----------------
class AdapterMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 8192,
                 n_layers: int = 2, p_drop: float = 0.0):
        super().__init__()
        layers = []
        last = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(last, hidden), nn.GELU(), nn.Dropout(p_drop)]
            last = hidden
        layers += [nn.Linear(last, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 固定使用 FP32 计算
        return self.net(x.float()).float()


# ----------------- Dataset：KReps + wav -----------------
class KRepsAudioDataset(torch.utils.data.Dataset):
    """
    每个样本： (x_flat, wav_path, safe_stem)

    - x_flat: float32, shape = (max_tokens * D,)  (reps 截断 / 0-pad 后 flatten)
    - wav_path: str, 指向对应的 .wav 文件
    - safe_stem: 例如 "dev_clotho_dev_00001"
    """

    def __init__(self, payload_root: Path, audio_root: Path, max_tokens: int):
        super().__init__()
        self.payload_root = payload_root
        self.audio_root = audio_root
        self.max_tokens = max_tokens

        payload_paths = sorted(payload_root.glob("*_payload.json"))
        if not payload_paths:
            raise RuntimeError(f"no *_payload.json in {payload_root}")

        samples = []
        no_audio = 0

        for p in payload_paths:
            name = p.name
            if not name.endswith("_payload.json"):
                continue

            safe_st = name[:-len("_payload.json")]  # 例如 dev_clotho_dev_00001

            # 假设 safe_st 是通过 st.replace("/", "_") 得到的，
            # 原始 st 形如 "dev/clotho_dev_00001"
            parts = safe_st.split("_", 1)
            if len(parts) < 2:
                no_audio += 1
                continue
            split, rest = parts[0], parts[1]         # "dev", "clotho_dev_00001"
            rel_stem = Path(split) / rest           # dev/clotho_dev_00001
            wav_path = audio_root / rel_stem.with_suffix(".wav")

            if not wav_path.exists():
                no_audio += 1
                continue

            samples.append((p, wav_path, safe_st))

        log(f"paired samples: {len(samples)} (payloads without audio: {no_audio})")
        if not samples:
            raise RuntimeError("no paired payload/audio samples after matching")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_payload, wav_path, safe_st = self.samples[idx]

        reps = load_kreps_json(p_payload)  # (K,D)
        K, D = reps.shape

        if self.max_tokens is None or self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

        if K > self.max_tokens:
            # 均匀下采样到 max_tokens
            idxs = np.linspace(0, K - 1, self.max_tokens, dtype=int)
            reps = reps[idxs]
        elif K < self.max_tokens:
            pad = np.zeros((self.max_tokens - K, D), dtype=np.float32)
            reps = np.concatenate([reps, pad], axis=0)

        x_flat = reps.reshape(-1).astype(np.float32)  # (max_tokens * D,)

        return torch.from_numpy(x_flat), str(wav_path), safe_st


# ----------------- 主训练逻辑 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload_root", type=str, required=True,
                    help="KToken-Reps payload 根目录（*_payload.json）")
    ap.add_argument("--audio_root",   type=str, required=True,
                    help="Clotho audio 根目录（例如 data_clotho/audio）")
    ap.add_argument("--outdir",       type=str, required=True,
                    help="输出 ckpt 的目录")

    ap.add_argument("--codi_root",    type=str,
                    default="/home/liz0g/semantic-communication/i-Code-V3",
                    help="i-Code-V3 根目录")
    ap.add_argument("--ckpt_name",    type=str, default="CoDi_encoders.pth",
                    help="CoDi checkpoints 目录下的权重文件名")

    ap.add_argument("--seconds",      type=float, default=8.0,
                    help="每条音频截取 / pad 的长度（秒）")
    ap.add_argument("--max_tokens",   type=int,   default=32,
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

    if not HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio 未安装，先 `pip install torchaudio` 再跑这个脚本")

    payload_root = Path(args.payload_root)
    audio_root   = Path(args.audio_root)
    outdir       = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}, batch_size={args.batch_size}")
    log(f"payload_root={payload_root}")
    log(f"audio_root={audio_root}")
    log(f"outdir={outdir}")

    # Dataset / DataLoader
    ds = KRepsAudioDataset(payload_root, audio_root, args.max_tokens)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # 加载 CoDi encoders（只用 clap）
    net, clap = load_codi(Path(args.codi_root), args.ckpt_name, device)

    # 用一个样本探测维度
    sample_x, sample_wav, _ = ds[0]
    D_in = sample_x.numel()
    with torch.no_grad():
        z_a = encode_audio(clap, sample_wav, device, seconds=args.seconds)
        D_a = z_a.numel()

    log(f"D_in={D_in} (max_tokens={args.max_tokens}, dim=512)")
    log(f"D_audio={D_a} (来自 CoDi clap)")

    # Adapter
    adapter = AdapterMLP(
        d_in=D_in,
        d_out=D_a,
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

    # 维护输入标准化的全局 mean/std（scalar）
    running_mean = 0.0
    running_m2 = 0.0
    global_n = 0

    best = float("inf")
    ckpt_best = outdir / "adapter_clap_best.pth"

    # 简单的 audio embedding 缓存（避免每个 step 重复算同一段 audio）
    audio_cache = {}

    step = 0
    while step < args.steps:
        for x_flat, wav_paths, stems in loader:
            step += 1
            if step > args.steps:
                break

            # ---- 更新 running mean/std ----
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

            # ---- 归一化 x_flat ----
            x_flat = x_flat.to(device=device, dtype=torch.float32)  # (B, D_in)
            x_norm = (x_flat - running_mean) / running_std          # scalar broadcast

            # ---- 计算 CLAP audio embedding 作为 GT ----
            zs = []
            for wp in wav_paths:
                wp = str(wp)
                if wp in audio_cache:
                    zs.append(audio_cache[wp])
                else:
                    z = encode_audio(clap, wp, device, seconds=args.seconds)
                    audio_cache[wp] = z
                    zs.append(z)
            feat_gt = torch.stack(zs, dim=0)  # (B, D_a)
            feat_gt = F.normalize(feat_gt.float(), dim=-1)

            # ---- Adapter forward ----
            opt.zero_grad(set_to_none=True)
            y_pred = adapter(x_norm)         # (B, D_a)
            y_pred = F.normalize(y_pred.float(), dim=-1)

            # ---- 损失 ----
            loss_mse = F.mse_loss(y_pred, feat_gt)
            loss_cos = 1.0 - (y_pred * feat_gt).sum(dim=-1).mean()
            loss = args.lambda_mse * loss_mse + args.lambda_cos * loss_cos

            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()

            # ---- 日志 ----
            if step % args.log_interval == 0 or step == 1:
                log(
                    f"step {step}/{args.steps} | "
                    f"Lmse={float(loss_mse):.4f} "
                    f"Lcos={float(loss_cos):.4f} "
                    f"loss={float(loss):.4f} | "
                    f"x_mean={running_mean:.4f} x_std={running_std:.4f} "
                    f"global_n={global_n}"
                )

            # ---- 定期保存 ckpt ----
            if step % args.save_interval == 0:
                ckpt_path = outdir / f"adapter_clap_step_{step:06d}.pth"
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_a,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "max_tokens": int(args.max_tokens),
                        "seconds": float(args.seconds),
                        "codi_root": str(Path(args.codi_root).resolve()),
                        "ckpt_name": args.ckpt_name,
                    },
                    ckpt_path,
                )
                log(f"  💾 saved ckpt -> {ckpt_path}")

            # ---- best ckpt（按总 loss）----
            if float(loss) < best:
                best = float(loss)
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_a,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "max_tokens": int(args.max_tokens),
                        "seconds": float(args.seconds),
                        "codi_root": str(Path(args.codi_root).resolve()),
                        "ckpt_name": args.ckpt_name,
                    },
                    ckpt_best,
                )
                log(f"  ⭐ new best loss={best:.6f} -> {ckpt_best}")

    log(f"✅ done. best={best:.6f} -> {ckpt_best}")


if __name__ == "__main__":
    main()

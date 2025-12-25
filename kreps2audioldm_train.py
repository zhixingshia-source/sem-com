#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
kreps2audioldm_train.py

训练一个 adapter，把 Stage-2 得到的 KToken-Reps（flatten 后）映射到 AudioLDM 的文本条件空间，
同时加一个很轻量的“重建约束”：让 adapter 的输出还能预测原始音频的 log-mel 频谱（全局均值），
这样既对齐语义，又不至于和真实波形完全无关。

输出 checkpoint 结构和你现有的 kreps2gen_audio_recon.py 兼容：
{
    "state_dict": adapter.state_dict(),
    "d_in": D_in,
    "d_out": D_cond,
    "x_mean": running_mean,
    "x_std": running_std,
    "max_tokens": max_tokens,
    "audioldm_repo": "...",
}
"""

import os
import json
import math
import zlib
import base64
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image  # 为了和原脚本依赖风格一致，其实这里用不到

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
from diffusers import AudioLDMPipeline


LOG_PREFIX = "[kreps2audioldm-train]"

def log(*args):
    print(LOG_PREFIX, *args, flush=True)


# ====================== KToken-Reps 反量化 ======================

def load_kreps_from_payload(payload_path: Path, max_tokens: int) -> Tuple[np.ndarray, int, int, int]:
    """
    读取 Stage-2 保存的 KToken-Reps payload：
      {
        "kind": "KToken-Reps",
        "n_tokens": N,
       dim": D,
        "dtype": "int8",
        "encoding": "zlib+base64",
        "scales": [...],
        "data": "<base64(zlib(int8[N*D]))>",
        ...
      }

    返回：
      x_flat: (max_tokens * D,) float32
      K_raw:  原始 token 数
      K_eff:  实际使用 token 数（截断后）
      D:      每个 token 的维度
    """
    obj = json.loads(payload_path.read_text())

    if not all(k in obj for k in ("n_tokens", "dim", "data", "scales")):
        raise KeyError(f"{payload_path} 不是 KToken-Reps 格式（缺少必要字段）")

    n_tokens = int(obj["n_tokens"])
    dim = int(obj["dim"])
    dtype_str = str(obj.get("dtype", "int8")).lower()
    encoding = str(obj.get("encoding", "zlib+base64")).lower()

    data_b64 = obj["data"]
    if not isinstance(data_b64, str):
        raise TypeError("KToken-Reps: data 应该是 base64 字符串")

    raw = base64.b64decode(data_b64)
    if "zlib" in encoding:
        raw = zlib.decompress(raw)

    if "int8" in dtype_str:
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32)

    if arr.size != n_tokens * dim:
        raise ValueError(
            f"{payload_path.name}: data 长度不匹配，got={arr.size}, expect={n_tokens*dim}"
        )

    arr = arr.reshape(n_tokens, dim)  # (N,D)

    scales = np.asarray(obj["scales"], dtype=np.float32)
    if scales.ndim == 1:
        if scales.size == 1:
            scales = np.repeat(scales, n_tokens)
        elif scales.size != n_tokens:
            raise ValueError(
                f"{payload_path.name}: scales 长度不匹配，got={scales.size}, expect=1 或 {n_tokens}"
            )
        scales = scales[:, None]
    elif scales.ndim == 2:
        if scales.shape[0] != n_tokens:
            raise ValueError(
                f"{payload_path.name}: scales.shape[0]={scales.shape[0]} != n_tokens={n_tokens}"
            )
    else:
        raise ValueError(f"{payload_path.name}: scales.ndim={scales.ndim} 不支持")

    reps = arr * scales  # (N,D) float32

    K_raw = reps.shape[0]
    D = reps.shape[1]

    if max_tokens <= 0:
        raise ValueError("max_tokens 必须 >0")

    if K_raw > max_tokens:
        reps = reps[:max_tokens, :]
    elif K_raw < max_tokens:
        pad = np.zeros((max_tokens - K_raw, D), dtype=np.float32)
        reps = np.concatenate([reps, pad], axis=0)

    K_eff = min(K_raw, max_tokens)
    x_flat = reps.reshape(-1).astype(np.float32)
    return x_flat, K_raw, K_eff, D


# ====================== Dataset ======================

def read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = path.read_text(errors="ignore")
    return txt.strip()


class KRepsAudioDataset(Dataset):
    """
    每个样本：
      - x_flat: (max_tokens * D,) float32 —— KToken-Reps flatten
      - caption: str                       —— 文本 caption（喂给 AudioLDM 的 tokenizer）
      - audio: Tensor (1,T)                —— 单声道波形，采样率 audio_sr
    """

    def __init__(
        self,
        payload_root: Path,
        audio_root: Path,
        text_root: Path,
        max_tokens: int = 16,
        audio_sr: int = 16000,
        audio_seconds: float = 8.0,
    ):
        super().__init__()
        self.payload_root = payload_root
        self.audio_root = audio_root
        self.text_root = text_root
        self.max_tokens = max_tokens
        self.audio_sr = audio_sr
        self.audio_seconds = audio_seconds

        self.payload_paths = []
        self.audio_paths = []
        self.texts = []
        self.stems = []

        # 遍历 payload_root 下所有 *_payload.json
        all_payloads = sorted(self.payload_root.rglob("*_payload.json"))
        if not all_payloads:
            raise RuntimeError(f"{self.payload_root} 下没有 *_payload.json，可先跑 second_ssss.py")

        for p in all_payloads:
            rel = p.relative_to(self.payload_root)
            stem_raw = rel.as_posix()[:-len("_payload.json")]  # 比如 dev_clotho_dev_00013

            # === 关键映射：把 dev_clotho_dev_00013 变成 dev/clotho_dev_00013 ===
            if "/" in stem_raw:
                # 已经是 dev/clotho_dev_00013 这种
                audio_rel = Path(stem_raw)
            else:
                parts = stem_raw.split("_", 1)
                if len(parts) == 2 and parts[0] in ("dev", "eval"):
                    split_dir, rest = parts  # dev, clotho_dev_00013
                    audio_rel = Path(split_dir) / rest
                else:
                    # 奇怪命名就扁平找
                    audio_rel = Path(stem_raw)

            # 优先按 dev/clotho_dev_xxx 去找
            audio_path = self.audio_root / audio_rel.with_suffix(".wav")
            text_path = self.text_root / audio_rel.with_suffix(".txt")

            # 再兜底：如果按子目录找不到，就尝试扁平文件名
            if not audio_path.exists():
                flat = self.audio_root / f"{stem_raw}.wav"
                if flat.exists():
                    audio_path = flat
            if not text_path.exists():
                flat = self.text_root / f"{stem_raw}.txt"
                if flat.exists():
                    text_path = flat

            if not audio_path.exists():
                print(f"[kreps2audioldm-train] ⚠️ 跳过 {stem_raw}: 找不到音频 {audio_path}")
                continue

            caption = read_text_file(text_path)
            if not caption:
                print(f"[kreps2audioldm-train] ⚠️ 跳过 {stem_raw}: 文本为空或缺失 {text_path}")
                continue

            self.payload_paths.append(p)
            self.audio_paths.append(audio_path)
            self.texts.append(caption)
            self.stems.append(stem_raw)

        if not self.payload_paths:
            raise RuntimeError("没有任何 payload+音频+文本 成功对齐的样本，检查路径是否正确。")

        print(f"[kreps2audioldm-train] dataset 构建完成: {len(self.payload_paths)} 个样本")
    def __len__(self):
        return len(self.payload_paths)


    def _load_audio(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))  # (C,T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.audio_sr:
            wav = torchaudio.functional.resample(wav, sr, self.audio_sr)
        max_len = int(self.audio_seconds * self.audio_sr)
        if wav.size(1) >= max_len:
            wav = wav[:, :max_len]
        else:
            pad = max_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        return wav  # (1, max_len)

    def __getitem__(self, idx):
        payload_path = self.payload_paths[idx]
        audio_path = self.audio_paths[idx]
        caption = self.texts[idx]
        stem = self.stems[idx]

        x_flat, K_raw, K_eff, D_token = load_kreps_from_payload(
            payload_path, max_tokens=self.max_tokens
        )
        wav = self._load_audio(audio_path)

        return (
            torch.from_numpy(x_flat),  # (max_tokens * D)
            wav,                       # (1,T)
            caption,
            stem,
        )


# ====================== 模型：Adapter + MelHead ======================

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
        return self.net(x.float()).float()


class MelHead(nn.Module):
    """
    从 adapter 输出的语义向量 z_pred (D_cond) 预测一个简单的 log-mel 全局向量，
    用作“重建约束”。这里不追求 HiFi，只是给 adapter 一个和原始波形绑定的信号。
    """

    def __init__(self, d_in: int, mel_dim: int = 64, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, mel_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).float()  # (B, mel_dim)


# ====================== AudioLDM 文本编码 & mel 特征 ======================

@torch.no_grad()
def extract_text_embeds(pipe: AudioLDMPipeline, texts: List[str], device: torch.device) -> torch.Tensor:
    """
    用 AudioLDM 自带的 tokenizer + text_encoder 把文本变成条件嵌入 (B, D_cond)
    """
    tok = pipe.tokenizer
    text_encoder = pipe.text_encoder

    inputs = tok(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(**inputs)
    # 不同的 CLAP text 模型返回字段略有不同，这里做几种兜底
    if hasattr(outputs, "text_embeds"):
        z = outputs.text_embeds  # (B, D)
    elif hasattr(outputs, "pooler_output"):
        z = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        z = outputs.last_hidden_state.mean(dim=1)
    else:
        # 最后兜底：把第一个 tensor 拿出来
        if isinstance(outputs, tuple):
            z = outputs[0]
        else:
            raise RuntimeError("无法从 text_encoder 输出中取出 embedding，检查 AudioLDM 版本")
    return z.float()


def make_mel_transform(sample_rate: int, n_mels: int = 64):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
    )


def wav_to_mel_global(mel_tfm, wav: torch.Tensor) -> torch.Tensor:
    """
    wav: (B,1,T) or (1,T)
    返回 (B, n_mels) —— 对时间维取均值的 log-mel
    """
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)  # (1,1,T)
    B = wav.size(0)
    wav = wav.view(-1, wav.size(-1))  # (B, T)
    wav = wav.unsqueeze(1)            # (B,1,T)

    spec = mel_tfm(wav)               # (B, n_mels, time)
    spec = torch.log(spec + 1e-6)
    spec_mean = spec.mean(dim=-1)     # (B, n_mels)
    return spec_mean


# ====================== 训练主逻辑 ======================

def main():
    ap = argparse.ArgumentParser("kreps2audioldm_train")
    ap.add_argument("--payload_root", type=str, required=True,
                    help="Stage-2 的 payload 根目录（里面有 dev/...*_payload.json）")
    ap.add_argument("--audio_root", type=str, required=True,
                    help="原始音频根目录，比如 data_clotho/audio")
    ap.add_argument("--text_root", type=str, required=True,
                    help="原始文本根目录，比如 data_clotho/text")
    ap.add_argument("--outdir", type=str, required=True,
                    help="adapter ckpt 输出目录，比如 runs/adapter_audio_audioldm")

    # AudioLDM
    ap.add_argument("--audioldm_repo", type=str,
                    default="cvssp/audioldm-s-full-v2",
                    help="diffusers 上的 AudioLDM repo 名称")
    ap.add_argument("--audioldm_dtype", type=str, default="fp16",
                    choices=["fp16", "fp32"])

    # K-token 设置
    ap.add_argument("--max_tokens", type=int, default=16,
                    help="payload 中最多使用多少个 K-token（不足 0-pad）")

    # 训练超参
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)

    # 损失权重
    ap.add_argument("--lambda_sem_mse", type=float, default=1.0,
                    help="adapter 输出 vs Text-Cond 的 MSE 权重")
    ap.add_argument("--lambda_sem_cos", type=float, default=0.5,
                    help="adapter 输出 vs Text-Cond 的 1-cos 权重")
    ap.add_argument("--lambda_mel", type=float, default=0.1,
                    help="log-mel 重建损失权重（0 表示不启用）")

    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--save_interval", type=int, default=2000)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    payload_root = Path(args.payload_root).resolve()
    audio_root = Path(args.audio_root).resolve()
    text_root = Path(args.text_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"payload_root={payload_root}")
    log(f"audio_root  ={audio_root}")
    log(f"text_root   ={text_root}")
    log(f"outdir      ={outdir}")

    # 1) 构建 dataset / dataloader
    ds = KRepsAudioDataset(
        payload_root=payload_root,
        audio_root=audio_root,
        text_root=text_root,
        max_tokens=args.max_tokens,
        audio_sr=16000,
        audio_seconds=8.0,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # 2) 加载 AudioLDM（只用来做文本编码，不会训练它）
    log(f"加载 AudioLDM: {args.audioldm_repo}")
    dtype = torch.float16 if (device.type == "cuda" and args.audioldm_dtype == "fp16") else torch.float32
    pipe = AudioLDMPipeline.from_pretrained(
        args.audioldm_repo,
        torch_dtype=dtype,
    ).to(device)

    # 冻结文本编码器（我们只训练 adapter + mel_head）
    pipe.text_encoder.eval()
    log("AudioLDM 文本编码器已就绪。")

    # 3) 用一个样本推断 D_in / D_cond / mel_dim
    sample_x, sample_wav, sample_txt, _ = ds[0]
    D_in = sample_x.numel()

    with torch.no_grad():
        text_emb = extract_text_embeds(pipe, [sample_txt], device=device)  # (1,D_cond)
        D_cond = int(text_emb.shape[-1])

    # mel 向量维度（global mean over time）
    mel_tfm = make_mel_transform(sample_rate=16000, n_mels=64)
    with torch.no_grad():
        mel_vec = wav_to_mel_global(mel_tfm, sample_wav.unsqueeze(0))  # (1,64)
        mel_dim = int(mel_vec.shape[-1])

    log(f"D_in   = {D_in}  (max_tokens={args.max_tokens}, token_dim≈{D_in//args.max_tokens})")
    log(f"D_cond = {D_cond}  (AudioLDM text condition dim)")
    log(f"mel_dim= {mel_dim}  (log-mel 全局向量维度)")

    # 4) 定义模型
    adapter = AdapterMLP(
        d_in=D_in,
        d_out=D_cond,
        hidden=args.hidden,
        n_layers=args.layers,
        p_drop=args.dropout,
    ).to(device)

    mel_head = MelHead(
        d_in=D_cond,
        mel_dim=mel_dim,
        hidden=1024,
    ).to(device)

    params = list(adapter.parameters()) + list(mel_head.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    # 运行时维护 x_flat 的 scalar mean/std
    running_mean = 0.0
    running_m2 = 0.0
    global_n = 0

    best_loss = float("inf")
    ckpt_best = outdir / "adapter_audioldm_best.pth"

    step = 0
    while step < args.steps:
        for x_flat, wav, captions, stems in loader:
            step += 1
            if step > args.steps:
                break

            # ----- 更新 running mean/std （在 CPU 上算，避免溢出） -----
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

            # ----- x 标准化 -----
            x_flat = x_flat.to(device=device, dtype=torch.float32)  # (B,D_in)
            x_norm = (x_flat - running_mean) / running_std

            # ----- 文本条件 embedding -----
            with torch.no_grad():
                z_text = extract_text_embeds(pipe, list(captions), device=device)  # (B,D_cond)
                z_text = F.normalize(z_text.float(), dim=-1)

            # ----- adapter 前向 -----
            opt.zero_grad(set_to_none=True)
            z_pred = adapter(x_norm)                # (B,D_cond)
            z_pred = F.normalize(z_pred.float(), dim=-1)

            # ----- 语义损失：MSE + 1-cos -----
            loss_sem_mse = F.mse_loss(z_pred, z_text)
            loss_sem_cos = 1.0 - (z_pred * z_text).sum(dim=-1).mean()

            loss = args.lambda_sem_mse * loss_sem_mse + args.lambda_sem_cos * loss_sem_cos

            # ----- log-mel 重建约束（可选） -----
            if args.lambda_mel > 0.0:
                wav = wav.to(device=device, dtype=torch.float32)  # (B,1,T)
                mel_tfm.to(device)
                mel_vec = wav_to_mel_global(mel_tfm, wav)  # (B, mel_dim)
                mel_vec = (mel_vec - mel_vec.mean(dim=0, keepdim=True)) / (mel_vec.std(dim=0, keepdim=True) + 1e-6)

                mel_pred = mel_head(z_pred)  # (B, mel_dim)
                mel_pred = (mel_pred - mel_pred.mean(dim=0, keepdim=True)) / (mel_pred.std(dim=0, keepdim=True) + 1e-6)

                loss_mel = F.mse_loss(mel_pred, mel_vec)
                loss = loss + args.lambda_mel * loss_mel
            else:
                loss_mel = torch.tensor(0.0, device=device)

            # ----- 反向传播 -----
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # ----- 日志 -----
            if step % args.log_interval == 0 or step == 1:
                log(
                    f"step {step}/{args.steps} | "
                    f"L_sem_mse={float(loss_sem_mse):.4f} "
                    f"L_sem_cos={float(loss_sem_cos):.4f} "
                    f"L_mel={float(loss_mel):.4f} "
                    f"loss={float(loss):.4f} | "
                    f"x_mean={running_mean:.4f} x_std={running_std:.4f} "
                    f"global_n={global_n}"
                )

            # ----- 保存 checkpoint -----
            if step % args.save_interval == 0:
                ckpt_path = outdir / f"adapter_audioldm_step_{step:06d}.pth"
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_cond,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "max_tokens": int(args.max_tokens),
                        "audioldm_repo": args.audioldm_repo,
                    },
                    ckpt_path,
                )
                log(f"  💾 saved ckpt -> {ckpt_path}")

            if float(loss) < best_loss:
                best_loss = float(loss)
                torch.save(
                    {
                        "state_dict": adapter.state_dict(),
                        "d_in": D_in,
                        "d_out": D_cond,
                        "x_mean": float(running_mean),
                        "x_std": float(running_std),
                        "max_tokens": int(args.max_tokens),
                        "audioldm_repo": args.audioldm_repo,
                    },
                    ckpt_best,
                )
                log(f"  ⭐ new best loss={best_loss:.6f} -> {ckpt_best}")

        # end for loader

    log(f"✅ done. best={best_loss:.6f} -> {ckpt_best}")


if __name__ == "__main__":
    main()

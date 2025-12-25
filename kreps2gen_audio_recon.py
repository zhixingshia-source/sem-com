#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kreps2gen_audio_recon.py

KToken-Reps 语义载荷 → adapter(kreps2clap) → CLAP embedding → AudioLDM 语义恢复音频。

链路（推理）：
    payload_root/stem_payload.json
        -> 反量化 reps: (K, D_token=512)
        -> 统一到 max_tokens（截断/0-pad），拉平: (max_tokens * 512,)
        -> 用训练好的 adapter + 全局 μ, σ 标准化: x_norm
        -> adapter(x_norm) → 预测 CLAP 向量 z_pred (D_clap)
        -> 喂给 AudioLDMPipeline(prompt_embeds=z_pred)，生成音频波形
        -> 保存 stem_recon.wav

依赖：
    pip install diffusers transformers scipy

示例用法（单条）：
    python kreps2gen_audio_recon.py \
      --payload_root data_clotho_rep_k16 \
      --stem dev/clotho_dev_00001 \
      --adapter_ckpt runs/adapter_audio/adapter_clap_best.pth \
      --audioldm_repo cvssp/audioldm-s-full-v2 \
      --outdir runs/recon_audio \
      --max_tokens 16 \
      --audio_length 8.0 \
      --steps 200 \
      --guidance_scale 2.5 \
      --seed 42

批量跑全部 payload：
    python kreps2gen_audio_recon.py \
      --payload_root data_clotho_rep_k16 \
      --all \
      --adapter_ckpt runs/adapter_audio/adapter_clap_best.pth \
      --audioldm_repo cvssp/audioldm-s-full-v2 \
      --outdir runs/recon_audio \
      --max_tokens 16
"""

import argparse
import base64
import json
import math
import zlib
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AudioLDMPipeline
import scipy.io.wavfile as wavfile


# --------------------------
# 小工具
# --------------------------
def log(*a):
    print("[kreps2audio-recon]", *a, flush=True)


# --------------------------
# Adapter 定义 + 从 ckpt 恢复
# --------------------------
class AdapterMLP(nn.Module):
    """跟你训练时一样的 MLP，只是封了个壳。"""

    def __init__(self, seq: nn.Sequential):
        super().__init__()
        self.net = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保证用 float32 算
        return self.net(x.float()).float()


def build_adapter_from_ckpt(ckpt_path: Path, device: torch.device):
    """
    从你之前 kreps2clap_train 保存的 ckpt 里恢复 adapter：
        {
            "state_dict": adapter.state_dict(),
            "d_in": int,
            "d_out": int,
            "x_mean": float,
            "x_std": float,
            "max_tokens": int,
            "clap_model": "...",
        }

    这里我们不假设 hidden / layers，直接从 state_dict 的 Linear 权重形状推结构，
    跟之前图像版的 kreps2clip_infer 一样。
    """
    blob = torch.load(str(ckpt_path), map_location="cpu")

    # 兼容两种写法
    state_dict = blob.get("state_dict", blob.get("model", {}))
    if not state_dict:
        raise RuntimeError(f"空的 state_dict: {ckpt_path}")

    x_mean = float(blob.get("x_mean", 0.0))
    x_std = float(blob.get("x_std", 1.0))
    d_in = int(blob.get("d_in"))
    d_out = int(blob.get("d_out"))
    max_tokens = int(blob.get("max_tokens", 32))

    # 找出 Linear 的 index：net.0.weight, net.3.weight 之类
    weight_keys = [k for k in state_dict.keys() if k.endswith(".weight")]
    lin_indices = sorted({int(k.split(".")[1]) for k in weight_keys})

    log(f"adapter weight keys: {weight_keys}")
    log(f"linear layer indices: {lin_indices}")

    if len(lin_indices) == 2:
        # 结构：Linear -> GELU -> Dropout -> Linear
        i0, i1 = lin_indices
        w0 = state_dict[f"net.{i0}.weight"]
        w1 = state_dict[f"net.{i1}.weight"]

        d_in_chk = w0.shape[1]
        hidden = w0.shape[0]
        d_out_chk = w1.shape[0]

        assert d_in_chk == d_in, f"d_in mismatch: ckpt {d_in_chk}, meta {d_in}"
        assert d_out_chk == d_out, f"d_out mismatch: ckpt {d_out_chk}, meta {d_out}"

        seq = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, d_out),
        )
    elif len(lin_indices) == 3:
        # 备用：Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear
        i0, i1, i2 = lin_indices
        w0 = state_dict[f"net.{i0}.weight"]
        w1 = state_dict[f"net.{i1}.weight"]
        w2 = state_dict[f"net.{i2}.weight"]

        d_in_chk = w0.shape[1]
        h1 = w0.shape[0]
        h2 = w1.shape[0]
        d_out_chk = w2.shape[0]

        assert d_in_chk == d_in, f"d_in mismatch: ckpt {d_in_chk}, meta {d_in}"
        assert d_out_chk == d_out, f"d_out mismatch: ckpt {d_out_chk}, meta {d_out}"

        seq = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(h2, d_out),
        )
    else:
        raise RuntimeError(
            f"adapter 里 Linear 数量异常（只支持 2 或 3 层）：{len(lin_indices)}; keys={weight_keys}"
        )

    adapter = AdapterMLP(seq)
    adapter.load_state_dict(state_dict, strict=True)
    adapter.to(device)
    adapter.eval()

    log(
        f"adapter inferred: d_in={d_in}, d_out={d_out}, "
        f"#linears={len(lin_indices)}, max_tokens={max_tokens}"
    )
    log(f"x_mean={x_mean:.6f}, x_std={x_std:.6f}")

    return adapter, d_in, d_out, x_mean, x_std, max_tokens


# --------------------------
# 反量化 KToken-Reps payload
# --------------------------
def load_kreps_from_payload(payload_path: Path, max_tokens: int) -> Tuple[np.ndarray, int, int, int]:
    """
    读取 Stage-2 ssss.py 保存的 KToken-Reps：
        {
          "kind": "KToken-Reps",
          "version": "1.1",
          "n_tokens": N,
          "dim": D,
          "dtype": "int8",
          "encoding": "zlib+base64",
          "scales": [N],
          "data": "<base64(zlib(int8[N*D]))>",
          ...
        }

    返回：
        x_flat: (max_tokens * D,) float32
        K_raw: 原始 token 数 N
        K_eff: 实际使用的 token 数（截断后）
        D:     token 维度
    """
    obj = json.loads(payload_path.read_text())

    if not all(k in obj for k in ("n_tokens", "dim", "data", "scales")):
        raise KeyError(f"{payload_path} 不是 KToken-Reps 格式")

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
        # 基本上用不到，就当 int8 处理
        arr = np.frombuffer(raw, dtype=np.int8).astype(np.float32)

    if arr.size != n_tokens * dim:
        raise ValueError(
            f"KToken-Reps data 长度不匹配: got {arr.size}, expect {n_tokens}*{dim}={n_tokens*dim}"
        )

    arr = arr.reshape(n_tokens, dim)  # (N,D)

    scales = np.asarray(obj["scales"], dtype=np.float32)
    if scales.ndim == 1:
        if scales.size == 1:
            scales = np.repeat(scales, n_tokens)
        elif scales.size != n_tokens:
            raise ValueError(
                f"KToken-Reps scales 长度不匹配: got {scales.size}, expect 1 or {n_tokens}"
            )
        scales = scales[:, None]  # (N,1)
    elif scales.ndim == 2:
        if scales.shape[0] != n_tokens:
            raise ValueError(
                f"KToken-Reps scales shape[0]={scales.shape[0]} != n_tokens={n_tokens}"
            )
    else:
        raise ValueError(f"KToken-Reps scales ndim={scales.ndim} 不支持")

    reps = arr * scales  # (N,D) float32

    K_raw = reps.shape[0]
    D = reps.shape[1]

    # 按 max_tokens 统一长度（前 K 或均匀采样都行，这里简单取前 K）
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须 >0")
    if K_raw > max_tokens:
        reps = reps[:max_tokens, :]
    elif K_raw < max_tokens:
        pad = np.zeros((max_tokens - K_raw, D), dtype=np.float32)
        reps = np.concatenate([reps, pad], axis=0)

    K_eff = min(K_raw, max_tokens)
    x_flat = reps.reshape(-1).astype(np.float32)  # (max_tokens * D,)

    return x_flat, K_raw, K_eff, D


# --------------------------
# 主流程：payload -> audio
# --------------------------
def generate_one_audio(
    pipe: AudioLDMPipeline,
    adapter: AdapterMLP,
    d_in: int,
    x_mean: float,
    x_std: float,
    max_tokens_ckpt: int,
    payload_path: Path,
    out_wav: Path,
    max_tokens_arg: int,
    audio_length: float,
    steps: int,
    guidance_scale: float,
    seed: int,
):
    """
    对单个 payload 做一次恢复：
        payload_path -> recon.wav
    """
    use_max_tokens = max_tokens_arg if max_tokens_arg is not None else max_tokens_ckpt

    x_flat, K_raw, K_eff, D_token = load_kreps_from_payload(
        payload_path, max_tokens=use_max_tokens
    )
    log(
        f"{payload_path.name}: K_raw={K_raw}, K_eff={K_eff}, D_token={D_token}, "
        f"x_flat_dim={x_flat.shape[0]}"
    )

    if x_flat.shape[0] != d_in:
        raise ValueError(
            f"flatten dim mismatch for {payload_path.name}: "
            f"got {x_flat.shape[0]}, expect {d_in} (max_tokens * D_token)"
        )

    device = next(adapter.parameters()).device

    # (D_in,) -> 标准化 -> (1, D_in)
    x = torch.from_numpy(x_flat).to(device=device, dtype=torch.float32)
    x_norm = (x - x_mean) / (x_std + 1e-8)
    x_norm = x_norm.unsqueeze(0)  # (1, D_in)
    
    # adapter 输出向量 -> 归一化 -> 对齐到 AudioLDM 期望的维度
    with torch.no_grad():
        z = adapter(x_norm)                     # (1, D_adapter)
        z = F.normalize(z.float(), dim=-1)      # (1, D_adapter)

        # AudioLDM 里 UNet 的 class_embedding 输入维度，比如 512
        exp_dim = pipe.unet.class_embedding.in_features

        # 如果维度不匹配（比如 768 -> 512），就做简单对齐：截断或 0-pad
        if z.shape[-1] != exp_dim:
            if z.shape[-1] > exp_dim:
                # 维度太大，直接截前 exp_dim 维
                z = z[..., :exp_dim]
            else:
                # 维度太小，后面补 0
                pad = torch.zeros(z.size(0), exp_dim - z.size(-1),
                                   device=z.device, dtype=z.dtype)
                z = torch.cat([z, pad], dim=-1)

        # negative prompt 也要同维度
        z_neg = torch.zeros_like(z)             # 用 0 向量当 negative prompt



    # 构造 Generator，保证可复现
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)

    # 用 AudioLDM：只走 prompt_embeds / negative_prompt_embeds 分支，
    # 完全绕开内部的 text_encoder，避免你之前那个 CLAP text 报错
    with torch.no_grad():
        out = pipe(
            prompt=None,
            negative_prompt=None,
            audio_length_in_s=audio_length,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_waveforms_per_prompt=1,
            generator=generator,
            prompt_embeds=z,
            negative_prompt_embeds=z_neg,
        )

    audio = out.audios[0]
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.squeeze(audio)

    sample_rate = 16000  # AudioLDM 默认采样率
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_wav), sample_rate, audio)
    log(f"saved wav -> {out_wav}")


def main():
    ap = argparse.ArgumentParser("kreps2gen-audio-recon")
    ap.add_argument("--payload_root", type=str, required=True,
                    help="Stage-2 输出的 payload 根目录（里面有 *_payload.json）")
    ap.add_argument("--stem", type=str, default=None,
                    help="要恢复的样本 stem（不带后缀），例如 dev/clotho_dev_00001")
    ap.add_argument("--all", action="store_true",
                    help="对 payload_root 下所有 *_payload.json 批量恢复")

    ap.add_argument("--adapter_ckpt", type=str, required=True,
                    help="kreps2clap_train 训好的 adapter ckpt 路径")
    ap.add_argument("--audioldm_repo", type=str,
                    default="cvssp/audioldm-s-full-v2",
                    help="HuggingFace diffusers 上的 AudioLDM 模型名")

    ap.add_argument("--outdir", type=str, required=True,
                    help="wav 输出目录")

    # 生成相关超参
    ap.add_argument("--audio_length", type=float, default=8.0,
                    help="生成音频长度（秒）")
    ap.add_argument("--steps", type=int, default=200,
                    help="AudioLDM 采样步数（越大越慢）")
    ap.add_argument("--guidance_scale", type=float, default=2.5,
                    help="classifier-free guidance scale")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（<0 表示不固定）")

    # K-token 配置
    ap.add_argument("--max_tokens", type=int, default=None,
                    help="使用前多少个 token，默认用 ckpt 里存的 max_tokens")

    args = ap.parse_args()

    payload_root = Path(args.payload_root).resolve()
    adapter_ckpt = Path(args.adapter_ckpt).resolve()
    outdir = Path(args.outdir).resolve()

    assert payload_root.exists(), f"payload_root 不存在: {payload_root}"
    assert adapter_ckpt.exists(), f"adapter_ckpt 不存在: {adapter_ckpt}"
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    log(f"payload_root={payload_root}")
    log(f"outdir={outdir}")
    log(f"audioldm_repo={args.audioldm_repo}")

    # 1) 加载 adapter
    adapter, d_in, d_out, x_mean, x_std, max_tokens_ckpt = build_adapter_from_ckpt(
        adapter_ckpt, device
    )

    # 2) 加载 AudioLDM
    log("loading AudioLDMPipeline ... （第一次会下权重，稍微等一下）")
    sd_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AudioLDMPipeline.from_pretrained(
        args.audioldm_repo,
        dtype=sd_dtype,
    ).to(device)
    pipe.enable_model_cpu_offload() if device.type == "cuda" else None
    # pipe.eval()  # AudioLDMPipeline 没有 eval(), 推理本身就是 eval 模式
    log("AudioLDM 已加载。")

    # 3) 要处理哪些 stems
    stems: List[str] = []
    if args.all:
        for p in sorted(payload_root.rglob("*_payload.json")):
            # 把 "dev/clotho_dev_00001_payload.json" → "dev/clotho_dev_00001"
            rel = p.relative_to(payload_root)
            stem = rel.as_posix()[:-len("_payload.json")]
            stems.append(stem)
        assert stems, f"{payload_root} 下没找到任何 *_payload.json"
        log(f"all mode: 共 {len(stems)} 个样本")
    else:
        assert args.stem is not None, "没开 --all 时必须指定 --stem"
        stems = [args.stem]

    # 4) 逐个样本恢复
    for stem in stems:
        payload_path = payload_root / f"{stem.replace('/', '_')}_payload.json"
        if not payload_path.exists():
            log(f"跳过 {stem}: 找不到 {payload_path}")
            continue
        # 把子目录结构也保留下来
        rel = Path(stem)
        out_wav = outdir / rel.with_suffix("")  # dev/clotho_dev_00001
        out_wav = out_wav.parent / f"{rel.name}_recon.wav"

        log(f"=== {stem} ===")
        generate_one_audio(
            pipe=pipe,
            adapter=adapter,
            d_in=d_in,
            x_mean=x_mean,
            x_std=x_std,
            max_tokens_ckpt=max_tokens_ckpt,
            payload_path=payload_path,
            out_wav=out_wav,
            max_tokens_arg=args.max_tokens,
            audio_length=args.audio_length,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
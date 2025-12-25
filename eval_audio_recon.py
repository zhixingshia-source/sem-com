#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

# ==== 配置（这里路径已经是对的） ====

ROOT = Path("/home/liz0g/semantic-communication").resolve()
STEM = "dev/clotho_dev_00010"  # 想换别的样本就改这一行

ORIG_WAV = ROOT / "data_clotho" / "audio" / f"{STEM}.wav"
RECON_WAV = ROOT / "runs" / "recon_audio_audioldm" / f"{STEM}_recon.wav"


def load_wav(path, target_sr=16000):
    wav, sr = torchaudio.load(str(path))  # (C,T)
    # 转单声道
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # 重采样
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr  # (1,T), sr


def align_waveforms(x, y):
    """
    把两条波形对齐到同一长度（取 min_len），裁掉多余部分。
    x, y: Tensor (1,T)
    """
    T = min(x.size(1), y.size(1))
    return x[:, :T], y[:, :T]


def snr_db(x, y):
    """
    x: clean, y: recon，对齐后的 (1,T)
    """
    noise = x - y
    p_sig = (x ** 2).mean()
    p_noise = (noise ** 2).mean() + 1e-12
    snr = 10.0 * torch.log10(p_sig / p_noise)
    return float(snr)


def log_mel(wav, sr, n_mels=64):
    """
    简单 log-mel 频谱
    wav: (1,T)
    """
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
    )(wav)  # (1, n_mels, time)
    spec = torch.log(spec + 1e-6)
    return spec


def main():
    print("原始音频:", ORIG_WAV)
    print("重建音频:", RECON_WAV)

    assert ORIG_WAV.exists(), f"原始 wav 不存在: {ORIG_WAV}"
    assert RECON_WAV.exists(), f"重建 wav 不存在: {RECON_WAV}"

    # 1) 加载波形
    x, sr_x = load_wav(ORIG_WAV, target_sr=16000)
    y, sr_y = load_wav(RECON_WAV, target_sr=16000)

    x, y = align_waveforms(x, y)
    print(f"对齐后长度: {x.size(1)} 样本点, 采样率={sr_x}")

    # 2) 波形级指标
    mse = torch.mean((x - y) ** 2).item()
    snr = snr_db(x, y)

    # 3) 频谱级指标
    spec_x = log_mel(x, sr_x)  # (1,F,T)
    spec_y = log_mel(y, sr_x)
    spec_mse = torch.mean((spec_x - spec_y) ** 2).item()

    # 4) 简单“语义”相似：log-mel 展开后的余弦相似度
    fx = spec_x.reshape(-1)
    fy = spec_y.reshape(-1)
    cos_sim = F.cosine_similarity(fx, fy, dim=0).item()

    print("===== Eval Results (no CLAP, 不会再报错) =====")
    print(f"MSE (waveform)      : {mse:.6f}")
    print(f"SNR  (dB)           : {snr:.2f} dB")
    print(f"MSE (log-mel spec)  : {spec_mse:.6f}")
    print(f"Log-mel cosine sim  : {cos_sim:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_clotho_text_audio_sem_db.py

功能：
- 针对 Clotho 风格的数据布局：

    data_clotho/
      audio/
        dev/  *.wav
        eval/ *.wav
      text/
        dev/  *.txt
        eval/ *.txt

  要求：audio 和 text 的相对路径 + 文件名（去掉扩展名）一致，例如：
    text/dev/clotho_dev_00001.txt
    audio/dev/clotho_dev_00001.wav

- 用 CoDi 的 clip 文本编码器 + clap 音频编码器，分别计算 text / audio 的 embedding
- 对每个 stem（相对路径，不带扩展名，例如 "dev/clotho_dev_00001"）保存：
    embeds_text[ i ] : 文本全局向量 (D_t)
    embeds_audio[i ] : 音频全局向量 (D_a)

- 输出一个 .pt 文件：
    {
        "stems": List[str],
        "embeds_text": FloatTensor(N, D_text),
        "embeds_audio": FloatTensor(N, D_audio),
        "codi_root": "...",
        "ckpt": "...",   # 用到的 CoDi checkpoint
        "note": "Clotho text+audio semantic db"
    }

用法示例：
python build_clotho_text_audio_sem_db.py \
  --data_root data_clotho \
  --out_path runs/sem_db/clotho_text_audio_sem_db.pt \
  --seconds 8.0
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

def _patch_codi_clip_tokenizer():
    """
    修 CoDi 自带 CLIPTokenizer 和新版 transformers 的兼容性。
    transformers 在 __init__ 里会提前调用 get_vocab()，
    但 CoDi 的 CLIPTokenizer 这时还没设置 self.encoder，就会报 AttributeError。
    这里把 get_vocab 换成一个容错版本：
      - 如果 encoder 还没准备好，就先返回一个最小 vocab
      - encoder 有了，再走原始实现
    """
    try:
        from core.models.encoders.clip_modules import tokenization_clip
        CTok = tokenization_clip.CLIPTokenizer
    except Exception as e:
        print(f"⚠️ 无法导入 CoDi CLIPTokenizer 进行补丁: {e}", flush=True)
        return

    # 避免重复 patch
    if getattr(CTok, "_patched_missing_encoder", False):
        return

    orig_get_vocab = CTok.get_vocab

    def safe_get_vocab(self):
        # 初始化早期：encoder 还没设，这时候 transformers 也会调 get_vocab
        if not hasattr(self, "encoder"):
            extra = getattr(self, "added_tokens_encoder", {})
            return dict(extra)
        # 正常情况：encoder 已有，走原始实现
        return orig_get_vocab(self)

    CTok.get_vocab = safe_get_vocab
    CTok._patched_missing_encoder = True
    print("⚙️ 已 patch CoDi CLIPTokenizer.get_vocab（build_clotho_text_audio_sem_db.py）", flush=True)

# ============== 解析参数 ==============

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_clotho",
                    help="包含 audio/ 与 text/ 的根目录")
    ap.add_argument("--out_path", type=str, required=True,
                    help="输出 .pt 文件路径")
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="每条音频截取 / pad 的长度（秒）")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cpu，默认自动检测")
    ap.add_argument("--codi_root", type=str,
                    default="/home/liz0g/semantic-communication/i-Code-V3",
                    help="i-Code-V3 根目录（里面有 core/models/... 和 checkpoints/）")
    ap.add_argument("--ckpt_name", type=str, default="CoDi_encoders.pth",
                    help="CoDi checkpoints 目录下的权重文件名")
    return ap.parse_args()


# ============== 加载 CoDi 编码器 ==============

def load_codi(codi_root: Path, ckpt_name: str, device: str):
    import sys
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

    print(f"✅ 已加载 CoDi encoders: {ckpt_dir / ckpt_name}", flush=True)
    return net, ckpt_dir / ckpt_name


# ============== 数据扫描：对齐 text / audio ==============

def build_stem_maps(data_root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    扫描：
      text_root = data_root/text
      audio_root = data_root/audio

    stem 定义为：相对 text_root 的路径、去掉扩展名、用 POSIX 风格，例如：
      text/dev/clotho_dev_00001.txt  -> stem = "dev/clotho_dev_00001"

    audio 路径按相同相对路径 + .wav 匹配：
      audio/dev/clotho_dev_00001.wav
    """
    text_root = data_root / "text"
    audio_root = data_root / "audio"

    assert text_root.exists(), f"text_root 不存在: {text_root}"
    assert audio_root.exists(), f"audio_root 不存在: {audio_root}"

    txt_map: Dict[str, Path] = {}
    aud_map: Dict[str, Path] = {}

    # 所有 txt（递归遍历 dev / eval）
    for p in text_root.rglob("*.txt"):
        rel = p.relative_to(text_root)             # dev/clotho_dev_00001.txt
        stem = rel.with_suffix("")                 # dev/clotho_dev_00001
        stem_str = stem.as_posix()
        txt_map[stem_str] = p

    # 所有 wav（递归遍历 dev / eval）
    for p in audio_root.rglob("*.wav"):
        rel = p.relative_to(audio_root)            # dev/clotho_dev_00001.wav
        stem = rel.with_suffix("")                 # dev/clotho_dev_00001
        stem_str = stem.as_posix()
        aud_map[stem_str] = p

    common = sorted(set(txt_map.keys()) & set(aud_map.keys()))
    print(f"文本样本数 = {len(txt_map)}, 音频样本数 = {len(aud_map)}, 交集样本数 = {len(common)}", flush=True)
    assert common, "text/audio 没有交集样本，检查 data_clotho 布局和命名！"

    # 只保留交集
    txt_map2 = {k: txt_map[k] for k in common}
    aud_map2 = {k: aud_map[k] for k in common}
    return txt_map2, aud_map2


# ============== 编码函数：text & audio ==============

@torch.no_grad()
def encode_text(net, text: str, device: str) -> torch.Tensor:
    """
    用 CoDi 的 clip 文本编码器（手动 tokenizer + model，强制 input_ids.long()）：
      - 避免 encode_text_noproj 内部出现 ShortTensor → embedding 报错
      - 输出：全局池化后 L2-normalized 的 (D,) 向量
    """
    clip = getattr(net, "clip", None)
    if clip is None:
        raise RuntimeError("CoDi net.clip 不存在，检查 i-Code-V3 安装")

    tok = getattr(clip, "tokenizer", None)
    model = getattr(clip, "model", None)
    if tok is None or model is None:
        raise RuntimeError("CoDi clip 缺少 tokenizer/model，无法 encode_text")

    # tokenizer 输出 -> input_ids 强制转 long
    toks = tok(text=[text], return_tensors="pt", padding=True, truncation=True)
    input_ids = toks["input_ids"].to(device)
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()

    # 兼容 text_model / 直接 model 两种写法
    if hasattr(model, "text_model"):
        out = model.text_model(input_ids=input_ids)
        z = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    else:
        out = model(input_ids=input_ids)
        if hasattr(out, "last_hidden_state"):
            z = out.last_hidden_state
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            z = out[0]
        else:
            z = out

    z = z.to(device=device, dtype=torch.float32)  # 可能是 (1,L,D) 或 (1,D)
    if z.ndim == 3:
        z = z.mean(dim=1)  # (1,D)
    z = z.squeeze(0)       # (D,)
    z = F.normalize(z, dim=-1)
    return z


@torch.no_grad()
def encode_audio(net, wav_path: Path, device: str, seconds: float = 8.0) -> torch.Tensor:
    """
    用 CoDi 的 clap 做音频编码：
      - 读取 wav，单声道、resample 到 clap.sample_rate
      - 截取 / pad 到 seconds 秒
      - clap.encode_audio_noproj -> (B, L, D) 或 (B, D)
      - 做一个全局池化 (mean over L)，再 L2 norm
    """
    try:
        import torchaudio
    except Exception as e:
        raise RuntimeError(f"未安装 torchaudio，无法编码音频: {e}")

    clap = getattr(net, "clap", None)
    if clap is None:
        raise RuntimeError("CoDi net.clap 不存在，检查 encoders 权重是否包含 clap 模块")

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {wav_path}")

    wav, sr = torchaudio.load(str(wav_path))  # (C,T)
    # to mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)   # (1,T)

    target_sr = getattr(clap, "sample_rate", 48000)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    max_len = int(seconds * target_sr)
    if wav.size(1) >= max_len:
        # 随机截一段（这里固定从开头也可以，简单一点）
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
    z = F.normalize(z, dim=-1)
    return z


# ============== 主流程 ==============

def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📁 data_root = {data_root}", flush=True)
    print(f"📁 out_path  = {out_path}", flush=True)
    print(f"💻 device    = {device}", flush=True)

    # 1) 文本 / 音频 对齐
    txt_map, aud_map = build_stem_maps(data_root)
    stems = sorted(txt_map.keys())
    print(f"最终可用 (text+audio 都有) 的样本数 = {len(stems)}", flush=True)

    # 2) 加载 CoDi encoders
    net, ckpt_path = load_codi(Path(args.codi_root), args.ckpt_name, device)

    all_stems: List[str] = []
    text_embeds: List[torch.Tensor] = []
    audio_embeds: List[torch.Tensor] = []

    # 3) 逐样本编码
    for stem in tqdm(stems, desc="encode text+audio"):
        txt_path = txt_map[stem]
        wav_path = aud_map[stem]

        # 读文本
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            text = txt_path.read_text(errors="ignore").strip()
        if not text:
            # 文本为空就跳过
            continue

        try:
            zt = encode_text(net, text, device)
        except Exception as e:
            print(f"!! 文本编码失败 ({stem}): {e}")
            continue

        try:
            za = encode_audio(net, wav_path, device, seconds=args.seconds)
        except Exception as e:
            print(f"!! 音频编码失败 ({stem}): {e}")
            continue

        all_stems.append(stem)
        text_embeds.append(zt.cpu())
        audio_embeds.append(za.cpu())

    assert all_stems, "没有任何样本成功编码，检查前面的报错输出。"

    embeds_text = torch.stack(text_embeds, dim=0)   # (N, D_text)
    embeds_audio = torch.stack(audio_embeds, dim=0) # (N, D_audio)

    print(f"✅ 最终 sem_db: N={len(all_stems)}, "
          f"D_text={embeds_text.shape[1]}, D_audio={embeds_audio.shape[1]}", flush=True)

    obj = {
        "stems": all_stems,
        "embeds_text": embeds_text,
        "embeds_audio": embeds_audio,
        "codi_root": str(Path(args.codi_root).resolve()),
        "ckpt": str(ckpt_path),
        "note": "Clotho text+audio semantic db (CoDi clip+clap, global pooled & L2-normalized)",
    }

    torch.save(obj, out_path)
    print(f"💾 saved sem_db -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Stage-2: Token Selection → Semantic-Equivalent Clustering → Efficient Transmission
=================================================================================
Loads trained SemanticExtractor(s) from Stage-1, extracts K-tokens, applies an
optional Adapter head, selects representative tokens (multi-rep per cluster),
clusters semantic equivalents, and packs representatives into a compact int8
payload (zlib+base64).

Usage examples:
---------------
# Single stem
python ssss.py --stem -1LecxKUMDk --sel 0.60 --clu 0.70 --budget 0 --rep_per_cluster 3 --adapter mlp --adapter_depth 3 --adapter_width 1024

# All stems that have >=2 modalities present
python ssss.py --all --sel 0.60 --clu 0.70 --topk_per_modality 64

Outputs:
--------
comprehensive_output/payloads/<stem>_payload.json    # wire payload (int8+scales, base64)
comprehensive_output/payloads/<stem>_reps.pt         # torch tensor of representatives (N,D)
comprehensive_output/payloads/<stem>_members.json    # provenance list per cluster
comprehensive_output/payloads/<stem>_stats.json      # selection & size stats

Notes:
------
- No training here.
- Rebuild SemanticExtractor modules from saved state_dict shapes (old/new formats).
- Optional AdapterHead (identity/mlp/xattn) refines tokens before selection/clustering.
"""

import os, sys, glob, json, math, base64, zlib, argparse, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# -----------------------------
# Paths and CoDi backbone load
# -----------------------------
ROOT = Path.cwd()
DATA = ROOT / "data"
DVID, DIMG, DAUD, DTXT = DATA/"vedio", DATA/"image", DATA/"audio", DATA/"text"

CODI_ROOT = Path(os.environ.get("CODI_ROOT", "/home/liz0g/semantic-communication/i-Code-V3")).resolve()
assert (CODI_ROOT / "core/models/model_module_infer.py").exists(), \
    f"找不到 {CODI_ROOT}/core/models/model_module_infer.py"

if str(CODI_ROOT) not in sys.path:
    sys.path.insert(0, str(CODI_ROOT))

from core.models.model_module_infer import model_module

def _patch_codi_clip_tokenizer():
    """
    修 CoDi 自带 CLIPTokenizer 和新版 transformers 的兼容性。
    transformers 在 __init__ 里会提前调用 get_vocab()，
    但 CoDi 的 CLIPTokenizer 这时还没设置 self.encoder，会抛 AttributeError。
    这里把 get_vocab 换成一个容错版本：
      - 如果 encoder 还没准备好，就先返回一个最小 vocab
      - encoder 有了，再走原始逻辑
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
    print("⚙️ 已 patch CoDi CLIPTokenizer.get_vocab（second_ssss.py）", flush=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_dir = CODI_ROOT / "checkpoints"
pth_name = "CoDi_encoders.pth"
if not (ckpt_dir / pth_name).exists():
    cand = list(ckpt_dir.glob("*.pth")); assert cand, f"没有找到权重：{ckpt_dir}"
    pth_name = cand[0].name

_patch_codi_clip_tokenizer()
inference_tester = model_module(data_dir=str(ckpt_dir), pth=[pth_name], fp16=False).to(device).eval()
net = inference_tester.net if hasattr(inference_tester, "net") else inference_tester
for p in net.parameters(): p.requires_grad_(False)
print("✅ CoDi 编码器已加载并冻结:", ckpt_dir / pth_name)

# -----------------------------
# Utilities
# -----------------------------
def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    if x is None: return None
    if not torch.is_floating_point(x): x = x.float()
    return x.detach().to("cpu", torch.float32)

def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=dim, eps=eps)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = l2n(a); b = l2n(b); return a @ b.T

def validate_input_strength(x, modality="unknown"):
    if x is None: return False
    x_var = x.var(dim=-1).mean(); x_std = x.std()
    if float(x_var) < 1e-6 or float(x_std) < 1e-6:
        print(f"!! 警告: {modality} 特征强度不足"); return False
    x_min, x_max = x.min().item(), x.max().item()
    if abs(x_max-x_min) < 1e-6:
        print(f"!! 警告: {modality} 特征范围过小"); return False
    return True

# -----------------------------
# Encoders → sequences
# -----------------------------
@torch.no_grad()
def encode_text(list_of_str: List[str]) -> torch.Tensor:
    """用 CoDi 的 clip 做 text 编码；如果底层还是 ShortTensor 的 bug，就手动 tokenizer+model，强制 long()."""
    clip = getattr(net, "clip", None)
    if clip is None:
        raise RuntimeError("CoDi net.clip 不存在，检查 CoDi 加载。")

    # 先试官方的 encode_text_noproj
    if hasattr(clip, "encode_text_noproj"):
        try:
            out = clip.encode_text_noproj(list_of_str)
            z = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            return z.float().detach().to(device)
        except RuntimeError as e:
            # 只有遇到 ShortTensor/embedding 相关错误才兜底
            if "ShortTensor" not in str(e):
                raise

    # 兜底：手动 tokenizer -> model.text_model，并且 input_ids.long()
    tok = getattr(clip, "tokenizer", None)
    model = getattr(clip, "model", None)
    if tok is None or model is None:
        raise RuntimeError("CoDi clip 没有 tokenizer/model，无法手动 encode_text。")

    toks = tok(text=list_of_str, return_tensors="pt", padding=True, truncation=True)
    input_ids = toks["input_ids"].to(device)
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()

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

    return z.float().detach().to(device)

def _vision_tfm():
    return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

def encode_image(paths: List[str]) -> torch.Tensor:
    tfm = _vision_tfm()
    xs = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        xs.append((tfm(img).unsqueeze(0)*2-1))
    x = torch.cat(xs,0).to(device)
    with torch.no_grad():
        out = net.clip.encode_vision_noproj(x)
        f_seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        f_seq = f_seq.float().detach()
    return f_seq

def encode_vedio(paths, max_frames=8):
    import cv2
    if isinstance(paths,(str,Path)): paths=[paths]
    feats=[]
    for p in paths:
        cap=cv2.VideoCapture(str(p))
        if not cap.isOpened(): continue
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs=list(range(total)) if total<=max_frames else np.linspace(0,total-1,max_frames,dtype=int).tolist()
        frames=[]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ok, fr=cap.read()
            if ok: frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames: continue
        tfm=_vision_tfm()
        batch=torch.cat([(tfm(Image.fromarray(f)).unsqueeze(0)*2-1) for f in frames],0).to(device)
        with torch.no_grad():
            out=net.clip.encode_vision_noproj(batch)
            f = out.last_hidden_state.float().detach()
        feats.append(f.mean(0,keepdim=True))
    if not feats: return None
    return torch.cat(feats,0)

def encode_audio(paths, seconds=8.0):
    try:
        import torchaudio
    except Exception as e:
        print("!! 未安装 torchaudio，跳过音频"); return None
    if isinstance(paths,(str,Path)): paths=[paths]
    target_sr=getattr(getattr(net,"clap",None),"sample_rate",48000)
    wavs=[]
    for p in paths:
        if not Path(p).exists(): continue
        w,sr=torchaudio.load(str(p)); w=w.mean(0,keepdim=True)
        if sr!=target_sr: w=torchaudio.functional.resample(w,sr,target_sr)
        max_len=int(seconds*target_sr)
        if w.size(1)>=max_len:
            import random
            st=random.randint(0,w.size(1)-max_len); w=w[:,st:st+max_len]
        else:
            w=torch.nn.functional.pad(w,(0,max_len-w.size(1)))
        wavs.append(w)
    if not wavs: return None
    batch=torch.cat(wavs,0).to(device)
    with torch.no_grad():
        clap=net.clap
        if hasattr(clap,"encode_audio_noproj"):
            out=clap.encode_audio_noproj(batch)
            z=out.last_hidden_state if hasattr(out,"last_hidden_state") else out
        elif hasattr(clap,"forward"):
            z=clap.forward(batch)
        else:
            return None
        if z.ndim==2: z=z.unsqueeze(1)
        return z.float().detach()
# -----------------------------
# Data helpers（一次性建索引，避免 N^2 的 glob）
# -----------------------------
from collections import defaultdict

_STEMS_INDEX_BUILT = False
_TXT_STEMS = set()
_SRT_STEMS = set()
_AUD_STEMS = set()
_VID_STEMS = set()
_IMG_INDEX = defaultdict(list)  # stem -> [img paths]


def _build_index():
    global _STEMS_INDEX_BUILT, _TXT_STEMS, _SRT_STEMS, _AUD_STEMS, _VID_STEMS, _IMG_INDEX
    if _STEMS_INDEX_BUILT:
        return

    # 文本 / 字幕
    _TXT_STEMS = {p.stem for p in DTXT.glob("*.txt")}
    _SRT_STEMS = {p.stem for p in DTXT.glob("*.srt")}

    # 音频 / 视频
    _AUD_STEMS = {p.stem for p in DAUD.glob("*.wav")}
    _VID_STEMS = {p.stem for p in DVID.glob("*.mp4")}

    # 图像：一次性扫 image 目录，然后按 stem 聚类
    _IMG_INDEX = defaultdict(list)
    for p in DIMG.glob("*_*.jpg"):
        # 比如 coco_000001_0001.jpg -> stem = "coco_000001"
        parts = p.stem.split("_")
        stem = "_".join(parts[:-1]) if len(parts) > 1 else p.stem
        _IMG_INDEX[stem].append(str(p))

    _STEMS_INDEX_BUILT = True
    print(f"[INDEX] txt={len(_TXT_STEMS)} img_stems={len(_IMG_INDEX)} "
          f"aud={len(_AUD_STEMS)} vid={len(_VID_STEMS)}", flush=True)


def list_stems():
    """所有出现过的 stem（任一模态存在即可）"""
    _build_index()
    stems = set()
    stems |= _TXT_STEMS
    stems |= _SRT_STEMS
    stems |= _AUD_STEMS
    stems |= _VID_STEMS
    stems |= set(_IMG_INDEX.keys())
    return sorted(stems)


def has_modalities(st: str) -> int:
    """至少 2 个模态才算（text + image 对齐就已经满足）"""
    _build_index()
    n = 0
    if st in _TXT_STEMS or st in _SRT_STEMS:
        n += 1
    if st in _IMG_INDEX and len(_IMG_INDEX[st]) > 0:
        n += 1
    if st in _AUD_STEMS:
        n += 1
    if st in _VID_STEMS:
        n += 1
    return n


def imgs_for(st: str, take=8):
    """直接从预先建好的字典里取，不再每个 stem 都 glob 整个目录"""
    _build_index()
    xs = _IMG_INDEX.get(st, [])
    if len(xs) > take:
        xs = random.sample(xs, take)
    return xs

def text_for(st: str)->str:
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

# -----------------------------
# SemanticExtractor rebuilders (old/new formats)
# -----------------------------
class SemanticExtractorLegacyV2(nn.Module):
    """Match old Stage-1 ckpt layout (query/key/value + FFN pieces)."""
    def __init__(self, in_dim, out_dim, k_tokens, hidden_dim=None,
                 has_ffn0_ln=True, keep_q_residual=True, q_res_w=0.2,
                 actual_in_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.actual_in_dim = actual_in_dim  # 实际编码器的输出维度
        self.out_dim = out_dim
        self.k_tokens = k_tokens
        self.keep_q_residual = keep_q_residual
        self.q_res_w = q_res_w

        self.queries = nn.Parameter(torch.randn(k_tokens, out_dim))
        with torch.no_grad(): nn.init.orthogonal_(self.queries)

        # 如果实际输入维度与期望维度不匹配，添加投影层
        if actual_in_dim is not None and actual_in_dim != in_dim:
            self.proj_in = nn.Linear(actual_in_dim, in_dim, bias=False)
            # 初始化投影层为单位变换（如果可能）
            # nn.Linear(actual_in_dim, in_dim) 的权重形状是 (in_dim, actual_in_dim)
            with torch.no_grad():
                if actual_in_dim >= in_dim:
                    # 截断：保留前 in_dim 维，权重形状 (in_dim, actual_in_dim)
                    eye = torch.eye(in_dim, device=self.queries.device)  # (in_dim, in_dim)
                    pad = torch.zeros(in_dim, actual_in_dim - in_dim, device=self.queries.device)  # (in_dim, actual_in_dim - in_dim)
                    self.proj_in.weight.copy_(torch.cat([eye, pad], dim=1))  # (in_dim, actual_in_dim)
                else:
                    # 填充：从 actual_in_dim 扩展到 in_dim，权重形状 (in_dim, actual_in_dim)
                    eye = torch.eye(actual_in_dim, device=self.queries.device)  # (actual_in_dim, actual_in_dim)
                    # 前 actual_in_dim 行是单位矩阵，后 (in_dim - actual_in_dim) 行是零
                    top = eye  # (actual_in_dim, actual_in_dim)
                    bottom = torch.zeros(in_dim - actual_in_dim, actual_in_dim, device=self.queries.device)  # (in_dim - actual_in_dim, actual_in_dim)
                    self.proj_in.weight.copy_(torch.cat([top, bottom], dim=0))  # (in_dim, actual_in_dim)
        else:
            self.proj_in = None

        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

        self.has_ffn0_ln = has_ffn0_ln
        self.ffn0 = nn.LayerNorm(out_dim) if has_ffn0_ln else nn.Identity()

        self.hidden_dim = hidden_dim
        if hidden_dim and hidden_dim > 0:
            self.fc1 = nn.Linear(out_dim, hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.fc1 = None
            self.act = None
            self.fc2 = None

    def forward(self, x: torch.Tensor):
        # 如果输入维度不匹配，使用投影层
        actual_dim = x.shape[-1]
        if self.proj_in is not None:
            # 检查已存在的投影层是否匹配当前输入维度
            if actual_dim != self.proj_in.in_features:
                # 输入维度变化，需要重新创建投影层
                print(f"⚠️  检测到输入维度变化: 实际={actual_dim}, 投影层期望={self.proj_in.in_features}，重新创建投影层")
                # 从模块中删除旧的投影层
                delattr(self, 'proj_in')
                self.proj_in = None
        if self.proj_in is None and actual_dim != self.in_dim:
            # 运行时检测维度不匹配，动态添加投影层
            print(f"⚠️  检测到输入维度不匹配: 实际={actual_dim}, 期望={self.in_dim}，自动添加投影层")
            self.proj_in = nn.Linear(actual_dim, self.in_dim, bias=False).to(x.device)
            # 将投影层注册为模块的子模块（否则不会被正确管理）
            self.add_module('proj_in', self.proj_in)
            # 初始化投影层
            # nn.Linear(actual_dim, in_dim) 的权重形状是 (in_dim, actual_dim)
            with torch.no_grad():
                if actual_dim >= self.in_dim:
                    # 截断：保留前 in_dim 维，权重形状 (in_dim, actual_dim)
                    eye = torch.eye(self.in_dim, device=x.device)  # (in_dim, in_dim)
                    pad = torch.zeros(self.in_dim, actual_dim - self.in_dim, device=x.device)  # (in_dim, actual_dim - in_dim)
                    self.proj_in.weight.copy_(torch.cat([eye, pad], dim=1))  # (in_dim, actual_dim)
                else:
                    # 填充：从 actual_dim 扩展到 in_dim，权重形状 (in_dim, actual_dim)
                    eye = torch.eye(actual_dim, device=x.device)  # (actual_dim, actual_dim)
                    # 前 actual_dim 行是单位矩阵，后 (in_dim - actual_dim) 行是零
                    top = eye  # (actual_dim, actual_dim)
                    bottom = torch.zeros(self.in_dim - actual_dim, actual_dim, device=x.device)  # (in_dim - actual_dim, actual_dim)
                    self.proj_in.weight.copy_(torch.cat([top, bottom], dim=0))  # (in_dim, actual_dim)
            # 验证权重形状
            assert self.proj_in.weight.shape == (self.in_dim, actual_dim), \
                f"权重形状错误: 期望({self.in_dim}, {actual_dim}), 实际{self.proj_in.weight.shape}"
        if self.proj_in is not None:
            x = self.proj_in(x)
        
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        attn = torch.softmax((Q @ K.transpose(1, 2)) / math.sqrt(K.size(-1)), dim=-1)
        out = attn @ V
        if self.keep_q_residual:
            out = (1.0 - self.q_res_w) * out + self.q_res_w * Q
        out = self.norm(out)
        out = self.ffn0(out)
        if self.fc1 is not None:
            out = self.fc2(self.act(self.fc1(out))) + out
        return out

def build_extractors_from_ckpt(ckpt_path: Path):
    blob = torch.load(str(ckpt_path), map_location="cpu")

    # 处理全局 K 和 D（支持多种格式）
    global_k_raw = blob.get("K_TOKENS") or blob.get("k_tokens") or blob.get("K")
    global_d_raw = blob.get("OUT_DIM") or blob.get("out_dim") or blob.get("OUT") or blob.get("D")
    
    # 如果 K 是字典（stage1_dupsafe 格式：dict(text=Kt,image=Ki,audio=Ka)），取第一个值
    if isinstance(global_k_raw, dict):
        global_k = next(iter(global_k_raw.values())) if global_k_raw else None
    elif isinstance(global_k_raw, (int, float)):
        global_k = int(global_k_raw)
    else:
        global_k = None
    
    # 如果 D 是字典，取第一个值；否则直接转换
    if isinstance(global_d_raw, dict):
        global_d = next(iter(global_d_raw.values())) if global_d_raw else None
    elif isinstance(global_d_raw, (int, float)):
        global_d = int(global_d_raw)
    else:
        global_d = None

    # 支持两种检查点格式：
    # 1. 旧格式：EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V
    # 2. stage1_dupsafe 格式：ET, EI, EA (video 用 EA)
    mods_new = {"text": "ET", "image": "EI", "audio": "EA", "video": "EA"}  # stage1_dupsafe 格式
    mods_old = {"text": "EXTRACTOR_T", "image": "EXTRACTOR_I", "audio": "EXTRACTOR_A", "video": "EXTRACTOR_V"}
    
    # 检测使用哪种格式（检查是否有新格式的键存在且不为None）
    has_new_format = any(blob.get(k) is not None for k in mods_new.values())
    if has_new_format:
        mods_map = mods_new
    else:
        mods_map = mods_old
    
    ex = {m: None for m in ["text", "image", "audio", "video"]}

    inferred_k = None
    inferred_d = None

    for m, key in mods_map.items():
        sd = blob.get(key)
        if not sd:
            ex[m] = None
            continue
        
        # 对于 video 使用 EA 的情况，需要单独处理
        if m == "video" and key == "EA" and ex.get("audio") is not None:
            # 如果已经有 audio，且 video 也用 EA，跳过或重用
            ex["video"] = ex["audio"]
            continue

        # new-style? detect proj_in
        linear_keys = [k for k in sd.keys() if k.endswith("proj_in.weight") or k.endswith("key.weight")]
        if not linear_keys:
            # some Stage-1 variants saved with keys directly; try to map
            pass

        # new style not supported here explicitly; most Stage-1 we shipped used legacy blocks
        # detect legacy:
        if ("key.weight" in sd) and ("value.weight" in sd) and ("query" in sd):
            in_dim = int(sd["key.weight"].shape[1])
            out_dim = int(sd["key.weight"].shape[0])
            q = sd["query"]
            if q.ndim != 2 or q.shape[1] != out_dim:
                raise ValueError(f"{key}: query 形状异常: {tuple(q.shape)}，期望 (K, {out_dim})")
            k_tokens = int(q.shape[0])

            has_ffn0_ln = ("ffn.0.weight" in sd and sd["ffn.0.weight"].ndim == 1 and int(sd["ffn.0.weight"].shape[0]) == out_dim)
            hidden_dim = None
            if "ffn.1.weight" in sd and sd["ffn.1.weight"].ndim == 2:
                hidden_dim = int(sd["ffn.1.weight"].shape[0])
            elif "ffn.3.weight" in sd and sd["ffn.3.weight"].ndim == 2:
                hidden_dim = int(sd["ffn.3.weight"].shape[1])

            inst = SemanticExtractorLegacyV2(in_dim, out_dim, k_tokens, hidden_dim=hidden_dim, has_ffn0_ln=has_ffn0_ln).to(device).eval()

            with torch.no_grad():
                inst.queries.copy_(q)
                inst.k_proj.weight.copy_(sd["key.weight"])
                inst.v_proj.weight.copy_(sd["value.weight"])
                if "norm.weight" in sd and "norm.bias" in sd:
                    inst.norm.weight.copy_(sd["norm.weight"]); inst.norm.bias.copy_(sd["norm.bias"])
                if has_ffn0_ln:
                    if "ffn.0.weight" in sd: inst.ffn0.weight.copy_(sd["ffn.0.weight"])
                    if "ffn.0.bias" in sd:   inst.ffn0.bias.copy_(sd["ffn.0.bias"])
                if inst.fc1 is not None and "ffn.1.weight" in sd and "ffn.1.bias" in sd:
                    w = sd["ffn.1.weight"]; b = sd["ffn.1.bias"]
                    if w.shape == inst.fc1.weight.shape: inst.fc1.weight.copy_(w)
                    elif w.T.shape == inst.fc1.weight.shape: inst.fc1.weight.copy_(w.T)
                    else:
                        h, d = inst.fc1.weight.shape; inst.fc1.weight.copy_(w[:h, :d])
                    inst.fc1.bias.copy_(b[:inst.fc1.bias.shape[0]])
                if inst.fc2 is not None and "ffn.3.weight" in sd and "ffn.3.bias" in sd:
                    w = sd["ffn.3.weight"]; b = sd["ffn.3.bias"]
                    if w.shape == inst.fc2.weight.shape: inst.fc2.weight.copy_(w)
                    elif w.T.shape == inst.fc2.weight.shape: inst.fc2.weight.copy_(w.T)
                    else:
                        o, h = inst.fc2.weight.shape; inst.fc2.weight.copy_(w[:o, :h])
                    inst.fc2.bias.copy_(b[:inst.fc2.bias.shape[0]])

            ex[m] = inst
            inferred_k = inferred_k or k_tokens
            inferred_d = inferred_d or out_dim
            continue

        # unknown format
        raise KeyError(f"未知的提取器权重格式: {key}，示例键: {list(sd.keys())[:12]}")

    # 类型安全的转换
    final_k = inferred_k
    if final_k is None:
        final_k = global_k if global_k is not None else 8
    try:
        final_k = int(final_k)
    except (TypeError, ValueError):
        final_k = 8
    
    final_d = inferred_d
    if final_d is None:
        final_d = global_d if global_d is not None else 512
    try:
        final_d = int(final_d)
    except (TypeError, ValueError):
        final_d = 512
    
    return ex, final_k, final_d

# -----------------------------
# Optional Adapter Head (identity / mlp / xattn)
# -----------------------------
class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim),
                          nn.Linear(dim, hidden),
                          nn.GELU(),
                          nn.Linear(hidden, dim))
            for _ in range(depth)
        ])
    def forward(self, x):
        for blk in self.layers:
            x = x + blk(x)
        return x

class SimpleCrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln   = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x_list: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
        valid = [x for x in x_list if x is not None]
        if len(valid) <= 1:
            return x_list
        ctx = torch.cat(valid, dim=0).unsqueeze(0)  # (1, sumK, D)
        outs=[]
        for x in x_list:
            if x is None: outs.append(None); continue
            q=self.ln(x).unsqueeze(0)
            attn,_=self.attn(q, ctx, ctx, need_weights=False)
            y = x + attn.squeeze(0)
            y = y + self.ffn(y)
            outs.append(y)
        return outs

class AdapterHead(nn.Module):
    def __init__(self, dim: int, kind: str = "identity", depth: int = 2, width: int = 1024, heads: int = 4):
        super().__init__()
        self.kind = kind
        if kind == "identity":
            self.body = nn.Identity()
        elif kind == "mlp":
            self.body = ResidualMLP(dim=dim, hidden=width, depth=depth)
        elif kind == "xattn":
            self.body = SimpleCrossAttn(dim=dim, num_heads=heads)
        else:
            raise ValueError(f"Unknown adapter kind: {kind}")

    def forward(self, xt: Optional[torch.Tensor], xi: Optional[torch.Tensor],
                xa: Optional[torch.Tensor], xv: Optional[torch.Tensor]):
        if isinstance(self.body, SimpleCrossAttn):
            xt, xi, xa, xv = self.body([xt, xi, xa, xv])
            return xt, xi, xa, xv
        else:
            def go(x): return None if x is None else self.body(x)
            return go(xt), go(xi), go(xa), go(xv)

    def load_from(self, path: Optional[str]) -> None:
        if not path: return
        sd = torch.load(path, map_location="cpu")
        # 允许直接给 adapter 的 state_dict 或包含在更大字典里
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            self.load_state_dict(sd, strict=False)
            print(f"✅ Adapter 权重已加载：{path}")
        except Exception as e:
            print(f"!! Adapter 权重加载失败（忽略，继续 identity/mlp/xattn 初始化）：{e}")

# -----------------------------
# Extract tokens for a stem (inference only)
# -----------------------------
def imgs_for_token(st: str, take=8):
    return imgs_for(st, take=take)

def encode_once(st: str, take_img=8, seconds=8.0):
    t_seq=i_seq=a_seq=v_seq=None
    txt=text_for(st)
    if txt: t_seq=encode_text([txt])
    ims=imgs_for_token(st,take=take_img)
    if len(ims)>0:
        zi=encode_image(ims); i_seq=zi.mean(0,keepdim=True)
    wa=DAUD/f"{st}.wav"
    if wa.exists():
        a_seq=encode_audio([wa],seconds=seconds)
    wv=DVID/f"{st}.mp4"
    if wv.exists():
        v_seq=encode_vedio([wv],max_frames=take_img)
    return t_seq,i_seq,a_seq,v_seq

@torch.no_grad()
def extract_tokens_for_stem(extractors, st: str):
    t_seq,i_seq,a_seq,v_seq = encode_once(st, take_img=8, seconds=8.0)
    def run(mod, x, name):
        if mod is None or x is None: return None
        if not validate_input_strength(x, name): return None
        z = mod(x).squeeze(0)  # (K, D)
        return z
    zt = run(extractors.get("text"), t_seq, "文本")
    zi = run(extractors.get("image"), i_seq, "图像")
    za = run(extractors.get("audio"), a_seq, "音频")
    zv = run(extractors.get("video"), v_seq, "视频")
    return {"text": zt, "image": zi, "audio": za, "video": zv}

# -----------------------------
# Selection / Clustering / Packing
# -----------------------------
@dataclass
class SelectionResult:
    selected_tokens: Dict[str, torch.Tensor]   # modality -> (K_sel, D)
    selected_indices: Dict[str, List[int]]     # modality -> indices kept
    scores: Dict[str, torch.Tensor]            # modality -> (K,) consensus scores

def _consensus_scores(mod: str, z: torch.Tensor, others: Dict[str, torch.Tensor]) -> torch.Tensor:
    z = _to_cpu_f32(z)
    present = {m: _to_cpu_f32(t) for m, t in others.items() if t is not None and t.numel() > 0}
    if len(present) == 0:
        if z.size(0) == 1: return torch.ones(1)
        S = cosine_sim(z, z); S.fill_diagonal_(0.0); return S.mean(dim=1)
    sims = []
    for _, t in present.items():
        S = cosine_sim(z, t)
        sims.append(S.max(dim=1).values)
    return torch.stack(sims, dim=1).mean(dim=1)

def select_tokens(tokens: Dict[str, Optional[torch.Tensor]], threshold: float = 0.8, floor:int=1,
                  topk_per_modality: Optional[int]=None) -> SelectionResult:
    mods = [m for m, z in tokens.items() if z is not None]
    Zn = {m: l2n(_to_cpu_f32(tokens[m])) if tokens[m] is not None else None for m in tokens}
    selected_tokens: Dict[str, torch.Tensor] = {}
    selected_indices: Dict[str, List[int]] = {}
    scores: Dict[str, torch.Tensor] = {}
    for m in mods:
        others = {k: Zn[k] for k in mods if k != m}
        s = _consensus_scores(m, Zn[m], others)
        scores[m] = s
        keep = torch.nonzero(s >= threshold, as_tuple=False).flatten().tolist()
        if topk_per_modality is not None and len(keep) < min(topk_per_modality, Zn[m].size(0)):
            order = torch.argsort(s, descending=True).tolist()
            keep = sorted(order[:max(topk_per_modality, max(1,floor))])
        if len(keep) < max(1,floor):
            order = torch.argsort(s, descending=True).tolist()
            keep = sorted(order[:max(1,floor)])
        selected_indices[m] = keep
        selected_tokens[m] = Zn[m][keep]
    return SelectionResult(selected_tokens, selected_indices, scores)

@dataclass
class Cluster:
    id: int
    members: List[Tuple[str, int]]   # (modality, local_index_within_selected)
    rep_indices: List[int]           # indices within `members` list (one or many)
    rep_vectors: torch.Tensor        # (R, D)

def _stack_with_prov(selected: Dict[str, torch.Tensor]):
    mats = []; prov: List[Tuple[str,int]] = []
    for m,Z in selected.items():
        if Z is None or Z.numel()==0: continue
        mats.append(Z)
        for i in range(Z.size(0)): prov.append((m,i))
    if len(mats)==0: return torch.empty(0,0), []
    M=torch.cat(mats,0); return M, prov

def _connected_components_from_sim(S: torch.Tensor, thr: float) -> List[List[int]]:
    N=S.size(0)
    if N==0: return []
    A=(S>=thr).cpu().numpy().astype(np.uint8)
    np.fill_diagonal(A,0)
    visited=np.zeros(N,dtype=bool); comps=[]
    for i in range(N):
        if visited[i]: continue
        stack=[i]; visited[i]=True; comp=[i]
        while stack:
            u=stack.pop()
            nbrs=np.nonzero(A[u])[0]
            for v in nbrs:
                if not visited[v]:
                    visited[v]=True; stack.append(v); comp.append(v)
        comps.append(comp)
    return comps

def _cluster_centrality(Ssub: torch.Tensor) -> torch.Tensor:
    """节点中心性 = 与其它点相似度的均值（对角置零）"""
    S = Ssub.clone()
    n = S.size(0)
    if n == 1:
        return torch.ones(1)
    S.fill_diagonal_(0.0)
    return S.mean(dim=1)

def _greedy_multi_medoid(Ssub: torch.Tensor, R: int) -> List[int]:
    """在子图里挑 R 个代表：先选中心性最大；之后贪心最大化与已选集合的最小距离（1-cos）"""
    n = Ssub.size(0)
    if R >= n:
        return list(range(n))
    centr = _cluster_centrality(Ssub)
    chosen = [int(torch.argmax(centr).item())]
    while len(chosen) < R:
        best_idx, best_score = None, -1.0
        for i in range(n):
            if i in chosen: continue
            # 最大化最小距离（1 - 相似度）
            min_dist = min([1.0 - float(Ssub[i, j].item()) for j in chosen])
            if min_dist > best_score:
                best_score = min_dist; best_idx = i
        if best_idx is None:
            break
        chosen.append(best_idx)
    return sorted(chosen)

def cluster_semantic_equivalents(selected: Dict[str, torch.Tensor], cluster_threshold: float = 0.8,
                                 rep_per_cluster:int=1) -> List[Cluster]:
    M, prov = _stack_with_prov(selected)
    if M.numel()==0: return []
    S=cosine_sim(M,M)
    comps=_connected_components_from_sim(S, cluster_threshold)
    clusters=[]
    for cid, idxs in enumerate(comps):
        sub=M[idxs]                     # (n_c, D)
        Ssub=cosine_sim(sub, sub)
        R = max(1, rep_per_cluster)
        rep_locals = _greedy_multi_medoid(Ssub, R)
        rep_vecs = torch.stack([sub[i] for i in rep_locals], dim=0)
        members=[prov[i] for i in idxs]
        clusters.append(Cluster(id=cid, members=members, rep_indices=rep_locals, rep_vectors=rep_vecs))
    return clusters

@dataclass
class Representatives:
    rep_vectors: torch.Tensor                 # (N, D)
    rep_members: List[List[Tuple[str, int]]]  # cluster-wise provenance lists (for reps only, 1-to-1 with rep_vectors)
    cluster_sizes: List[int]                  # size of each original cluster

def choose_representatives(clusters: List[Cluster]) -> Representatives:
    if len(clusters)==0:
        return Representatives(torch.empty(0,0), [], [])
    reps = []
    rep_members = []
    sizes = []
    for c in clusters:
        reps.append(c.rep_vectors)  # (R,D)
        # 仅记录被选为代表的成员（对应 rep_indices）
        rep_members.append([c.members[i] for i in c.rep_indices])
        sizes.append(len(c.members))
    mat = torch.cat(reps, dim=0)
    return Representatives(mat, rep_members, sizes)

@dataclass
class Quantized:
    q_int8: np.ndarray        # (N,D)
    scales: np.ndarray        # (N,)
    shape: Tuple[int,int]

def quantize_int8(vectors: torch.Tensor) -> Quantized:
    V=_to_cpu_f32(vectors).numpy()
    if V.size==0: return Quantized(np.zeros((0,0),np.int8), np.zeros((0,),np.float32), (0,0))
    maxabs=np.maximum(np.max(np.abs(V),axis=1, keepdims=True), 1e-8)
    scales=(maxabs/127.0).astype(np.float32).squeeze(1)
    Q=np.clip(np.round(V/ scales[:,None]), -127, 127).astype(np.int8)
    return Quantized(Q, scales, Q.shape)

def pack_payload(reps: Representatives, q: Quantized, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if meta is None: meta={}
    raw=q.q_int8.tobytes(order="C")
    comp=zlib.compress(raw, level=9)
    b64=base64.b64encode(comp).decode("ascii")
    payload = {
        "kind": "KToken-Reps",
        "version": meta.get("version", "1.1"),
        "n_tokens": int(q.shape[0]),
        "dim": int(q.shape[1]) if len(q.shape)==2 else 0,
        "dtype": "int8",
        "encoding": "zlib+base64",
        "scales": q.scales.tolist(),
        "data": b64,
        "metadata": {
            **meta,
            "rep_members": reps.rep_members,
            "cluster_sizes": reps.cluster_sizes,
        },
    }
    return payload

# -----------------------------
# Main
# -----------------------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="数据根目录，比如 data_clotho")
    parser.add_argument("--stem", type=str, default=None, help="video/text/image/audio common stem id")
    parser.add_argument("--all", action="store_true", help="process all stems that have >=2 modalities")

    # selection & clustering
    parser.add_argument("--sel", type=float, default=0.8, help="selection threshold")
    parser.add_argument("--clu", type=float, default=0.8, help="cluster threshold")
    parser.add_argument("--floor", type=int, default=1, help="min tokens to keep per modality")
    parser.add_argument("--topk_per_modality", type=int, default=None, help="after thresholding, keep Top-K per modality before clustering")

    # representative tokens per cluster
    parser.add_argument("--rep_per_cluster", type=int, default=1, help="number of representative tokens to pick per cluster (>=1)")

    # budget
    parser.add_argument("--budget", type=int, default=0, help="max number of representative tokens to transmit (0=no cap)")
    parser.add_argument("--budget_strategy", type=str, default="by_cluster_size", choices=["by_cluster_size","uniform","none"],
                        help="how to trim when budget < total reps")

    # checkpoints & outputs
    parser.add_argument("--ckpt", type=str, default=str(ROOT/"snapshots/step_0300.pth"))
    parser.add_argument("--out", type=str, default=str(ROOT/"comprehensive_output/payloads"))

    # adapter options (optional, default identity)
    parser.add_argument("--adapter", type=str, default="identity", choices=["identity","mlp","xattn"])
    parser.add_argument("--adapter_depth", type=int, default=2)
    parser.add_argument("--adapter_width", type=int, default=1024)
    parser.add_argument("--adapter_heads", type=int, default=4)
    parser.add_argument("--adapter_ckpt", type=str, default=None, help="optional state_dict path for adapter")

    # encode knobs
    parser.add_argument("--img_take", type=int, default=8)
    parser.add_argument("--aud_seconds", type=float, default=8.0)

    args=parser.parse_args()
    # 覆盖数据根目录（如果指定了 data_root）
    global DATA, DVID, DIMG, DAUD, DTXT
    if args.data_root:
        DATA = Path(args.data_root).resolve()
        DVID, DIMG, DAUD, DTXT = DATA/"vedio", DATA/"image", DATA/"audio", DATA/"text"
        print("📁 数据目录:", DATA, flush=True)


    # 兼容 ckpt 路径里的 vedio/video 变体
    ckpt_arg = Path(args.ckpt)
    ckpt_variants = [ckpt_arg]
    s_arg = str(ckpt_arg)
    try_alts = []
    if "vedio" in s_arg: try_alts.append(Path(s_arg.replace("vedio","video")))
    if "video" in s_arg: try_alts.append(Path(s_arg.replace("video","vedio")))
    for _cand in try_alts:
        if _cand != ckpt_arg: ckpt_variants.append(_cand)
    found = next((c for c in ckpt_variants if c.exists()), None)
    assert found is not None, f"未找到权重: {ckpt_arg}（尝试过变体：{', '.join(map(str, ckpt_variants))}）"
    ckpt_path = found

    extractors, K, D = build_extractors_from_ckpt(ckpt_path)

    if args.all:
        # 对类似 Clotho 这种 text+audio 结构：递归枚举子目录，
        # 找到 audio/ 和 text/ 下同时存在的相对路径 stem，例如 "dev/clotho_dev_000001"
        audio_stems = {
            str(p.relative_to(DAUD).with_suffix(""))
            for p in DAUD.rglob("*.wav")
        }
        text_stems = {
            str(p.relative_to(DTXT).with_suffix(""))
            for p in DTXT.rglob("*.txt")
        }
        stems = sorted(audio_stems & text_stems)
        assert len(stems) > 0, f"没有找到任何 text+audio 成对样本，检查 data_root 是否正确: {DATA}"
    else:
        assert args.stem is not None, "--stem 需要一个值（或使用 --all）"
        stems=[args.stem]

    out_dir=Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Adapter head
    adapter = AdapterHead(dim=D, kind=args.adapter, depth=args.adapter_depth, width=args.adapter_width, heads=args.adapter_heads).to(device)
    adapter.load_from(args.adapter_ckpt)

    for st in stems:
        print(f"\n=== Processing stem: {st} ===")
        # 1) 抽取 tokens
        t_seq,i_seq,a_seq,v_seq = encode_once(st, take_img=args.img_take, seconds=args.aud_seconds)
        def run(mod, x, name):
            if mod is None or x is None: return None
            if not validate_input_strength(x, name): return None
            z = mod(x).squeeze(0)  # (K, D)
            return z
        zt_orig = run(extractors.get("text"), t_seq, "文本")
        zi_orig = run(extractors.get("image"), i_seq, "图像")
        za_orig = run(extractors.get("audio"), a_seq, "音频")
        zv_orig = run(extractors.get("video"), v_seq, "视频")

        # 保存原始 tokens（用于最终保存到 payload）
        orig_tokens = {"text": zt_orig, "image": zi_orig, "audio": za_orig, "video": zv_orig}

        # 2) Adapter 变换（mlp/xattn 可增强表示，仅用于选择和聚类时的跨模态对齐）
        zt, zi, za, zv = adapter(zt_orig, zi_orig, za_orig, zv_orig)

        avail = {m: z for m,z in {"text":zt,"image":zi,"audio":za,"video":zv}.items() if z is not None}
        if len(avail)==0:
            print("!! 该 stem 没有任何可用模态的 tokens，跳过")
            continue

        # 3) 按跨模态共识分数选择（支持每模态 Top-K 放大池子）
        # 注意：使用 adapter 后的 tokens 进行选择（用于跨模态对齐）
        sel = select_tokens(avail, threshold=args.sel, floor=args.floor, topk_per_modality=args.topk_per_modality)
        print("Selected per modality:", {m: len(sel.selected_indices.get(m, [])) for m in sel.selected_indices})

        # 4) 语义等价聚类 + 每簇多代表
        # 注意：使用 adapter 后的 tokens 进行聚类（用于跨模态对齐）
        clusters = cluster_semantic_equivalents(sel.selected_tokens, cluster_threshold=args.clu, rep_per_cluster=max(1,args.rep_per_cluster))
        
        # 5) 从原始 tokens（不是 adapter 后的）中提取对应的代表向量
        # cluster.members 中的 local_idx 是在 sel.selected_tokens 中的索引
        # sel.selected_indices[m][local_idx] 是在原始 tokens 中的真实索引
        
        # 先提取原始 tokens 中对应的 selected tokens（按索引）
        orig_selected = {}
        for m in sel.selected_indices:
            if orig_tokens[m] is not None and len(sel.selected_indices[m]) > 0:
                # sel.selected_indices[m] 是在原始 tokens 中的索引
                orig_selected[m] = orig_tokens[m][sel.selected_indices[m]]
            else:
                orig_selected[m] = None
        
        # 重建 clusters，使用原始 tokens 的向量
        orig_clusters = []
        for cluster in clusters:
            # 从原始 tokens 中提取对应的向量
            orig_cluster_vectors = []
            orig_cluster_members = []
            for mod, local_idx in cluster.members:
                # local_idx 是在 sel.selected_tokens[mod] 中的索引
                # 对应的原始 token 在 orig_selected[mod][local_idx]
                if orig_selected[mod] is not None and local_idx < orig_selected[mod].size(0):
                    orig_cluster_vectors.append(orig_selected[mod][local_idx:local_idx+1])
                    orig_cluster_members.append((mod, len(orig_cluster_vectors) - 1))
            
            if len(orig_cluster_vectors) == 0:
                continue
            
            orig_cluster_mat = torch.cat(orig_cluster_vectors, dim=0)
            # cluster.rep_indices 是在 cluster.members 中的索引
            # 直接使用这些索引从 orig_cluster_mat 中提取
            rep_indices_in_cluster = cluster.rep_indices
            orig_rep_vectors = orig_cluster_mat[rep_indices_in_cluster]
            
            orig_cluster = Cluster(
                id=cluster.id,
                members=orig_cluster_members,
                rep_indices=list(range(len(orig_rep_vectors))),
                rep_vectors=orig_rep_vectors
            )
            orig_clusters.append(orig_cluster)
        
        reps = choose_representatives(orig_clusters)
        N_total = reps.rep_vectors.size(0)

        # 6) 预算裁剪
        keep_idx = list(range(N_total))
        if args.budget > 0 and N_total > args.budget:
            if args.budget_strategy == "by_cluster_size":
                # 大簇优先：按簇大小把每簇 reps 摊平后的顺序，保留前 budget 个
                # 展开时按每簇的 reps 自然顺序（中心性/多样性）拼接
                # 注意：必须使用 orig_clusters（不是 clusters），因为 reps 是基于 orig_clusters 的
                expanded = []
                offset = 0
                for c in orig_clusters:
                    r = len(c.rep_indices)
                    expanded.extend(range(offset, offset+r))
                    offset += r
                keep_idx = expanded[:args.budget]
            elif args.budget_strategy == "uniform":
                # 轮询每簇取 1 个代表，直到达到上限
                # 注意：必须使用 orig_clusters（不是 clusters），因为 reps 是基于 orig_clusters 的
                expanded = []
                per = [len(c.rep_indices) for c in orig_clusters]
                pos = [0]*len(orig_clusters)
                taken = 0
                while taken < args.budget:
                    progressed = False
                    for ci, rlen in enumerate(per):
                        if pos[ci] < rlen and taken < args.budget:
                            # 计算该代表在全局 reps 中的顺序索引
                            base = sum(len(c.rep_indices) for c in orig_clusters[:ci])
                            expanded.append(base + pos[ci])
                            pos[ci] += 1
                            taken += 1
                            progressed = True
                    if not progressed: break
                keep_idx = expanded
            else:
                # none: 不裁剪（已在 if 中）
                pass

            reps = Representatives(reps.rep_vectors[keep_idx], [reps.rep_members[i] for i in keep_idx], reps.cluster_sizes)

        # 7) 量化与打包
        q = quantize_int8(reps.rep_vectors)
        raw_bytes = int(q.q_int8.size) # int8 bytes
        scale_bytes = int(q.scales.size * 4)
        est_wire_bytes_uncompressed = raw_bytes + scale_bytes

        meta = {
            "selection_threshold": float(args.sel),
            "cluster_threshold": float(args.clu),
            "modalities_present": list(avail.keys()),
            "k_tokens": int(K),
            "dim": int(D),
            "stem": st,
            "budget_cap": int(args.budget),
            "rep_per_cluster": int(args.rep_per_cluster),
            "adapter": args.adapter,
            "adapter_depth": int(args.adapter_depth),
            "adapter_width": int(args.adapter_width),
            "adapter_heads": int(args.adapter_heads),
            "topk_per_modality": None if args.topk_per_modality is None else int(args.topk_per_modality),
            "est_uncompressed_bytes": est_wire_bytes_uncompressed,
        }
        payload = pack_payload(reps, q, meta=meta)

        # 8) 保存
        out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
        # 文件名里不能直接带 "/"，否则会变成子目录，这里做一个安全版本
        safe_st = st.replace("/", "_")
        (out_dir / f"{safe_st}_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        torch.save(reps.rep_vectors.cpu(), str(out_dir / f"{safe_st}_reps.pt"))
        (out_dir / f"{safe_st}_members.json").write_text(json.dumps(reps.rep_members, ensure_ascii=False, indent=2))

        stats = {
            "selected_per_modality": {m: len(sel.selected_indices.get(m, [])) for m in sel.selected_indices},
            "n_clusters": len(clusters),
            "n_representatives": int(reps.rep_vectors.size(0)),
            "n_representatives_before_budget": int(N_total),
            "budget_strategy": args.budget_strategy,
            "est_uncompressed_bytes": est_wire_bytes_uncompressed,
        }
        (out_dir / f"{safe_st}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

        print(f"Clusters: {len(clusters)}, Reps(before/after budget): {N_total} / {stats['n_representatives']}")
        print(f"Estimated raw bytes (int8 + scales): {est_wire_bytes_uncompressed}")
        print(f"Saved: {out_dir/f'{st}_payload.json'}")

if __name__ == "__main__":
    main()



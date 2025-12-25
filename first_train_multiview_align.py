# -*- coding: utf-8 -*-
"""
first_train_multiview_align.py
Stage-1 训练脚本 — CoDi (get_model) + Sinkhorn-OT + anti-collapse schedule
（强化版：可配置更大K、多 representative token、可选更强 AdapterHead）

这份脚本做的事：
1) 从 /home/liz0g/semantic-communication/i-Code-V3/core/models/model_module_infer.py
   正确导入 CoDi 模型，加载 /home/liz0g/semantic-communication/i-Code-V3/checkpoints 下的权重。
2) 提供 encode_text / encode_image / encode_audio / encode_vedio 四个编码函数；
   若 CoDi 的相应模块缺失，会自动 fallback 到 HuggingFace CLIP（仅文本/图像），确保能先跑起来。
3) 用 Sinkhorn-OT（熵正则最优传输）实现“软一对一匹配”，替换原来易塌缩的 Chamfer(min)。
4) 反塌缩训练调度：提高 tau、降低 qres、开启注意力熵奖励、增加多样性/非重叠/排斥项权重。
5) 新增代表子集一致性（rep-consistency）损失：对每模态选前 R 个代表 token 做一致性约束。
6) 新增 AdapterHead（identity/mlp/xattn），在对齐损失前进行更强的可学习变换，提升重建/生成可用性。
7) 训练 loop + 日志 + 快照保存。

运行示例：
  python first_train_multiview_align.py --steps 900 --lr 1e-4 --weight_decay 0.0 --save_every 300 --K 16 --rep_tokens 4 --adapter mlp --adapter_depth 3 --adapter_width 1024
  python first_train_multiview_align.py --video_id your_stem --steps 900 --K_image 24 --K_text 16 --rep_tokens 6
"""

import os, sys, glob, random, math, signal, importlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ----------------------- 路径 & 设备 -----------------------
# 自动检测 i-Code-V3 路径（优先使用项目目录中的）
ROOT = Path.cwd()
_ICODE_V3 = ROOT / "i-Code-V3"
if _ICODE_V3.exists():
    CODI_ROOT = _ICODE_V3.resolve()
    CODI_CKPT = _ICODE_V3 / "checkpoints"
else:
    # 如果项目目录中没有，使用默认路径（可通过命令行参数覆盖）
    CODI_ROOT = Path("/home/liz0g/semantic-communication/i-Code-V3").resolve()
    CODI_CKPT = Path("/home/liz0g/semantic-communication/i-Code-V3/checkpoints").resolve()

if str(CODI_ROOT) not in sys.path:
    sys.path.insert(0, str(CODI_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径（可通过命令行参数覆盖）
DATA = ROOT / "data"
DVID, DIMG, DAUD, DTXT = DATA/"vedio", DATA/"image", DATA/"audio", DATA/"text"

# ----------------------- 加载 CoDi -----------------------
_CODI_READY = False
_codi = None
net = None

def _load_codi(ckpt_dir: Path):
    """第一阶段训练只需加载编码器权重，不需要diffuser权重"""
    from core.models.model_module_infer import model_module as CoDiModule
    return CoDiModule(
        data_dir=str(ckpt_dir),
        pth=["CoDi_encoders.pth"],  # 只加载编码器，不加载diffuser（音频/文本/视频生成器）
        fp16=False
    )

def _ensure_codi():
    """延迟加载 CoDi 模型（在参数解析后调用）"""
    global _CODI_READY, _codi, net
    if _CODI_READY:
        return
    print("🔍 加载 CoDi 模型…")
    _codi = _load_codi(CODI_CKPT)
    # 有些版本返回对象含 .net，有些本身就是 net
    net = _codi.net if hasattr(_codi, "net") else _codi
    for p in net.parameters():  # 冻结底座编码器
        p.requires_grad_(False)
    net = net.to(device).eval()
    print(f"✅ 已加载 CoDi，权重目录: {CODI_CKPT}")
    _CODI_READY = True

# ----------------------- HF CLIP 兜底（仅文本/图像） -----------------------
_HF_READY = False
def _ensure_hf():
    global _HF_READY, _tok, _txt, _vis, _imgp
    if _HF_READY:
        return
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
    HF_MODEL = os.environ.get("HF_CLIP", "openai/clip-vit-base-patch32")
    _tok  = CLIPTokenizer.from_pretrained(HF_MODEL)
    _txt  = CLIPTextModel.from_pretrained(HF_MODEL).to(device).eval()
    _imgp = CLIPImageProcessor.from_pretrained(HF_MODEL)
    _vis  = CLIPVisionModel.from_pretrained(HF_MODEL).to(device).eval()
    _HF_READY = True
    print("ℹ️ 使用 HF CLIP 作为兜底（text/image）")

# ----------------------- 编码函数 -----------------------
@torch.no_grad()
def encode_text(list_of_str: List[str]) -> torch.Tensor:
    # 优先使用 CoDi 的 CLIP 文本编码
    _ensure_codi()
    clip = getattr(net, "clip", None)
    if clip is not None and hasattr(clip, "encode_text_noproj"):
        out = clip.encode_text_noproj(list_of_str)
        z_seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        return z_seq.float().detach().to(device)  # (B, L, D)
    # 兜底：HF
    _ensure_hf()
    toks = _tok(text=list_of_str, return_tensors="pt", padding=True, truncation=True).to(device)
    out  = _txt(**toks)
    return out.last_hidden_state.float().detach()  # (B, L, D)

@torch.no_grad()
def encode_image(paths: List[str]) -> torch.Tensor:
    # CoDi 的视觉编码
    _ensure_codi()
    clip = getattr(net, "clip", None)
    if clip is not None and hasattr(clip, "encode_vision_noproj"):
        tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        xs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            xs.append((tfm(img).unsqueeze(0)*2-1))
        x = torch.cat(xs, 0).to(device)
        out = clip.encode_vision_noproj(x)
        f_seq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        return f_seq.float().detach()  # (N, L, D)
    # 兜底：HF
    _ensure_hf()
    imgs   = [Image.open(p).convert("RGB") for p in paths]
    inputs = _imgp(images=imgs, return_tensors="pt").to(device)
    out    = _vis(**inputs)
    return out.last_hidden_state.float().detach()  # (N, L, D)

@torch.no_grad()
def encode_audio(paths: List[str], seconds: float = 8.0) -> Optional[torch.Tensor]:
    # 仅当 CoDi 包含 clap 时可用；否则返回 None
    _ensure_codi()
    clap = getattr(net, "clap", None)
    if clap is None:
        print("(音频后端缺失，跳过 audio)")
        return None
    try:
        import torchaudio
    except Exception as e:
        print("(未安装 torchaudio，跳过 audio)", e)
        return None
    target_sr = getattr(clap, "sample_rate", 48000)
    if isinstance(paths, (str, Path)):
        paths = [paths]
    wavs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        w, sr = torchaudio.load(str(p))
        w = w.mean(0, keepdim=True)  # mono
        if sr != target_sr:
            w = torchaudio.functional.resample(w, sr, target_sr)
        max_len = int(seconds * target_sr)
        if w.size(1) >= max_len:
            st = random.randint(0, w.size(1) - max_len)
            w = w[:, st:st+max_len]
        else:
            w = torch.nn.functional.pad(w, (0, max_len - w.size(1)))
        wavs.append(w)
    if not wavs:
        return None
    batch = torch.cat(wavs, 0).to(device)
    if hasattr(clap, "encode_audio_noproj"):
        out = clap.encode_audio_noproj(batch)
        z   = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    else:
        z = clap(batch)
    if z.ndim == 2:
        z = z.unsqueeze(1)  # (B, 1, D)
    return z.float().detach()

@torch.no_grad()
def encode_vedio(paths, max_frames: int = 8) -> Optional[torch.Tensor]:
    """简单的视频编码：抽帧 -> 逐帧用视觉编码 -> 帧均值（保留 patch 维）"""
    try:
        import cv2
    except Exception as e:
        print("(未安装 opencv-python，跳过 video)", e)
        return None
    if isinstance(paths, (str, Path)):
        paths = [paths]
    feats = []
    for p in paths:
        p = str(p)
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs  = list(range(total)) if total <= max_frames else np.linspace(0, total-1, max_frames, dtype=int).tolist()
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            continue
        # 用和 encode_image 一样的流程做一批 encode，然后在帧维做均值
        _ensure_codi()
        clip = getattr(net, "clip", None)
        if clip is not None and hasattr(clip, "encode_vision_noproj"):
            tfm   = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
            batch = torch.cat([(tfm(Image.fromarray(f)).unsqueeze(0)*2-1) for f in frames], 0).to(device)
            out   = clip.encode_vision_noproj(batch)
            fseq  = out.last_hidden_state.float().detach()  # (F, L, D)
            feats.append(fseq)
        else:
            _ensure_hf()
            imgs   = [Image.fromarray(f) for f in frames]
            inputs = _imgp(images=imgs, return_tensors="pt").to(device)
            out    = _vis(**inputs)
            fseq   = out.last_hidden_state.float().detach()  # (F, L, D)
            feats.append(fseq)
    if not feats:
        return None
    return torch.cat(feats, 0)  # (B=1, L, D)

# ----------------------- 数据工具 -----------------------
def list_stems() -> List[str]:
    vids = {Path(p).stem for p in glob.glob(str(DVID/"*.mp4"))}
    aus  = {Path(p).stem for p in glob.glob(str(DAUD/"*.wav"))}
    txts = {Path(p).stem for p in glob.glob(str(DTXT/"*.txt"))}
    srt  = {Path(p).stem for p in glob.glob(str(DTXT/"*.srt"))}
    imgs = set(Path(p).name.split("_")[0] for p in glob.glob(str(DIMG/"*_*.jpg")))
    return sorted(vids | aus | txts | srt | imgs)

def imgs_for(st: str, take: int = 8) -> List[str]:
    xs = sorted(glob.glob(str(DIMG/f"{st}_*.jpg")))
    if len(xs) > take:
        xs = random.sample(xs, take)
    return xs

def text_for(st: str) -> str:
    p = DTXT / f"{st}.txt"
    if p.exists():
        try:
            return p.read_text(errors="ignore").strip()
        except Exception:
            pass
    ps = DTXT / f"{st}.srt"
    if ps.exists():
        raw = ps.read_text(errors="ignore")
        lines = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln or ln.isdigit() or "-->" in ln:
                continue
            lines.append(ln)
        return " ".join(lines).strip()
    return ""

def has_modalities(st: str) -> int:
    c = 0
    if (DVID/f"{st}.mp4").exists(): c += 1
    if (DAUD/f"{st}.wav").exists(): c += 1
    if (DTXT/f"{st}.txt").exists() or (DTXT/f"{st}.srt").exists(): c += 1
    if len(imgs_for(st)) > 0: c += 1
    return c

# ----------------------- Adapter 头（更强表示） -----------------------
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
    """
    轻量 cross-attn：把同一视频的多模态 token 做一次门控融合（不依赖生成端）
    输入 (K,D)，返回 (K,D)。可在 Stage-1 里作为更强的“融合头”。
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln   = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # x_list: [x_t, x_i, x_a, x_v] 中的若干非空 (K,D)
        valid = [x for x in x_list if x is not None]
        if len(valid) <= 1:  # 单模态时直接返回
            return x_list
        ctx = torch.cat(valid, dim=0).unsqueeze(0)  # (1, sumK, D)
        out_list = []
        for x in x_list:
            if x is None:
                out_list.append(None)
                continue
            q = self.ln(x).unsqueeze(0)  # (1,K,D)
            attn, _ = self.attn(q, ctx, ctx, need_weights=False)
            y = x + attn.squeeze(0)
            y = y + self.ffn(y)
            out_list.append(y)
        return out_list

class AdapterHead(nn.Module):
    """
    可选 adapter:
      - identity: 不做变换
      - mlp: 残差 MLP（可更深/更宽）
      - xattn: 轻量 cross-attn 融合（返回仍按模态拆开）
    """
    def __init__(self, dim: int, kind: str = "mlp", depth: int = 2, width: int = 1024, heads: int = 4):
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
            def go(x):
                return None if x is None else self.body(x)
            return go(xt), go(xi), go(xa), go(xv)

# ----------------------- 抽 K 个语义 token 的提取器 -----------------------
class SemanticExtractor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512, K: int = 8, q_res_w: float = 0.6):
        super().__init__()
        self.K = K
        self.query = nn.Parameter(torch.randn(K, out_dim))
        nn.init.orthogonal_(self.query)
        self.key   = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.norm  = nn.LayerNorm(out_dim)
        self.ffn   = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.attn_tau = 1.0
        self.q_res_w  = q_res_w
        self.pool_stride = 1
        self.last_attn_w = None   # (1, K, S)
        self.last_attn_entropy = None

    # —— 原 softmax 聚合逻辑被更强的 Gumbel-Sinkhorn + 等份分配替换，见下方 _gs_forward ——


# ----------------------- 损失 & 指标 -----------------------
def _l2n(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1)

def diversity_loss(Z: torch.Tensor) -> torch.Tensor:
    Z = _l2n(Z)
    G = Z @ Z.t()
    eye = torch.eye(G.size(0), device=G.device)
    return ((G - eye) ** 2).mean()

def query_ortho_loss(mod: SemanticExtractor) -> torch.Tensor:
    q = _l2n(mod.query)
    G = q @ q.t()
    eye = torch.eye(G.size(0), device=G.device)
    return ((G - eye) ** 2).mean()

def attention_entropy_penalty(extractors: List[Optional[SemanticExtractor]]) -> Optional[torch.Tensor]:
    ents = []
    for m in extractors:
        if m is None or getattr(m, "last_attn_entropy", None) is None:
            continue
        ents.append(m.last_attn_entropy)
    if ents:
        return torch.stack(ents).mean()
    return None

def attention_nonoverlap_loss(extractors: List[Optional[SemanticExtractor]]) -> Optional[torch.Tensor]:
    losses = []
    for m in extractors:
        if m is None or getattr(m, "last_attn_w", None) is None:
            continue
        W = m.last_attn_w  # (1, K, S)
        G = W @ W.transpose(1, 2)  # (1, K, K)
        eye = torch.eye(G.size(-1), device=G.device).unsqueeze(0)
        off = G * (1.0 - eye)
        losses.append((off ** 2).mean())
    if losses:
        return torch.stack(losses).mean()
    return None

def direct_dup_penalty(extractors: List[Optional[SemanticExtractor]]) -> Optional[torch.Tensor]:
    """直接惩罚 dup：鼓励每个 K-token 选择不同的输入 token"""
    penalties = []
    for m in extractors:
        if m is None or getattr(m, "last_attn_w", None) is None:
            continue
        W = m.last_attn_w[0]  # (K, S)，行已归一化
        # 计算每个输入 token 被多少 K-token 选择（列和）
        col_sum = W.sum(dim=0)  # (S,)
        # 惩罚：如果某个输入 token 被多个 K-token 选择，增加惩罚
        # 理想情况：每个输入 token 最多被 1/K 的 K-token 选择（当完全均匀时）
        ideal_per_col = 1.0 / W.size(0)  # 1/K
        # 惩罚超过理想值的情况
        excess = (col_sum - ideal_per_col).clamp(min=0)
        penalties.append(excess.sum())
    if penalties:
        return torch.stack(penalties).mean()
    return None

# --- Sinkhorn-OT 软一对一匹配 ---
def _pairwise_cost(A: torch.Tensor, B: torch.Tensor, metric: str = "cos") -> torch.Tensor:
    A_n = _l2n(A); B_n = _l2n(B)
    if metric == "cos":
        return 1.0 - torch.einsum("id,jd->ij", A_n, B_n)
    elif metric == "l2":
        return torch.cdist(A_n, B_n, p=2)
    else:
        raise ValueError("unknown metric")

@torch.no_grad()
def _sinkhorn(logK: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    logP = logK.clone()
    for _ in range(n_iter):
        logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)
        logP = logP - torch.logsumexp(logP, dim=0, keepdim=True)
    return logP

def sinkhorn_ot_loss(A: torch.Tensor, B: torch.Tensor, eps: float = 0.03, n_iter: int = 30,
                     metric: str = "cos", stopgrad: bool = True) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2 and A.size(0) == B.size(0), "A,B must be (K,D) with same K"
    def _ot(a, b):
        C = _pairwise_cost(a, b, metric)     # (K,K)
        logK = -C / eps
        logP = _sinkhorn(logK, n_iter=n_iter)
        P = logP.exp()                        # (K,K) 近似双随机
        return (P * C).sum() / A.size(0)
    if stopgrad:
        return 0.5 * (_ot(A, B.detach()) + _ot(B, A.detach()))
    else:
        return _ot(A, B)

# === Anti-dup: Gumbel-Sinkhorn + SwAV balance (top methods), and DPP regularizer ===
import math

def _sinkhorn_balanced_log(logM, r, c, n_iter=10):
    # log-domain Sinkhorn，行和→r，列和→c
    logP = logM
    for _ in range(n_iter):
        logP = logP - torch.logsumexp(logP, dim=1, keepdim=True) + torch.log(r.clamp_min(1e-8))[:, None]
        logP = logP - torch.logsumexp(logP, dim=0, keepdim=True) + torch.log(c.clamp_min(1e-8))[None, :]
    return logP

def gumbel_sinkhorn_balanced(logits, tau=0.7, n_iter=10, r=None, c=None, training=True):
    # logits: (S', K)
    if training:
        g = -(torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-8))))
    else:
        g = 0.0
    logM = (logits + g) / max(tau, 1e-6)
    S, K = logM.shape
    if r is None: r = torch.full((S,), K/float(S), device=logM.device, dtype=logM.dtype)  # 每行容量≈K/S
    if c is None: c = torch.ones(K, device=logM.device, dtype=logM.dtype)                 # 每列总量=1（每个slot被用一次）
    logP = _sinkhorn_balanced_log(logM, r, c, n_iter=n_iter)
    return logP.exp()  # (S',K) 近似双随机（满足目标边缘）

def dpp_penalty_from_extractors(extrs, eps=1e-3):
    # 对每个提取器的注意力 W(K,S) 做 DPP：L = W W^T，loss = -logdet(L+eps I)
    losses = []
    for m in extrs:
        W = getattr(m, "last_attn_w", None)
        if W is None:
            continue
        W = W[0]  # (K,S)，行已归一化
        L = W @ W.t() + eps * torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        sign, logdet = torch.slogdet(L)
        if sign > 0:
            losses.append(-logdet)
    if not losses:
        return None
    return sum(losses) / len(losses)

def _gs_forward(self, seq: torch.Tensor) -> torch.Tensor:
    # —— 将原来的 softmax 聚合替换为：候选位置选择 + Gumbel-Sinkhorn 平衡分配 —— #
    if seq.dim() == 3:
        seq = seq.reshape(seq.shape[0]*seq.shape[1], seq.shape[-1])
    elif seq.dim() != 2:
        raise ValueError("seq must be (L,D) or (1,L,D) or (T,L,D)")
    if getattr(self, "pool_stride", 1) and self.pool_stride > 1:
        seq = seq[:: self.pool_stride]

    Q = self.query
    K = self.key(seq)     # (S,D)
    V = self.value(seq)   # (S,D)

    K_n = torch.nn.functional.normalize(K, dim=-1)
    Q_n = torch.nn.functional.normalize(Q, dim=-1)
    logits = K_n @ Q_n.t()    # (S,K)

    S, Kslots = logits.shape
    Sprime = min(S, max(Kslots, int(math.ceil(1.5 * Kslots))))   # 取 ~1.5K 个候选位置
    with torch.no_grad():
        scores, _ = logits.max(dim=1)
        idx = torch.topk(scores, k=Sprime, sorted=False).indices

    logits_sel = logits.index_select(0, idx)    # (S',K)
    r = torch.full((Sprime,), Kslots/float(Sprime), device=logits_sel.device, dtype=logits_sel.dtype)
    c = torch.ones(Kslots, device=logits_sel.device, dtype=logits_sel.dtype)

    P = gumbel_sinkhorn_balanced(
        logits_sel,
        tau=getattr(self, "attn_tau", 1.0), n_iter=10, r=r, c=c, training=self.training
    )   # (S',K)

    V_sel = V.index_select(0, idx)              # (S',D)
    Z = P.t() @ V_sel                           # (K,D)

    if getattr(self, "q_res_w", 0.0) > 0:
        Z = Z + self.q_res_w * self.query
    Z = self.ffn(self.norm(Z))

    # 记录注意力（K,S），行归一化后用于 dup/熵/DPP
    W_full = logits.new_zeros(Kslots, S)
    W_full[:, idx] = P.t()
    Wn = W_full / (W_full.sum(dim=1, keepdim=True) + 1e-8)
    self.last_attn_w = Wn.unsqueeze(0).detach()
    ent = -(Wn * (Wn.clamp_min(1e-8).log())).sum(dim=1).mean()
    self.last_attn_entropy = ent.detach()
    return Z

# 替换聚合策略：启用 Gumbel-Sinkhorn + 等份分配
SemanticExtractor.forward = _gs_forward

# ----------------------- K/R/Extractor 初始化 -----------------------
def _probe_dim_text() -> Optional[int]:
    try:
        z = encode_text(["probe"])
        return int(z.shape[-1])
    except Exception:
        return None

def _probe_dim_image() -> Optional[int]:
    try:
        any_img = next(iter(glob.glob(str(DIMG/"*_*.jpg"))), None)
        if not any_img: return None
        z = encode_image([any_img])
        return int(z.shape[-1])
    except Exception:
        return None

def _probe_dim_audio() -> Optional[int]:
    try:
        any_aud = next(iter(glob.glob(str(DAUD/"*.wav"))), None)
        if not any_aud: return None
        z = encode_audio([any_aud])
        return None if z is None else int(z.shape[-1])
    except Exception:
        return None

def _probe_dim_vedio() -> Optional[int]:
    try:
        any_vid = next(iter(glob.glob(str(DVID/"*.mp4"))), None)
        if not any_vid: return None
        z = encode_vedio([any_vid], max_frames=4)
        return None if z is None else int(z.shape[-1])
    except Exception:
        return None

def build_extractors(K_text, K_image, K_audio, K_vedio, out_dim=512):
    DTXT_DIM, DIMG_DIM, DAUD_DIM, DVID_DIM = _probe_dim_text(), _probe_dim_image(), _probe_dim_audio(), _probe_dim_vedio()
    EXTRACTOR_T = SemanticExtractor(DTXT_DIM, out_dim, K_text, q_res_w=0.10).to(device) if DTXT_DIM else None
    EXTRACTOR_I = SemanticExtractor(DIMG_DIM, out_dim, K_image, q_res_w=0.10).to(device) if DIMG_DIM else None
    EXTRACTOR_A = SemanticExtractor(DAUD_DIM, out_dim, K_audio, q_res_w=0.10).to(device) if DAUD_DIM else None
    EXTRACTOR_V = SemanticExtractor(DVID_DIM, out_dim, K_vedio, q_res_w=0.10).to(device) if DVID_DIM else None
    return EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V

# ----------------------- 编码一次 + 抽 token -----------------------
def _encode_vedio_once(st: str, take_img: int = 4, seconds: float = 8.0):
    t_seq = i_seq = a_seq = v_seq = None
    txt = text_for(st)
    if txt:
        t_seq = encode_text([txt])            # (1, L, D)
    ims = imgs_for(st, take=take_img)
    if ims:
        zi = encode_image(ims)                # (N, L, D)
        i_seq = zi
    wa = DAUD / f"{st}.wav"
    if wa.exists():
        a_seq = encode_audio([wa], seconds=seconds)
        if a_seq is None:
            print(f"!! 音频编码失败，跳过该模态: {wa}"); a_seq = None
    wv = DVID / f"{st}.mp4"
    if wv.exists():
        v_seq = encode_vedio([wv], max_frames=take_img)  # (1, L, D)
    return t_seq, i_seq, a_seq, v_seq

def _extract_K_tokens(t_seq, i_seq, a_seq, v_seq, EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V):
    out_t = EXTRACTOR_T(t_seq.squeeze(0)) if (t_seq is not None and EXTRACTOR_T is not None) else None
    out_i = EXTRACTOR_I(i_seq.squeeze(0)) if (i_seq is not None and EXTRACTOR_I is not None) else None
    out_a = EXTRACTOR_A(a_seq.squeeze(0)) if (a_seq is not None and EXTRACTOR_A is not None) else None
    out_v = EXTRACTOR_V(v_seq.squeeze(0)) if (v_seq is not None and EXTRACTOR_V is not None) else None
    return out_t, out_i, out_a, out_v   # (K,D) each

# ----------------------- 代表 token 选择 -----------------------
def top_rep_indices_from_attn(extractor: Optional[SemanticExtractor], R: int) -> Optional[torch.Tensor]:
    """
    从记录的注意力 W(K,S) 里，挑选“代表性最强”的 R 个 K-slot。
    策略：对每个 token（行）取其最大注意力权重，按降序选前 R。
    返回 shape (R,) 的索引。
    """
    if extractor is None or getattr(extractor, "last_attn_w", None) is None:
        return None
    W = extractor.last_attn_w[0]  # (K,S)
    score = W.max(dim=1).values   # (K,)
    R = min(R, W.size(0))
    idx = torch.topk(score, k=R, largest=True).indices
    return idx

def gather_rep_tokens(Z: Optional[torch.Tensor], idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if Z is None or idx is None:
        return None
    return Z.index_select(0, idx)

# ----------------------- 聚合损失 -----------------------
def one_vedio_loss(st, it, cfg, EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V, adapter_head: AdapterHead):
    # 两次随机视图/裁剪
    t1, i1, a1, v1 = _encode_vedio_once(st, cfg["img_take"], cfg["aud_seconds"])
    zt1, zi1, za1, zv1 = _extract_K_tokens(t1, i1, a1, v1, EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V)
    t2, i2, a2, v2 = _encode_vedio_once(st, cfg["img_take"], max(4.0, cfg["aud_seconds"] - 2.0))
    zt2, zi2, za2, zv2 = _extract_K_tokens(t2, i2, a2, v2, EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V)

    # Adapter （更强表示头）
    zt1, zi1, za1, zv1 = adapter_head(zt1, zi1, za1, zv1)
    zt2, zi2, za2, zv2 = adapter_head(zt2, zi2, za2, zv2)

    total, terms = 0.0, {}
    # 跨模态对齐（Sinkhorn-OT）
    pairs = []
    if zt1 is not None and zi1 is not None: pairs.append((zt1, zi1))
    if zt1 is not None and za1 is not None: pairs.append((zt1, za1))
    if zt1 is not None and zv1 is not None: pairs.append((zt1, zv1))
    if zi1 is not None and za1 is not None: pairs.append((zi1, za1))
    if zi1 is not None and zv1 is not None and it > cfg["skip_iv_until"]: pairs.append((zi1, zv1))
    if za1 is not None and zv1 is not None: pairs.append((za1, zv1))
    if pairs:
        xmodal = sum(sinkhorn_ot_loss(a, b) for a, b in pairs) / len(pairs)
        terms["xmodal"] = xmodal
        total += cfg["lambda_xmodal"] * xmodal

    # 跨视图一致（view consistency）
    vloss = []
    if zt1 is not None and zt2 is not None: vloss.append(sinkhorn_ot_loss(zt1, zt2))
    if zi1 is not None and zi2 is not None: vloss.append(sinkhorn_ot_loss(zi1, zi2))
    if za1 is not None and za2 is not None: vloss.append(sinkhorn_ot_loss(za1, za2))
    if zv1 is not None and zv2 is not None: vloss.append(sinkhorn_ot_loss(zv1, zv2))
    if vloss:
        vmean = sum(vloss) / len(vloss)
        terms["view"] = vmean
        total += cfg["lambda_view"] * vmean

    # 多样性（intra-K decorrelation）
    dloss = []
    for z in [zt1, zi1, za1, zv1, zt2, zi2, za2, zv2]:
        if z is not None:
            dloss.append(diversity_loss(z))
    if dloss:
        dmean = sum(dloss) / len(dloss)
        terms["div"] = dmean
        total += cfg["lambda_div"] * dmean

    # 注意力熵（奖励 or 惩罚）
    attn_pen = attention_entropy_penalty([EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V])
    if attn_pen is not None:
        terms["attn"] = attn_pen
        total += (-cfg["lambda_attn"] if cfg["attn_reward"] else cfg["lambda_attn"]) * attn_pen

    # 注意力非重叠
    ano_pen = attention_nonoverlap_loss([EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V])
    if ano_pen is not None:
        terms["ano"] = ano_pen
        total += cfg["lambda_ano"] * ano_pen

    # Query 正交
    qlist = []
    for M in (EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V):
        if M is not None:
            qlist.append(query_ortho_loss(M))
    if qlist:
        qloss = sum(qlist) / len(qlist)
        terms["q"] = qloss
        total += cfg["lambda_q"] * qloss

    # 批内排斥
    def batch_token_repel(Zs):
        Zs = [z for z in Zs if z is not None]
        if not Zs: return None
        Z = torch.cat(Zs, dim=0)
        Z = _l2n(Z)
        sim = Z @ Z.t()
        eye = torch.eye(sim.size(0), device=sim.device)
        diff = (sim - eye).clamp(min=0)
        return diff.mean()

    rep = batch_token_repel([zt1, zi1, za1, zv1, zt2, zi2, za2, zv2])
    if rep is not None:
        terms["rep"] = rep
        total += cfg["lambda_repel"] * rep

    # 代表子集一致性（R 个 representative token，缓解“只选一个代表导致结果不清晰”）
    if cfg["rep_tokens"] > 0:
        Rt = top_rep_indices_from_attn(EXTRACTOR_T, cfg["rep_tokens"])
        Ri = top_rep_indices_from_attn(EXTRACTOR_I, cfg["rep_tokens"])
        Ra = top_rep_indices_from_attn(EXTRACTOR_A, cfg["rep_tokens"])
        Rv = top_rep_indices_from_attn(EXTRACTOR_V, cfg["rep_tokens"])

        zt1_r = gather_rep_tokens(zt1, Rt); zi1_r = gather_rep_tokens(zi1, Ri)
        za1_r = gather_rep_tokens(za1, Ra); zv1_r = gather_rep_tokens(zv1, Rv)
        zt2_r = gather_rep_tokens(zt2, Rt); zi2_r = gather_rep_tokens(zi2, Ri)
        za2_r = gather_rep_tokens(za2, Ra); zv2_r = gather_rep_tokens(zv2, Rv)

        # 代表子集跨视图一致
        repv = []
        if zt1_r is not None and zt2_r is not None and zt1_r.size(0) == zt2_r.size(0):
            repv.append(sinkhorn_ot_loss(zt1_r, zt2_r))
        if zi1_r is not None and zi2_r is not None and zi1_r.size(0) == zi2_r.size(0):
            repv.append(sinkhorn_ot_loss(zi1_r, zi2_r))
        if za1_r is not None and za2_r is not None and za1_r.size(0) == za2_r.size(0):
            repv.append(sinkhorn_ot_loss(za1_r, za2_r))
        if zv1_r is not None and zv2_r is not None and zv1_r.size(0) == zv2_r.size(0):
            repv.append(sinkhorn_ot_loss(zv1_r, zv2_r))

        # 代表子集跨模态一致
        repp = []
        for a, b in [(zt1_r, zi1_r), (zt1_r, za1_r), (zt1_r, zv1_r),
                     (zi1_r, za1_r), (zi1_r, zv1_r), (za1_r, zv1_r)]:
            if a is not None and b is not None and a.size(0) == b.size(0):
                repp.append(sinkhorn_ot_loss(a, b))

        if repv:
            rv = sum(repv) / len(repv)
            terms["rep_view"] = rv
            total += cfg["lambda_rep_view"] * rv
        if repp:
            rp = sum(repp) / len(repp)
            terms["rep_xmodal"] = rp
            total += cfg["lambda_rep_xmodal"] * rp

    return total, terms

# ----------------------- 训练调度 & loop -----------------------
def dup_rate_of(mod: Optional[SemanticExtractor]) -> float:
    if not mod or getattr(mod, "last_attn_w", None) is None:
        return 0.0
    top = mod.last_attn_w[0].argmax(-1)  # (K,)
    return 1 - top.unique().numel() / top.numel()

def phase(it: int) -> Dict[str, float]:
    # anti-collapse schedule（增强早期多样性惩罚，防止dup过高）
    if it <= 400:
        return dict(tau=0.8, qres=0.50, ps=1,  # 降低tau，让注意力更均匀
                    lam_x=0.20, lam_div=0.40, lam_rep=1.50, lam_ano=1.00,  # 增强多样性损失
                    lam_attn=0.20, attn_reward=True)
    elif it <= 700:
        return dict(tau=1.5, qres=0.20, ps=2, lam_x=0.30, lam_div=0.35, lam_rep=1.20, lam_ano=0.90, lam_attn=0.15, attn_reward=True)
    elif it <= 900:
        return dict(tau=1.2, qres=0.10, ps=3, lam_x=0.50, lam_div=0.30, lam_rep=1.00, lam_ano=0.70, lam_attn=0.12, attn_reward=True)
    else:
        return dict(tau=0.80, qres=0.10, ps=4,
                    lam_x=0.60, lam_div=0.25, lam_rep=0.80, lam_ano=0.50,
                    lam_attn=0.10, attn_reward=True)

def save_snapshot(path: Path, EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V, Kcfg, OUT_DIM):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "EXTRACTOR_T": (EXTRACTOR_T.state_dict() if EXTRACTOR_T else None),
        "EXTRACTOR_I": (EXTRACTOR_I.state_dict() if EXTRACTOR_I else None),
        "EXTRACTOR_A": (EXTRACTOR_A.state_dict() if EXTRACTOR_A else None),
        "EXTRACTOR_V": (EXTRACTOR_V.state_dict() if EXTRACTOR_V else None),
        "K_TOKENS": Kcfg,
        "OUT_DIM": OUT_DIM,
    }, str(path))
    print("💾 snapshot saved:", path)

def train_single_vedio(st: str, args):
    # === 构建 Extractors（可配置 K） ===
    K_text  = args.K_text  if args.K_text  is not None else args.K
    K_image = args.K_image if args.K_image is not None else args.K
    K_audio = args.K_audio if args.K_audio is not None else args.K
    K_vedio = args.K_vedio if args.K_vedio is not None else args.K

    OUT_DIM = args.out_dim
    EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V = build_extractors(K_text, K_image, K_audio, K_vedio, OUT_DIM)

    params = []
    for M in (EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V):
        if M: params += list(M.parameters())
    assert params, "没有任何 extractor 可训练参数（至少需要一种模态的提取器被初始化）"

    # === AdapterHead（更强表示；默认 mlp） ===
    adapter = AdapterHead(dim=OUT_DIM,
                          kind=args.adapter,
                          depth=args.adapter_depth,
                          width=args.adapter_width,
                          heads=args.adapter_heads).to(device)

    params += list(adapter.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    CKP = Path("snapshots"); CKP.mkdir(parents=True, exist_ok=True)

    def _handle_sigint(sig, frame):
        print("\n[CTRL-C] 保存快照后退出…")
        save_snapshot(Path("snapshots")/f"stage1_it{it}.pth",
                      EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V,
                      dict(text=K_text, image=K_image, audio=K_audio, vedio=K_vedio),
                      OUT_DIM)
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    for it in range(1, args.steps + 1):
        sched = phase(it)
        for M in (EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V):
            if M:
                M.attn_tau = sched["tau"]
                M.q_res_w  = sched["qres"]
                M.pool_stride = sched["ps"]

        dup = max(dup_rate_of(EXTRACTOR_T), dup_rate_of(EXTRACTOR_I),
                  dup_rate_of(EXTRACTOR_A), dup_rate_of(EXTRACTOR_V))
        fuse = dict(**sched)
        dpp_weight = 1e-3  # 默认 DPP 权重
        
        if dup > 0.90:  # 超极端情况：dup > 0.90（需要最激进的策略）
            # 超级激进的惩罚
            fuse["lam_div"] *= 50.0  # 大幅增加多样性损失
            fuse["lam_rep"] *= 10.0
            fuse["lam_ano"] *= 10.0
            fuse["lam_x"]   *= 0.1   # 暂时降低跨模态对齐，专注多样性
            fuse["lam_attn"] *= 5.0
            fuse["attn_reward"] = True
            # 极低tau，强制注意力均匀分布
            fuse["tau"] = 0.2
            dpp_weight = 1e-2  # 大幅增加 DPP 惩罚
            for M in (EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V):
                if M:
                    M.attn_tau = fuse["tau"]
            if it % 50 == 0:  # 每50轮重新初始化 query（如果 dup 持续很高）
                print(f"⚠️  超极端 dup={dup:.3f}，考虑重新初始化 query...")
        elif dup > 0.80:  # 极端情况：dup > 0.80
            # 非常激进的惩罚
            fuse["lam_div"] *= 20.0  # 从10.0提升到20.0
            fuse["lam_rep"] *= 8.0   # 从5.0提升到8.0
            fuse["lam_ano"] *= 8.0
            fuse["lam_x"]   *= 0.2   # 进一步降低
            fuse["lam_attn"] *= 4.0
            fuse["attn_reward"] = True
            # 降低tau，让注意力更均匀
            fuse["tau"] = 0.3  # 从0.5降到0.3
            dpp_weight = 5e-3  # 增加 DPP 惩罚
            for M in (EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V):
                if M:
                    M.attn_tau = fuse["tau"]
        elif dup > 0.50:  # 严重情况：0.50 < dup <= 0.80
            fuse["lam_div"] *= 5.0
            fuse["lam_rep"] *= 3.0
            fuse["lam_ano"] *= 3.0
            fuse["lam_x"]   *= 0.5
            fuse["lam_attn"] *= 2.0
            fuse["attn_reward"] = True
            fuse["tau"] = min(fuse["tau"], 0.8)
            dpp_weight = 2e-3
            for M in (EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V):
                if M:
                    M.attn_tau = fuse["tau"]
        elif dup > 0.30:  # 一般情况：0.30 < dup <= 0.50
            fuse["lam_div"] *= 2.0
            fuse["lam_rep"] *= 1.5
            fuse["lam_ano"] *= 1.5
            fuse["lam_x"]   *= 0.7
            fuse["attn_reward"] = True
            dpp_weight = 1.5e-3

        cfg = dict(
            img_take=args.img_take,
            aud_seconds=args.aud_seconds,
            skip_iv_until=args.skip_iv_until,
            lambda_xmodal=fuse["lam_x"],
            lambda_view=args.lambda_view,
            lambda_div=fuse["lam_div"],
            lambda_attn=fuse["lam_attn"],
            attn_reward=fuse["attn_reward"],
            lambda_q=args.lambda_q,
            lambda_ano=fuse["lam_ano"],
            lambda_repel=fuse["lam_rep"],
            rep_tokens=args.rep_tokens,
            lambda_rep_view=args.lambda_rep_view,
            lambda_rep_xmodal=args.lambda_rep_xmodal,
        )

        loss, terms = one_vedio_loss(st, it, cfg, EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V, adapter)
        
        # DPP 惩罚（动态权重）
        dpp = dpp_penalty_from_extractors([EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V], eps=1e-3)
        if dpp is not None:
            loss = loss + dpp_weight * dpp
            terms["dpp"] = dpp
        
        # 直接 dup 惩罚（当 dup 很高时）
        if dup > 0.70:
            dup_pen = direct_dup_penalty([EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V])
            if dup_pen is not None:
                dup_lambda = (dup - 0.70) * 10.0  # dup 越高，惩罚越大
                loss = loss + dup_lambda * dup_pen
                terms["dup_pen"] = dup_pen
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); optim.step()

        def grad_norm(m):
            if not m: return 0.0
            g = 0.0
            for p in m.parameters():
                if p.grad is not None:
                    g += p.grad.data.norm(2).item()
            return g

        if it % 50 == 0:
            T_g = grad_norm(EXTRACTOR_T); I_g = grad_norm(EXTRACTOR_I)
            A_g = grad_norm(EXTRACTOR_A); V_g = grad_norm(EXTRACTOR_V)
            dup_pen_str = f" dup_pen={terms.get('dup_pen',0):.3f}" if 'dup_pen' in terms else ""
            print((f"[{it}/{args.steps}] loss={loss.item():.4f} "
                   f"x={terms.get('xmodal',0):.3f} v={terms.get('view',0):.3f} "
                   f"d={terms.get('div',0):.3f} attn={terms.get('attn',0):.3f} "
                   f"a={terms.get('ano',0):.3f} r={terms.get('rep',0):.3f} q={terms.get('q',0):.3f} "
                   f"rv={terms.get('rep_view',0):.3f} rx={terms.get('rep_xmodal',0):.3f} "
                   f"dpp={terms.get('dpp',0):.3f}{dup_pen_str} "
                   f"grad|T={T_g:.3e} grad|I={I_g:.3e} grad|A={A_g:.3e} grad|V={V_g:.3e} "
                   f"tau={fuse['tau']:.2f} qres={fuse['qres']:.2f} ps={fuse['ps']} dup={dup:.2f} "
                   f"attn_reward={'on' if fuse['attn_reward'] else 'off'} "
                   f"K[T/I/A/V]={K_text}/{K_image}/{K_audio}/{K_vedio} "
                   f"R={args.rep_tokens} adapter={args.adapter}"))

        if it % args.save_every == 0:
            save_snapshot(CKP/f"step_{it:04d}.pth",
                          EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V,
                          dict(text=K_text, image=K_image, audio=K_audio, vedio=K_vedio),
                          OUT_DIM)

    save_snapshot(CKP/"stage1_final.pth",
                  EXTRACTOR_T, EXTRACTOR_I, EXTRACTOR_A, EXTRACTOR_V,
                  dict(text=K_text, image=K_image, audio=K_audio, vedio=K_vedio),
                  OUT_DIM)

# ----------------------- CLI -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--data_root", type=str, default=None, help="数据根目录（默认为 ./data）")
    parser.add_argument("--codi_root", type=str, default=None, help="CoDi 模型根目录")
    parser.add_argument("--codi_ckpt", type=str, default=None, help="CoDi 模型检查点目录")
    
    # 训练参数
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=300)
    parser.add_argument("--lambda_view", type=float, default=0.50)
    parser.add_argument("--lambda_q", type=float, default=1e-2)
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--img_take", type=int, default=8)
    parser.add_argument("--aud_seconds", type=float, default=8.0)
    parser.add_argument("--skip_iv_until", type=int, default=600)

    # 表示维度/Adapter
    parser.add_argument("--out_dim", type=int, default=512)
    parser.add_argument("--adapter", type=str, default="mlp", choices=["identity","mlp","xattn"])
    parser.add_argument("--adapter_depth", type=int, default=2)
    parser.add_argument("--adapter_width", type=int, default=1024)
    parser.add_argument("--adapter_heads", type=int, default=4)

    # K（共享语义 token 数）：全局或逐模态
    parser.add_argument("--K", type=int, default=16)  # 默认比原来大（8 -> 16）
    parser.add_argument("--K_text", type=int, default=None)
    parser.add_argument("--K_image", type=int, default=None)
    parser.add_argument("--K_audio", type=int, default=None)
    parser.add_argument("--K_vedio", type=int, default=None)

    # R（代表 token 数）
    parser.add_argument("--rep_tokens", type=int, default=4)
    parser.add_argument("--lambda_rep_view", type=float, default=0.25)
    parser.add_argument("--lambda_rep_xmodal", type=float, default=0.25)

    args = parser.parse_args()
    
    # 更新路径（如果提供了命令行参数）
    if args.data_root:
        DATA = Path(args.data_root).resolve()
        DVID, DIMG, DAUD, DTXT = DATA/"vedio", DATA/"image", DATA/"audio", DATA/"text"
        print(f"📁 使用数据目录: {DATA}")
    if args.codi_root:
        CODI_ROOT = Path(args.codi_root).resolve()
        if str(CODI_ROOT) not in sys.path:
            sys.path.insert(0, str(CODI_ROOT))
        print(f"📁 使用 CoDi 根目录: {CODI_ROOT}")
    if args.codi_ckpt:
        CODI_CKPT = Path(args.codi_ckpt).resolve()
        print(f"📁 使用 CoDi 检查点目录: {CODI_CKPT}")

    stems = list_stems()
    cand = [st for st in stems if has_modalities(st) >= 2]
    assert cand, "没有可用的视频（至少需要两种模态）"
    st = args.video_id or cand[0]
    print("训练单视频:", st)

    train_single_vedio(st, args)

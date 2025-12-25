# -*- coding: utf-8 -*-
"""
stage1_dupsafe.py
Stage-1 训练（抗塌缩稳定版）
- Gumbel-Sinkhorn 平衡分配抽 K-token
- VICReg 去塌缩（variance / covariance）
- EMA teacher 视图一致性（BYOL风格）
- Sinkhorn-OT 跨模态对齐
- DPP + 注意力熵 + 非重叠（dup 自适应调权）
"""

import os, sys, glob, random, math, signal
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ============== 路径/设备 ==============
ROOT = Path.cwd()
_ICODE_V3 = ROOT / "i-Code-V3"
if _ICODE_V3.exists():
    CODI_ROOT = _ICODE_V3.resolve()
    CODI_CKPT = _ICODE_V3 / "checkpoints"
else:
    CODI_ROOT = Path("/home/liz0g/semantic-communication/i-Code-V3").resolve()
    CODI_CKPT = Path("/home/liz0g/semantic-communication/i-Code-V3/checkpoints").resolve()
if str(CODI_ROOT) not in sys.path:
    sys.path.insert(0, str(CODI_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = ROOT / "data"
DVID, DIMG, DAUD, DTXT = DATA/"vedio", DATA/"image", DATA/"audio", DATA/"text"
CKP = Path("snapshots"); CKP.mkdir(parents=True, exist_ok=True)

# ============== CoDi / HF 编码 ==============
_CODI_READY = False
_HF_READY = False
net = None

def _load_codi(ckpt_dir: Path):
    from core.models.model_module_infer import model_module as CoDiModule
    return CoDiModule(data_dir=str(ckpt_dir), pth=["CoDi_encoders.pth"], fp16=True)

def _ensure_codi():
    global _CODI_READY, net
    if _CODI_READY: return
    mod = _load_codi(CODI_CKPT)
    net = mod.net if hasattr(mod, "net") else mod
    for p in net.parameters(): p.requires_grad_(False)
    net = net.to(device).eval()
    print(f"✅ 已加载 CoDi encoders: {CODI_CKPT}")
    _CODI_READY = True

def _ensure_hf():
    global _HF_READY, _tok, _txt, _vis, _imgp
    if _HF_READY: return
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
    name = os.environ.get("HF_CLIP", "openai/clip-vit-base-patch32")
    _tok = CLIPTokenizer.from_pretrained(name)
    _txt = CLIPTextModel.from_pretrained(name).to(device).eval()
    _imgp = CLIPImageProcessor.from_pretrained(name)
    _vis = CLIPVisionModel.from_pretrained(name).to(device).eval()
    _HF_READY = True
    print("ℹ️ 使用 HF CLIP 兜底（text/image）")

@torch.no_grad()
def encode_text(list_of_str: List[str]) -> torch.Tensor:
    """只用 CoDi 的 clip 做 text 编码；遇到 ShortTensor bug 时手动 tokenizer+model，强制 long()。"""
    _ensure_codi()
    clip = getattr(net, "clip", None)
    if clip is None:
        raise RuntimeError("CoDi net.clip 不存在，检查 CoDi 加载。")

    # 先尝试原来的 encode_text_noproj（如果底层已经修好就直接用）
    if hasattr(clip, "encode_text_noproj"):
        try:
            out = clip.encode_text_noproj(list_of_str)
            z = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            return z.float().detach()
        except RuntimeError as e:
            # 只有碰到 ShortTensor 的 embedding 报错才兜底到手动路径，其他错误照常抛出
            if "ShortTensor" not in str(e):
                raise

    # 手动路径：tokenizer -> model.text_model，强制 input_ids.long()
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
    return z.float().detach()

@torch.no_grad()
def encode_image(paths: List[str]) -> torch.Tensor:
    """只用 CoDi 的 clip.encode_vision_noproj 做 image 编码，不再走 HF 兜底。"""
    _ensure_codi()
    clip = getattr(net, "clip", None)
    if clip is None or not hasattr(clip, "encode_vision_noproj"):
        raise RuntimeError("CoDi clip.encode_vision_noproj 不存在，检查 CoDi 安装。")

    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    xs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append((tfm(img).unsqueeze(0) * 2 - 1))
    x = torch.cat(xs, 0).to(device)
    out = clip.encode_vision_noproj(x)
    f = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    return f.float().detach()

@torch.no_grad()
def encode_audio(paths: List[str], seconds: float = 8.0) -> Optional[torch.Tensor]:
    _ensure_codi()
    clap = getattr(net, "clap", None)
    if clap is None: return None
    try:
        import torchaudio
    except Exception:
        print("(未安装 torchaudio，跳过 audio)")
        return None
    target_sr = getattr(clap, "sample_rate", 48000)
    if isinstance(paths, (str, Path)): paths=[paths]
    wavs=[]
    for p in paths:
        p=Path(p); 
        if not p.exists(): continue
        w, sr = torchaudio.load(str(p))
        w = w.mean(0, keepdim=True)
        if sr != target_sr:
            w = torchaudio.functional.resample(w, sr, target_sr)
        L = int(seconds*target_sr)
        if w.size(1)>=L:
            st = random.randint(0, w.size(1)-L); w = w[:, st:st+L]
        else:
            w = F.pad(w, (0, L-w.size(1)))
        wavs.append(w)
    if not wavs: return None
    batch = torch.cat(wavs,0).to(device)
    out = clap.encode_audio_noproj(batch) if hasattr(clap,"encode_audio_noproj") else clap(batch)
    if out.ndim==2: out = out.unsqueeze(1)
    return out.float().detach()

@torch.no_grad()
def encode_video(paths: List[str], max_frames: int = 8) -> Optional[torch.Tensor]:
    """
    简易视频编码：
    - 抽 max_frames 帧
    - 逐帧用与 encode_image 相同的视觉编码器
    - 返回帧拼接后的 token 序列 (F*L, D)，保持"多 token"性质给 SemanticExtractor 抽 K
    """
    try:
        import cv2
    except Exception as e:
        print("(未安装 opencv-python，跳过 video)", e)
        return None
    _ensure_codi()
    clip = getattr(net, "clip", None)
    use_codi = clip is not None and hasattr(clip, "encode_vision_noproj")

    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    all_tokens = []
    for vp in paths:
        vp = str(vp)
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        idxs = list(range(total)) if total <= max_frames else np.linspace(0, total-1, max_frames, dtype=int).tolist()
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            continue

        # 批量过视觉编码
        batch = torch.cat([(tfm(img).unsqueeze(0)*2-1) for img in frames], 0).to(device)
        if use_codi:
            out = clip.encode_vision_noproj(batch)
            fseq = out.last_hidden_state if hasattr(out, "last_hidden_state") else out  # (F,L,D)
        else:
            _ensure_hf()
            inputs = _imgp(images=frames, return_tensors="pt").to(device)
            out = _vis(**inputs)
            fseq = out.last_hidden_state  # (F,L,D)
        # 拼成 (F*L, D)，给抽 K 更密的候选
        all_tokens.append(fseq.reshape(-1, fseq.shape[-1]).float().detach())

    if not all_tokens:
        return None
    return torch.cat(all_tokens, dim=0)

# ============== 数据工具 ==============
def list_stems() -> List[str]:
    vids = {Path(p).stem for p in glob.glob(str(DVID/"*.mp4"))}
    aus  = {Path(p).stem for p in glob.glob(str(DAUD/"*.wav"))}
    txts = {Path(p).stem for p in glob.glob(str(DTXT/"*.txt"))}
    srt  = {Path(p).stem for p in glob.glob(str(DTXT/"*.srt"))}
    imgs = set(Path(p).name.split("_")[0] for p in glob.glob(str(DIMG/"*_*.jpg")))
    return sorted(vids | aus | txts | srt | imgs)

def imgs_for(st: str, take: int = 8) -> List[str]:
    xs = sorted(glob.glob(str(DIMG/f"{st}_*.jpg")))
    if len(xs) > take: xs = random.sample(xs, take)
    return xs

def text_for(st: str) -> str:
    p = DTXT / f"{st}.txt"
    if p.exists():
        try: return p.read_text(errors="ignore").strip()
        except: pass
    ps = DTXT / f"{st}.srt"
    if ps.exists():
        raw = ps.read_text(errors="ignore")
        lines=[]
        for ln in raw.splitlines():
            ln=ln.strip()
            if not ln or ln.isdigit() or "-->" in ln: continue
            lines.append(ln)
        return " ".join(lines).strip()
    return ""

def has_modalities(st: str) -> int:
    c=0
    if (DVID/f"{st}.mp4").exists(): c+=1
    if (DAUD/f"{st}.wav").exists(): c+=1
    if (DTXT/f"{st}.txt").exists() or (DTXT/f"{st}.srt").exists(): c+=1
    if len(imgs_for(st))>0: c+=1
    return c

# ============== Adapter / 抽 K-token ==============
class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            for _ in range(depth)
        ])
    def forward(self, x):
        for blk in self.layers: x = x + blk(x)
        return x

class SimpleCrossAttn(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ln   = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x_list):
        valid=[x for x in x_list if x is not None]
        if len(valid)<=1: return x_list
        ctx = torch.cat(valid,0).unsqueeze(0)
        outs=[]
        for x in x_list:
            if x is None: outs.append(None); continue
            q = self.ln(x).unsqueeze(0)
            attn,_ = self.attn(q, ctx, ctx, need_weights=False)
            y = x + attn.squeeze(0)
            y = y + self.ffn(y)
            outs.append(y)
        return outs

class AdapterHead(nn.Module):
    def __init__(self, dim, kind="mlp", depth=2, width=1024, heads=4):
        super().__init__()
        self.kind = kind
        if kind=="identity": self.body = nn.Identity()
        elif kind=="mlp": self.body = ResidualMLP(dim, width, depth)
        elif kind=="xattn": self.body = SimpleCrossAttn(dim, heads)
        else: raise ValueError(kind)
    def forward(self, xt, xi, xa, xv):
        if isinstance(self.body, SimpleCrossAttn):
            xt,xi,xa,xv = self.body([xt,xi,xa,xv]); return xt,xi,xa,xv
        go = (lambda x: None if x is None else self.body(x))
        return go(xt),go(xi),go(xa),go(xv)

class SemanticExtractor(nn.Module):
    def __init__(self, in_dim, out_dim=512, K=8, q_res_w=0.6):
        super().__init__()
        self.K=K
        self.query = nn.Parameter(torch.randn(K, out_dim))
        nn.init.orthogonal_(self.query)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.ffn  = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.attn_tau = 1.0
        self.q_res_w  = q_res_w
        self.pool_stride = 1
        self.candidate_ratio = 1.2  # 候选位置比例，image模态可设为1.1
        self.last_attn_w = None   # (1,K,S)
        self.last_attn_entropy = None

# --- utilities ---
def _iter_extrs(extrs):
    for e in extrs:
        if isinstance(e, (tuple, list)) and len(e)==2:
            name, m = e
        else:
            name, m = "?", e
        yield name, m

def _l2n(x): return F.normalize(x, dim=-1)

def diversity_loss(Z):
    Z=_l2n(Z); G=Z@Z.t(); eye=torch.eye(G.size(0), device=G.device)
    return ((G-eye)**2).mean()

def query_ortho_loss(mod):
    q=_l2n(mod.query); G=q@q.t(); eye=torch.eye(G.size(0), device=G.device)
    return ((G-eye)**2).mean()

def attention_entropy_penalty(extrs):
    ents=[]
    for name,m in _iter_extrs(extrs):
        if m is None or getattr(m,"last_attn_entropy",None) is None: continue
        ents.append(m.last_attn_entropy)
    if ents: return torch.stack(ents).mean()
    return None

def attention_nonoverlap_loss(extrs):
    losses=[]
    for name,m in _iter_extrs(extrs):
        if m is None or getattr(m,"last_attn_w",None) is None: continue
        W=m.last_attn_w  # (1,K,S)
        G=W@W.transpose(1,2)
        eye=torch.eye(G.size(-1), device=G.device).unsqueeze(0)
        off=G*(1.0-eye)
        losses.append((off**2).mean())
    if losses: return torch.stack(losses).mean()
    return None

def direct_dup_penalty(extrs):
    penalties=[]
    for name,m in _iter_extrs(extrs):
        if m is None or getattr(m,"last_attn_w",None) is None: continue
        W=m.last_attn_w[0]  # (K,S)
        col_sum=W.sum(dim=0)
        ideal=1.0/W.size(0)
        excess=(col_sum-ideal).clamp(min=0)
        penalties.append(excess.sum())
    if penalties: return torch.stack(penalties).mean()
    return None

def _pairwise_cost(A,B,metric="cos"):
    A=_l2n(A); B=_l2n(B)
    if metric=="cos": return 1.0 - torch.einsum("id,jd->ij", A,B)
    return torch.cdist(A,B,p=2)

@torch.no_grad()
def _sinkhorn(logK, n_iter=30):
    logP=logK.clone()
    for _ in range(n_iter):
        logP = logP - torch.logsumexp(logP,1,True)
        logP = logP - torch.logsumexp(logP,0,True)
    return logP

def sinkhorn_ot_loss(A,B,eps=0.03,n_iter=30,metric="cos",stopgrad=True):
    assert A.dim()==2 and B.dim()==2 and A.size(0)==B.size(0)
    def _ot(a,b):
        C=_pairwise_cost(a,b,metric); logK=-C/eps
        P=_sinkhorn(logK,n_iter).exp()
        return (P*C).sum()/A.size(0)
    return 0.5*(_ot(A,B.detach())+_ot(B,A.detach())) if stopgrad else _ot(A,B)

# --- Gumbel-Sinkhorn balanced ---
def _sinkhorn_balanced_log(logM, r, c, n_iter=10):
    logP=logM
    for _ in range(n_iter):
        logP = logP - torch.logsumexp(logP,1,True) + torch.log(r.clamp_min(1e-8))[:,None]
        logP = logP - torch.logsumexp(logP,0,True) + torch.log(c.clamp_min(1e-8))[None,:]
    return logP

def gumbel_sinkhorn_balanced(logits, tau=0.7, n_iter=10, r=None, c=None, training=True):
    g = -(torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-8)))) if training else 0.0
    logM=(logits+g)/max(tau,1e-6)
    S,K=logM.shape
    if r is None: r=torch.full((S,), K/float(S), device=logM.device, dtype=logM.dtype)
    if c is None: c=torch.ones(K, device=logM.device, dtype=logM.dtype)
    return _sinkhorn_balanced_log(logM,r,c,n_iter).exp()

def dpp_penalty_from_extractors(extrs, eps=1e-3, audio_boost:float=1.0):
    losses=[]
    for name,m in _iter_extrs(extrs):
        W=getattr(m,"last_attn_w",None)
        if W is None: continue
        W=W[0]
        L=W@W.t() + eps*torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        sign,logdet=torch.slogdet(L)
        if sign>0: losses.append((-logdet) * (audio_boost if name=="A" else 1.0))
    return None if not losses else sum(losses)/len(losses)

def _gs_forward(self, seq: torch.Tensor) -> torch.Tensor:
    if seq.dim()==3: seq = seq.reshape(seq.shape[0]*seq.shape[1], seq.shape[-1])
    elif seq.dim()!=2: raise ValueError("seq must be (L,D) or (B,L,D)")
    if getattr(self,"pool_stride",1)>1: seq = seq[::self.pool_stride]

    Q=self.query
    K=self.key(seq); V=self.value(seq)
    K_n=_l2n(K); Q_n=_l2n(Q)
    logits = K_n @ Q_n.t()  # (S,K)

    S,Kslots=logits.shape
    ratio = getattr(self, "candidate_ratio", 1.2)
    Sprime=min(S, max(Kslots, int(math.ceil(ratio*Kslots))))
    with torch.no_grad():
        scores,_=logits.max(dim=1)
        idx=torch.topk(scores, k=Sprime, sorted=False).indices

    logits_sel = logits.index_select(0, idx)   # (S',K)
    r = torch.full((Sprime,), Kslots/float(Sprime), device=logits_sel.device, dtype=logits_sel.dtype)
    c = torch.ones(Kslots, device=logits_sel.device, dtype=logits_sel.dtype)
    P = gumbel_sinkhorn_balanced(logits_sel, tau=getattr(self,"attn_tau",1.0), r=r, c=c, training=self.training) # (S',K)

    V_sel = V.index_select(0, idx)
    Z = P.t() @ V_sel                           # (K,D)
    if getattr(self,"q_res_w",0.0)>0: Z = Z + self.q_res_w*self.query
    Z = self.ffn(self.norm(Z))

    W_full = logits.new_zeros(Kslots, S)
    W_full[:, idx] = P.t()
    Wn = W_full / (W_full.sum(dim=1, keepdim=True)+1e-8)
    self.last_attn_w = Wn.unsqueeze(0).detach()
    ent = -(Wn*(Wn.clamp_min(1e-8).log())).sum(dim=1).mean()
    self.last_attn_entropy = ent.detach()
    return Z
SemanticExtractor.forward = _gs_forward

# ============== 维度探测 / 构建提取器 ==============
def _probe_dim_text():
    try: return int(encode_text(["probe"]).shape[-1])
    except: return None
def _probe_dim_image():
    try:
        any_img = next(iter(glob.glob(str(DIMG/"*_*.jpg"))), None)
        if not any_img: return None
        return int(encode_image([any_img]).shape[-1])
    except: return None
def _probe_dim_audio():
    """复用函数名，但现在探测的是 Video 维度"""
    try:
        any_vid = next(iter(glob.glob(str(DVID/"*.mp4"))), None)
        if not any_vid: return None
        z = encode_video([any_vid], max_frames=4)
        return None if z is None else int(z.shape[-1])
    except:
        return None

def build_extractors(Kt,Ki,Ka, out_dim=512):
    """
    注意：第三个 Ka 现在表示 K_video（沿用变量名，避免到处改）。
    """
    Dt,Di,Da = _probe_dim_text(), _probe_dim_image(), _probe_dim_audio()
    ET = SemanticExtractor(Dt,out_dim,Kt,q_res_w=0.10).to(device) if Dt else None
    EI = SemanticExtractor(Di,out_dim,Ki,q_res_w=0.10).to(device) if Di else None
    EA = SemanticExtractor(Da,out_dim,Ka,q_res_w=0.10).to(device) if Da else None
    return ET,EI,EA

# ============== 数据一次编码 ==============
@torch.no_grad()
def _encode_once(st: str, take_img=4, aud_seconds=8.0):
    """
    现在第三路改为视频；复用 aud_seconds 作为每次抽帧数（整数）。
    """
    t=i=a=None
    txt=text_for(st)
    if txt: t=encode_text([txt])
    ims=imgs_for(st, take=take_img)
    if ims: i=encode_image(ims)
    # 替换音频为视频
    wv = DVID / f"{st}.mp4"
    if wv.exists():
        try:
            max_frames = max(4, int(aud_seconds))
        except:
            max_frames = 8
        a = encode_video([wv], max_frames=max_frames)
        if a is None:
            print("视频编码失败，跳过 video")
    return t,i,a

def _extract_K(t,i,a, ET,EI,EA):
    zt = ET(t.squeeze(0)) if (t is not None and ET is not None) else None
    zi = EI(i.squeeze(0)) if (i is not None and EI is not None) else None
    za = EA(a.squeeze(0)) if (a is not None and EA is not None) else None
    return zt,zi,za

# ============== 指标 ==============
def dup_rate_of(mod: Optional[SemanticExtractor]) -> float:
    if not mod or getattr(mod,"last_attn_w",None) is None: return 0.0
    top = mod.last_attn_w[0].argmax(-1)  # (K,)
    return 1 - top.unique().numel()/top.numel()

def coverage_rate_of(mod: Optional[SemanticExtractor], thresh: float=0.6) -> float:
    if not mod or getattr(mod,"last_attn_w",None) is None: return 0.0
    W = mod.last_attn_w[0]  # (K,S)
    mx = W.max(dim=1).values
    return (mx >= thresh).float().mean().item()

# ============== VICReg（variance/covariance） ==============
def _vicreg_variance(z, gamma=1.0, eps=1e-4):
    if z is None: return None
    std = torch.sqrt(z.var(dim=0)+eps)
    # 强制每一维标准差 >= 1
    return gamma * torch.mean(F.relu(1.0 - std))

def _vicreg_covariance(z, gamma=1.0):
    if z is None: return None
    z = z - z.mean(dim=0, keepdim=True)
    N,D = z.shape
    c = (z.t() @ z) / (N-1)
    off = c - torch.diag(torch.diag(c))
    return gamma * (off**2).sum()/D

# ============== EMA Teacher（BYOL式视图一致） ==============
class EMATeacher(nn.Module):
    def __init__(self, student: nn.Module, decay=0.996):
        super().__init__()
        self.decay = decay
        self.teacher = copy_like(student)
        for p in self.teacher.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, student: nn.Module):
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), self.teacher.parameters()):
                pt.data.mul_(self.decay).add_(ps.data*(1-self.decay))

def copy_like(m: nn.Module):
    import copy
    t = copy.deepcopy(m)
    return t

# ============== 单视频一次损失 ==============
def one_video_loss(st, it, cfg, ET,EI,EA, adapter, teacher):
    # 两个视图
    t1,i1,a1 = _encode_once(st, cfg["img_take"], cfg["aud_seconds"])
    zt1,zi1,za1 = _extract_K(t1,i1,a1, ET,EI,EA)
    t2,i2,a2 = _encode_once(st, cfg["img_take"], max(4.0, cfg["aud_seconds"]-2.0))
    zt2,zi2,za2 = _extract_K(t2,i2,a2, ET,EI,EA)

    # Adapter
    zt1,zi1,za1,_ = adapter(zt1,zi1,za1,None)
    zt2,zi2,za2,_ = adapter(zt2,zi2,za2,None)

    total, terms = 0.0, {}

    # === VICReg 去塌缩（单模态）
    vvar = []
    for z in [zt1,zi1,za1, zt2,zi2,za2]:
        if z is not None:
            vvar.append(_vicreg_variance(z, gamma=1.0))
            vvar.append(_vicreg_covariance(z, gamma=1.0))
    if vvar:
        vmean = sum(vvar)/len(vvar)
        terms["vicreg"] = vmean
        total += cfg["lambda_vicreg"] * vmean

    # === 视图一致（BYOL：student -> teacher）
    def _byol(a,b):
        if a is None or b is None: return None
        a=_l2n(a); b=_l2n(b)
        return (2 - 2*(a*b).sum(dim=-1).mean())  # 1 - cos → [0,2]
    # teacher 只用第1视图 EMA，和第2视图 student 对齐（或反之）
    bys=[]
    if zt1 is not None and zt2 is not None: bys.append(_byol(zt2, zt1.detach()))
    if zi1 is not None and zi2 is not None: bys.append(_byol(zi2, zi1.detach()))
    if za1 is not None and za2 is not None: bys.append(_byol(za2, za1.detach()))
    if bys:
        bmean = sum(bys)/len(bys)
        terms["byol"] = bmean
        total += cfg["lambda_byol"] * bmean

    # === 跨模态对齐（Sinkhorn-OT）
    pairs=[]
    if zt1 is not None and zi1 is not None: pairs.append((zt1,zi1))
    if zt1 is not None and za1 is not None: pairs.append((zt1,za1))
    if zi1 is not None and za1 is not None and it > cfg["skip_ti_until"]: pairs.append((zi1,za1))
    if pairs:
        xmodal = sum(sinkhorn_ot_loss(a,b) for a,b in pairs)/len(pairs)
        terms["xmodal"] = xmodal
        total += cfg["lambda_xmodal"] * xmodal

    # === 结构正则（多样性 + 注意力）
    dloss=[]
    for z in [zt1,zi1,za1, zt2,zi2,za2]:
        if z is not None: dloss.append(diversity_loss(z))
    if dloss:
        dmean=sum(dloss)/len(dloss)
        terms["div"]=dmean
        total += cfg["lambda_div"] * dmean

    attn_pen = attention_entropy_penalty([ET,EI,EA])
    if attn_pen is not None:
        terms["attn"]=attn_pen
        total += (-cfg["lambda_attn"] if cfg["attn_reward"] else cfg["lambda_attn"]) * attn_pen

    ano_pen = attention_nonoverlap_loss([ET,EI,EA])
    if ano_pen is not None:
        terms["ano"]=ano_pen
        total += cfg["lambda_ano"] * ano_pen

    qlist=[]
    for M in (ET,EI,EA):
        if M is not None: qlist.append(query_ortho_loss(M))
    if qlist:
        qloss=sum(qlist)/len(qlist)
        terms["q"]=qloss
        total += cfg["lambda_q"] * qloss

    # === DPP
    dpp = dpp_penalty_from_extractors([ET,EI,EA], eps=1e-3)
    if dpp is not None:
        terms["dpp"]=dpp
        total += cfg["lambda_dpp"] * dpp

    # === 直罚 dup（高 dup 时启用）
    dup_global = max(dup_rate_of(ET), dup_rate_of(EI), dup_rate_of(EA))
    if dup_global > 0.70:
        dup_pen = direct_dup_penalty([ET,EI,EA])
        if dup_pen is not None:
            terms["dup_pen"]=dup_pen
            total += (dup_global-0.70)*8.0 * dup_pen

    return total, terms

# ============== 调度 ==============
def sched_by_dup(it, dup):
    s = dict(
        tau=1.0, attn_reward=True,
        lam_x=0.40, lam_div=0.30, lam_attn=0.15, lam_ano=0.50, lam_q=1e-2,
        lam_vicreg=1.0, lam_byol=0.4, lam_dpp=1.5e-3,
        # 覆盖/边际
        lambda_cov=0.0, cov_thresh=0.6,
        lambda_margin=0.0, margin_thresh=0.15
    )
    if it <= 300:
        s.update(tau=0.7, lam_div=0.5, lam_attn=0.25, lam_x=0.25, lam_byol=0.6)
    elif it <= 700:
        s.update(tau=1.1, lam_div=0.35, lam_x=0.35, lam_byol=0.5)
    elif it <= 900:
        s.update(tau=0.9, lam_div=0.30, lam_x=0.45, lam_byol=0.4)
    else:
        # 锐化阶段 v2
        s.update(
            tau=0.45,
            attn_reward=False,
            lam_div=0.08,
            lam_attn=0.05,
            lam_ano=0.40,
            lam_vicreg=0.40,
            lam_byol=0.30,
            lam_dpp=6.0e-4,
            lam_x=0.70,
            lambda_cov=1.80,
            cov_thresh=0.72,
            lambda_margin=1.10,
            margin_thresh=0.20
        )

    if dup > 0.9:
        s.update(tau=0.30, lam_div=max(0.15, s["lam_div"]*1.5), lam_ano=s["lam_ano"]*1.3,
                 lam_x=s["lam_x"]*0.8, lambda_cov=max(1.2, s["lambda_cov"]), lambda_margin=max(1.0, s["lambda_margin"]))
    elif dup > 0.7:
        s.update(tau=0.40, lam_div=max(0.12, s["lam_div"]*1.2),
                 lambda_cov=max(1.0, s["lambda_cov"]), lambda_margin=max(0.9, s["lambda_margin"]))
    return s

# ============== 训练主循环 ==============
def train_one(st, args):
    import copy
    Kt = args.K_text  if args.K_text  is not None else args.K
    Ki = args.K_image if args.K_image is not None else args.K
    Ka = args.K_audio if args.K_audio is not None else args.K
    OUT = args.out_dim

    ET,EI,EA = build_extractors(Kt,Ki,Ka, OUT)
    # 对image模态设置更紧的候选位置比例
    if EI is not None:
        EI.candidate_ratio = 1.1
    params=[]
    for M in (ET,EI,EA):
        if M: params += list(M.parameters())
    assert params, "没有任何可训练的提取器"

    adapter = AdapterHead(dim=OUT, kind=args.adapter, depth=args.adapter_depth,
                          width=args.adapter_width, heads=args.adapter_heads).to(device)
    params += list(adapter.parameters())

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Teacher：只复制 adapter 与各 Extractor（结构一致即可）
    teacher = None  # 这里我们用 student 的第1视图作 target，不额外跑 teacher 计算图

    def _save(tag):
        path = CKP/f"{tag}.pth"
        torch.save({
            "ET": (ET.state_dict() if ET else None),
            "EI": (EI.state_dict() if EI else None),
            "EA": (EA.state_dict() if EA else None),
            "OUT": OUT,
            "K": dict(text=Kt,image=Ki,audio=Ka),
            "adapter": adapter.state_dict()
        }, str(path))
        print("💾 saved", path)

    def _sigint(sig, frame):
        print("\n[CTRL-C] saving snapshot ...")
        _save(f"stage1_dupsafe_it{it}")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    for it in range(1, args.steps+1):
        dup_now = max(dup_rate_of(ET), dup_rate_of(EI), dup_rate_of(EA))
        sch = sched_by_dup(it, dup_now)

        # 强制进入锐化阶段做小收口（只用在这 200~300 步微调）
        if True:
            sch["tau"] = 0.45
            sch["attn_reward"] = False
            sch["lam_div"] = 0.08
            sch["lam_attn"] = 0.05
            sch["lam_ano"] = 0.40
            sch["lam_vicreg"] = 0.40
            sch["lam_byol"] = 0.30
            sch["lam_dpp"] = 6.0e-4
            sch["lam_x"] = 0.70
            sch["lambda_cov"] = 1.80
            sch["cov_thresh"] = 0.72
            sch["lambda_margin"] = 1.10
            sch["margin_thresh"] = 0.20

        for name, M in (("T", ET), ("I", EI), ("V", EA)):
            if not M: continue
            M.attn_tau = sch["tau"]
            M.q_res_w = 0.15 if it < 200 else 0.05
            M.pool_stride = 1 if it < 200 else 2
            if name == "I":  # 图像更尖一点
                M.attn_tau = max(0.35, sch["tau"] * 0.85)  # 再降一些温度
                M.q_res_w = 0.03  # 降 query 残差

        cfg = dict(
            img_take=args.img_take,
            aud_seconds=args.aud_seconds,
            skip_ti_until=args.skip_iv_until,
            lambda_xmodal=sch["lam_x"],
            lambda_div=sch["lam_div"],
            lambda_attn=sch["lam_attn"],
            attn_reward=sch["attn_reward"],
            lambda_q=sch["lam_q"],
            lambda_ano=sch["lam_ano"],
            lambda_vicreg=sch["lam_vicreg"],
            lambda_byol=sch["lam_byol"],
            lambda_dpp=sch["lam_dpp"],
            lambda_cov=sch["lambda_cov"],
            cov_thresh=sch["cov_thresh"],
            lambda_margin=sch["lambda_margin"],
            margin_thresh=sch["margin_thresh"]
        )

        loss, terms = one_video_loss(st, it, cfg, ET,EI,EA, adapter, teacher)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()

        if it == 1 or it % 10 == 0:
            dupT, dupI, dupA = dup_rate_of(ET), dup_rate_of(EI), dup_rate_of(EA)
            covT, covI, covA = coverage_rate_of(ET), coverage_rate_of(EI), coverage_rate_of(EA)
            print(f"[{it}/{args.steps}] loss={loss.item():.4f} "
                  f"x={terms.get('xmodal',0):.3f} byol={terms.get('byol',0):.3f} "
                  f"vicreg={terms.get('vicreg',0):.3f} div={terms.get('div',0):.3f} "
                  f"attn={terms.get('attn',0):.3f} ano={terms.get('ano',0):.3f} q={terms.get('q',0):.3f} "
                  f"dpp={terms.get('dpp',0):.3f} dup_pen={terms.get('dup_pen',0):.3f} "
                  f"| dupT={dupT:.2f} dupI={dupI:.2f} dupV={dupA:.2f} "
                  f"covT@.6={covT:.2f} covI@.6={covI:.2f} covV@.6={covA:.2f} "
                  f"(tau={sch['tau']:.2f})")

        if it % args.save_every == 0:
            _save(f"dupsafe_step_{it:04d}")

    _save("stage1_dupsafe_final")

# ============== CLI ==============
# ============== CLI ==============
def train_multi(stems, args):
    # 多样本 Stage1：每一步随机抽一个 stem（比如 coco_000123）
    import signal, sys, random

    Kt = args.K_text  if args.K_text  is not None else args.K
    Ki = args.K_image if args.K_image is not None else args.K
    Ka = args.K_audio if args.K_audio is not None else args.K
    OUT = args.out_dim

    # 一套通用提取器 + adapter，所有样本共享
    ET, EI, EA = build_extractors(Kt, Ki, Ka, OUT)
    if EI is not None:
        EI.candidate_ratio = 1.1

    params = []
    for M in (ET, EI, EA):
        if M is not None:
            params += list(M.parameters())

    adapter = AdapterHead(
        dim=OUT,
        kind=args.adapter,
        depth=args.adapter_depth,
        width=args.adapter_width,
        heads=args.adapter_heads,
    ).to(device)
    params += list(adapter.parameters())

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    teacher = None  # 保持和原脚本一致

    def _save(tag: str):
        path_ck = CKP / f"{tag}.pth"
        torch.save(
            {
                "ET": (ET.state_dict() if ET is not None else None),
                "EI": (EI.state_dict() if EI is not None else None),
                "EA": (EA.state_dict() if EA is not None else None),
                "OUT": OUT,
                "K": dict(text=Kt, image=Ki, audio=Ka),
                "adapter": adapter.state_dict(),
            },
            str(path_ck),
        )
        print("💾 saved", path_ck, flush=True)

    def _sigint(sig, frame):
        print("\n[CTRL-C] saving snapshot ...", flush=True)
        _save(f"stage1_dupsafe_it{it}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    print(f"[Stage1] 多样本训练启动，样本数 = {len(stems)}", flush=True)

    for it in range(1, args.steps + 1):
        # ⭐ 每一步随机抽一个样本 ID（例如 coco_000123）
        st = random.choice(stems)

        dup_now = max(dup_rate_of(ET), dup_rate_of(EI), dup_rate_of(EA))
        sch = sched_by_dup(it, dup_now)

        # 保持你之前的“锐化阶段”设置
        sch["tau"] = 0.45
        sch["attn_reward"] = False
        sch["lam_div"] = 0.08
        sch["lam_attn"] = 0.05
        sch["lam_ano"] = 0.40
        sch["lam_vicreg"] = 0.40
        sch["lam_byol"] = 0.30
        sch["lam_dpp"] = 6.0e-4
        sch["lam_x"] = 0.70
        sch["lambda_cov"] = 1.80
        sch["cov_thresh"] = 0.72
        sch["lambda_margin"] = 1.10
        sch["margin_thresh"] = 0.20

        for name, M in (("T", ET), ("I", EI), ("V", EA)):
            if M is None:
                continue
            M.attn_tau = sch["tau"]
            M.q_res_w = 0.15 if it < 200 else 0.05
            M.pool_stride = 1 if it < 200 else 2
            if name == "I":
                M.attn_tau = max(0.35, sch["tau"] * 0.85)
                M.q_res_w = 0.03

        cfg = dict(
            img_take=args.img_take,
            aud_seconds=args.aud_seconds,
            skip_ti_until=args.skip_iv_until,
            lambda_xmodal=sch["lam_x"],
            lambda_div=sch["lam_div"],
            lambda_attn=sch["lam_attn"],
            attn_reward=sch["attn_reward"],
            lambda_q=sch["lam_q"],
            lambda_ano=sch["lam_ano"],
            lambda_vicreg=sch["lam_vicreg"],
            lambda_byol=sch["lam_byol"],
            lambda_dpp=sch["lam_dpp"],
            lambda_cov=sch["lambda_cov"],
            cov_thresh=sch["cov_thresh"],
            lambda_margin=sch["lambda_margin"],
            margin_thresh=sch["margin_thresh"],
        )

        # ⭐ 这里的 st 每步都在变（不同样本）
        loss, terms = one_video_loss(st, it, cfg, ET, EI, EA, adapter, teacher)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()

        if it == 1 or it % 50 == 0:
            dupT, dupI, dupA = dup_rate_of(ET), dup_rate_of(EI), dup_rate_of(EA)
            covT, covI, covA = coverage_rate_of(ET), coverage_rate_of(EI), coverage_rate_of(EA)
            print(
                f"[{it}/{args.steps}] loss={loss.item():.4f} "
                f"x={terms.get('xmodal',0):.3f} byol={terms.get('byol',0):.3f} "
                f"vicreg={terms.get('vicreg',0):.3f} div={terms.get('div',0):.3f} "
                f"attn={terms.get('attn',0):.3f} ano={terms.get('ano',0):.3f} q={terms.get('q',0):.3f} "
                f"dpp={terms.get('dpp',0):.3f} dup_pen={terms.get('dup_pen',0):.3f} "
                f"| dupT={dupT:.2f} dupI={dupI:.2f} dupV={dupA:.2f} "
                f"covT@.6={covT:.2f} covI@.6={covI:.2f} covV@.6={covA:.2f} "
                f"(tau={sch['tau']:.2f}) st={st}",
                flush=True,
            )

        if it % args.save_every == 0:
            _save(f"dupsafe_step_{it:04d}")

    _save("stage1_dupsafe_final")


def main():
    import argparse, sys
    from pathlib import Path
    # 一开始就声明 global，避免 SyntaxError
    global DATA, DVID, DIMG, DAUD, DTXT, CODI_ROOT, CODI_CKPT

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--codi_root", type=str, default=None)
    parser.add_argument("--codi_ckpt", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=300)
    parser.add_argument("--skip_iv_until", type=int, default=600)
    parser.add_argument("--img_take", type=int, default=8)
    parser.add_argument("--aud_seconds", type=float, default=8.0)
    parser.add_argument("--out_dim", type=int, default=512)
    parser.add_argument("--adapter", type=str, default="mlp", choices=["identity","mlp","xattn"])
    parser.add_argument("--adapter_depth", type=int, default=2)
    parser.add_argument("--adapter_width", type=int, default=1024)
    parser.add_argument("--adapter_heads", type=int, default=4)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--K_text", type=int, default=None)
    parser.add_argument("--K_image", type=int, default=None)
    parser.add_argument("--K_audio", type=int, default=None)
    parser.add_argument("--video_id", type=str, default=None)  # 现在不用，但保留避免报错

    args = parser.parse_args()

    # 覆盖数据 / CoDi 路径
    if args.data_root:
        DATA = Path(args.data_root).resolve()
        DVID, DIMG, DAUD, DTXT = DATA/"vedio", DATA/"image", DATA/"audio", DATA/"text"
        print("📁 数据目录:", DATA, flush=True)

    if args.codi_root:
        CODI_ROOT = Path(args.codi_root).resolve()
        if str(CODI_ROOT) not in sys.path:
            sys.path.insert(0, str(CODI_ROOT))
        print("📁 CoDi 根目录:", CODI_ROOT, flush=True)

    if args.codi_ckpt:
        CODI_CKPT = Path(args.codi_ckpt).resolve()
        print("📁 CoDi 检查点:", CODI_CKPT, flush=True)

    # 用 text 目录里的所有 .txt 当作样本池（COCO）
    txt_dir = DTXT
    all_stems = sorted(p.stem for p in txt_dir.glob("*.txt"))
    assert all_stems, f"在 {txt_dir} 里没找到任何 .txt，确认 data_root 填对了吗？"
    print(f"样本总数: {len(all_stems)}", flush=True)

    train_multi(all_stems, args)


if __name__ == "__main__":
    main()

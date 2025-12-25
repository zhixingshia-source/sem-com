# -*- coding: utf-8 -*-
"""
stage1_reworked.py
反塌缩强化版 Stage-1：
- 槽使用均衡 (slot_usage_loss)
- 列熵奖励 (col_entropy_loss)
- 槽级 InfoNCE (slot_infonce_loss) + 匈牙利匹配可选
- 槽丢弃 (slot_dropout) 强制多槽分工
- 继续使用你的 Sinkhorn-OT + 多样性/DPP/正交/非重叠项，但调度更激进

用法（示例）：
python stage1_reworked.py --steps 1500 --K 16 --rep_tokens 4 --lr 1e-4 --save_every 300 \
  --slot_dropout 0.25 --lambda_slot_usage 1.0 --lambda_col_entropy 0.5 --lambda_slot_infonce 0.5
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

# ----------------------- 路径/设备 -----------------------
ROOT = Path.cwd()
DATA = ROOT / "data"
DVID, DIMG, DAUD, DTXT = DATA/"vedio", DATA/"image", DATA/"audio", DATA/"text"

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

# ----------------------- 加载 CoDi（encoder 冻结） -----------------------
from core.models.model_module_infer import model_module as CoDiModule
_print_once = False
def _ensure_codi():
    global _print_once, net
    mod = CoDiModule(data_dir=str(CODI_CKPT), pth=["CoDi_encoders.pth"], fp16=False)
    net = mod.net if hasattr(mod, "net") else mod
    for p in net.parameters(): p.requires_grad_(False)
    net.to(device).eval()
    if not _print_once:
        print(f"✅ CoDi encoders @ {CODI_CKPT}")
        _print_once = True

# ----------------------- 简化版编码函数 -----------------------
@torch.no_grad()
def encode_text(list_of_str: List[str]) -> torch.Tensor:
    out = net.clip.encode_text_noproj(list_of_str)
    z = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    return z.float().detach().to(device)  # (B, L, D)

@torch.no_grad()
def encode_image(paths: List[str]) -> torch.Tensor:
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    xs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append((tfm(img).unsqueeze(0)*2-1))
    x = torch.cat(xs,0).to(device)
    out = net.clip.encode_vision_noproj(x)
    f = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    return f.float().detach()              # (N, L, D)

# （音频/视频与原脚本一致，可按需添加，这里主线展示 text/image）
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
        try: return p.read_text(errors="ignore").strip()
        except Exception: pass
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

# ----------------------- 语义提取器（复用你原来的结构骨架） -----------------------
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

class SemanticExtractor(nn.Module):
    """
    轻量 slot 抽取：线性K/V + 可学习查询Q + FFN
    注意：我们会记录 last_attn_w 用于损失
    """
    def __init__(self, in_dim, out_dim=512, K=16, q_res_w=0.1):
        super().__init__()
        self.K = K
        self.query = nn.Parameter(torch.randn(K, out_dim))
        nn.init.orthogonal_(self.query)
        self.key   = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.norm  = nn.LayerNorm(out_dim)
        self.ffn   = ResidualMLP(dim=out_dim, hidden=out_dim*2, depth=1)
        self.attn_tau = 1.0
        self.q_res_w  = q_res_w
        self.pool_stride = 1
        self.last_attn_w = None  # (1,K,S)
        self.last_attn_entropy = None

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (L,D) or (T,L,D) -> 展平成 (S,D)
        if seq.dim() == 3:
            seq = seq.reshape(seq.shape[0]*seq.shape[1], seq.shape[-1])
        S = seq.size(0)
        if self.pool_stride > 1:
            seq = seq[:: self.pool_stride]
            S = seq.size(0)

        Q = F.normalize(self.query, dim=-1)                      # (K,D)
        Kmat = F.normalize(self.key(seq), dim=-1)                # (S,D)
        Vmat = self.value(seq)                                   # (S,D)
        logits = (Kmat @ Q.t()) / max(self.attn_tau, 1e-6)       # (S,K)
        attn = logits.softmax(dim=0)                             # 列归一化：每个槽从所有位置“取料”
        # 转置成 (K,S)，再行归一化便于统计
        W = attn.t()                                             # (K,S)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-8)
        self.last_attn_w = W.unsqueeze(0).detach()               # (1,K,S)
        ent = -(W * (W.clamp_min(1e-8).log())).sum(dim=1).mean()
        self.last_attn_entropy = ent.detach()

        Z = W @ Vmat                                             # (K,D)
        if self.q_res_w > 0:
            Z = Z + self.q_res_w * self.query
        Z = self.norm(Z)
        Z = self.ffn(Z)
        return Z  # (K,D)

# ----------------------- 工具/损失 -----------------------
def _l2n(x): return F.normalize(x, dim=-1)

def diversity_loss(Z: torch.Tensor) -> torch.Tensor:
    Z = _l2n(Z); G = Z @ Z.t()
    eye = torch.eye(G.size(0), device=G.device)
    return ((G - eye) ** 2).mean()

def query_ortho_loss(mod: SemanticExtractor) -> torch.Tensor:
    q = _l2n(mod.query); G = q @ q.t()
    eye = torch.eye(G.size(0), device=G.device)
    return ((G - eye) ** 2).mean()

def attention_nonoverlap_loss(W: torch.Tensor) -> torch.Tensor:
    # W: (K,S) 行归一化；鼓励行间相关性低
    G = W @ W.t()
    eye = torch.eye(G.size(0), device=G.device)
    off = G * (1.0 - eye)
    return (off ** 2).mean()

def dpp_penalty(W: torch.Tensor, eps=1e-3) -> torch.Tensor:
    L = W @ W.t() + eps * torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    sign, logdet = torch.slogdet(L)
    return -logdet if sign > 0 else 0.0 * L.sum()

def dup_rate(mod) -> float:
    if (mod is None) or (getattr(mod, "last_attn_w", None) is None):
        return float("nan")
    # last_attn_w: (1, K, S)
    W = mod.last_attn_w[0]           # (K, S)
    top = W.argmax(-1)               # (K,)
    unique = top.unique().numel()
    K = top.numel()
    return 1.0 - unique / float(K)   # 0=不重复, ~1-1/K=几乎全撞同一列


def sinkhorn_ot_loss(A: torch.Tensor, B: torch.Tensor, eps: float = 0.03, iters: int = 30) -> torch.Tensor:
    assert A.dim()==2 and B.dim()==2 and A.size(0)==B.size(0)
    A = _l2n(A); B = _l2n(B)
    C = 1.0 - A @ B.t()                # (K,K)
    logK = -C / eps
    for _ in range(iters):
        logK = logK - torch.logsumexp(logK, dim=1, keepdim=True)
        logK = logK - torch.logsumexp(logK, dim=0, keepdim=True)
    P = logK.exp()
    return (P * C).sum() / A.size(0)

# ====== 新增：跨 batch 的均衡/覆盖 ======
def slot_usage_loss(extractors: List[SemanticExtractor]) -> Optional[torch.Tensor]:
    """
    让每个槽在“被用的总质量”上更均匀：
    usage_k = mean_b sum_s W_b[k, s]
    目标：usage 逼近 1/K（或均匀）。用方差或 KL 到均匀分布。
    """
    usages = []
    for m in extractors:
        if m is None or m.last_attn_w is None: continue
        W = m.last_attn_w[0]          # (K,S)
        usage = W.sum(dim=1)          # (K,)
        usage = usage / (usage.sum() + 1e-8)
        usages.append(usage)
    if not usages: return None
    loss = 0.0
    for u in usages:
        K = u.size(0)
        target = torch.full_like(u, 1.0 / K)
        loss += F.mse_loss(u, target)
    return loss / len(usages)

def col_entropy_loss(extractors: List[SemanticExtractor]) -> Optional[torch.Tensor]:
    """
    对每列（输入位置）的被槽选择分布 W[:, s] 做熵奖励（越大越好 -> 我们最小化负熵）。
    """
    ents = []
    for m in extractors:
        if m is None or m.last_attn_w is None: continue
        W = m.last_attn_w[0]          # (K,S)
        P = W / (W.sum(dim=0, keepdim=True) + 1e-8)  # 按列归一化：槽对该列的分布
        ent_col = -(P * (P.clamp_min(1e-8).log())).sum(dim=0).mean()
        ents.append(ent_col)
    if not ents: return None
    # 我们最小化 loss => 用负熵（越小越好）
    return -torch.stack(ents).mean()

# ====== 新增：槽级 InfoNCE ======
def slot_infonce_loss(Z1: torch.Tensor, Z2: torch.Tensor, tau: float = 0.07, use_hungarian: bool = False) -> torch.Tensor:
    """
    对同一模态两次视图的 (K,D) 槽做对比：
    - 正样本：同一槽（或匈牙利匹配到的槽）
    - 负样本：其他槽
    """
    Z1 = _l2n(Z1); Z2 = _l2n(Z2)  # (K,D)
    if use_hungarian:
        with torch.no_grad():
            C = 1.0 - (Z1 @ Z2.t())         # (K,K)
            # 简单匈牙利：用 torch linear_sum_assignment 替代（需要 scipy），这里用贪心近似
            perm = torch.argmin(C, dim=1)    # 近似：每行最小
    else:
        perm = torch.arange(Z1.size(0), device=Z1.device)

    sim = Z1 @ Z2.t() / max(tau, 1e-6)      # (K,K)
    pos = sim[torch.arange(sim.size(0), device=sim.device), perm]  # (K,)
    logsumexp = torch.logsumexp(sim, dim=1)                         # (K,)
    nce = -(pos - logsumexp).mean()
    return nce

def drop_slots(Z: Optional[torch.Tensor], p: float) -> Optional[torch.Tensor]:
    if Z is None: return None
    if p <= 0: return Z
    K = Z.size(0)
    mask = (torch.rand(K, device=Z.device) > p).float()  # keep prob = 1-p
    # 保证至少留一个
    if mask.sum() == 0:
        mask[random.randint(0, K-1)] = 1.0
    return Z * mask.unsqueeze(1)

# ----------------------- 训练一个样本两视图 -----------------------
def encode_once(st: str, take_img=8):
    txt = text_for(st)
    t_seq = encode_text([txt]) if txt else None            # (1,L,D)
    ims   = imgs_for(st, take=take_img)
    i_seq = encode_image(ims) if ims else None             # (N,L,D)
    return t_seq, i_seq

def extract_K(t_seq, i_seq, ET: Optional[SemanticExtractor], EI: Optional[SemanticExtractor]):
    zt = ET(t_seq.squeeze(0)) if (t_seq is not None and ET is not None) else None
    zi = EI(i_seq.squeeze(0)) if (i_seq is not None and EI is not None) else None
    return zt, zi

# ----------------------- 主训练循环 -----------------------
def train(args):
    _ensure_codi()

    # 动态探测维度
    # 动态探测维度（注意：不要覆盖全局的 DIMG/DTXT 路径变量）
    DTXT_DIM = encode_text(["probe"]).shape[-1]

    any_img_path = next(iter(glob.glob(str(DIMG / "*_*.jpg"))), None)
    assert any_img_path is not None, f"找不到图片样本：{DIMG}"
    DIMG_DIM = encode_image([any_img_path]).shape[-1]

    ET = SemanticExtractor(DTXT_DIM, out_dim=args.out_dim, K=args.K, q_res_w=0.05).to(device)
    EI = SemanticExtractor(DIMG_DIM, out_dim=args.out_dim, K=args.K, q_res_w=0.05).to(device)

    params = list(ET.parameters()) + list(EI.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    stems = [st for st in list_stems() if len(imgs_for(st))>0 and text_for(st)]
    assert stems, "没有同时具备 text+image 的样本"

    CKP = Path("snapshots"); CKP.mkdir(parents=True, exist_ok=True)

    it = 0
    while it < args.steps:
        it += 1
        st = random.choice(stems)

        # 两次视图
        t1,i1 = encode_once(st, take_img=args.img_take)
        zt1, zi1 = extract_K(t1, i1, ET, EI)
        t2,i2 = encode_once(st, take_img=args.img_take)
        zt2, zi2 = extract_K(t2, i2, ET, EI)

        # 槽丢弃（仅用于损失构造）
        zt1_d = drop_slots(zt1, args.slot_dropout); zt2_d = drop_slots(zt2, args.slot_dropout)
        zi1_d = drop_slots(zi1, args.slot_dropout); zi2_d = drop_slots(zi2, args.slot_dropout)

        loss = 0.0
        terms = {}

        # 1) 跨模态 Sinkhorn-OT（视图1）
        pairs = []
        if zt1_d is not None and zi1_d is not None: pairs.append((zt1_d, zi1_d))
        if pairs:
            xmodal = sum(sinkhorn_ot_loss(a,b) for a,b in pairs)/len(pairs)
            terms["xmodal"] = xmodal; loss = loss + args.lambda_xmodal * xmodal

        # 2) 槽级 InfoNCE（同模态跨视图）
        if zt1_d is not None and zt2_d is not None:
            lz = slot_infonce_loss(zt1_d, zt2_d, tau=0.07, use_hungarian=False)
            terms["slot_txt"] = lz; loss = loss + args.lambda_slot_infonce * lz
        if zi1_d is not None and zi2_d is not None:
            li = slot_infonce_loss(zi1_d, zi2_d, tau=0.07, use_hungarian=False)
            terms["slot_img"] = li; loss = loss + args.lambda_slot_infonce * li

        # 3) 多样性/非重叠/查询正交（单视图）
        for name, Z in [("div_t", zt1), ("div_i", zi1), ("div_t2", zt2), ("div_i2", zi2)]:
            if Z is not None:
                dv = diversity_loss(Z); terms[name] = dv
                loss = loss + args.lambda_div * dv

        for M in [ET, EI]:
            if M.last_attn_w is not None:
                W = M.last_attn_w[0]
                ano = attention_nonoverlap_loss(W); terms["ano"] = terms.get("ano", 0.0)+ano
                dpp = dpp_penalty(W);                terms["dpp"] = terms.get("dpp", 0.0)+dpp
                loss = loss + args.lambda_ano * ano + args.lambda_dpp * dpp
                ql = query_ortho_loss(M);            terms["q"] = terms.get("q", 0.0)+ql
                loss = loss + args.lambda_q * ql

        # 4) 批级均衡（关键）：槽使用均匀 + 列熵高
        su = slot_usage_loss([ET, EI])
        if su is not None:
            terms["slot_usage"] = su; loss = loss + args.lambda_slot_usage * su
        ce = col_entropy_loss([ET, EI])
        if ce is not None:
            terms["col_entropy"] = ce; loss = loss + args.lambda_col_entropy * ce

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # 简单调度：前期更强多样性/均衡，后期加大对齐
        if it in [1, 200, 600, 1000]:
            if it <= 200:
                for M in [ET, EI]: M.attn_tau = 1.5; M.q_res_w = 0.05
            elif it <= 600:
                for M in [ET, EI]: M.attn_tau = 1.2; M.q_res_w = 0.08
            else:
                for M in [ET, EI]: M.attn_tau = 1.0; M.q_res_w = 0.10

        if it % 50 == 0:
            dupT = dup_rate(ET); dupI = dup_rate(EI)
            print(f"[{it}/{args.steps}] loss={loss.item():.4f} ... (tau={ET.attn_tau:.2f})")
            print(f"dupT={dupT:.2f} dupI={dupI:.2f}")
            su_v = float(terms.get("slot_usage", 0.0))
            ce_v = float(terms.get("col_entropy", 0.0))
            print(f"[{it}/{args.steps}] loss={loss.item():.4f} "
                  f"x={float(terms.get('xmodal',0)):.3f} "
                  f"slot(txt)={float(terms.get('slot_txt',0)):.3f} "
                  f"slot(img)={float(terms.get('slot_img',0)):.3f} "
                  f"div={float(terms.get('div_t',0))+float(terms.get('div_i',0)):.3f} "
                  f"ano={float(terms.get('ano',0)):.3f} dpp={float(terms.get('dpp',0)):.3f} q={float(terms.get('q',0)):.3f} "
                  f"su={su_v:.3f} colH={-ce_v:.3f}  (tau={ET.attn_tau:.2f})")

        if it % args.save_every == 0 or it == args.steps:
            out = {
                "EXTRACTOR_T": ET.state_dict(),
                "EXTRACTOR_I": EI.state_dict(),
                "K_TOKENS": {"text": args.K, "image": args.K},
                "OUT_DIM": args.out_dim,
            }
            p = Path("snapshots")/f"reworked_step_{it:04d}.pth"
            torch.save(out, str(p))
            print("💾 saved", p)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--save_every", type=int, default=300)
    ap.add_argument("--out_dim", type=int, default=512)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--img_take", type=int, default=8)

    # 旧项（与原脚本同名）：
    ap.add_argument("--lambda_xmodal", type=float, default=0.30)
    ap.add_argument("--lambda_div", type=float, default=0.30)
    ap.add_argument("--lambda_ano", type=float, default=0.50)
    ap.add_argument("--lambda_q", type=float, default=1e-2)
    ap.add_argument("--lambda_dpp", type=float, default=1e-3)

    # 新增项：
    ap.add_argument("--slot_dropout", type=float, default=0.25)
    ap.add_argument("--lambda_slot_usage", type=float, default=1.0)
    ap.add_argument("--lambda_col_entropy", type=float, default=0.5)
    ap.add_argument("--lambda_slot_infonce", type=float, default=0.5)

    args = ap.parse_args()
    train(args)

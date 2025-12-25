#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kreps2gen_img2img_(11-26).py

整体流程（生成版）：
1) 读取 payload_root 下某个 stem 的 K-reps 语义载荷（K×512）。
2) 用训练好的 adapter (kreps2clip) 把扁平化后的向量映射到 CLIP embedding (D=768)。
3) 在 image_sem_db_(11-22).pt 里做最近邻检索，得到 top-k 图像 stem。
4) 在 top-k 里，从 gen_rank 开始往下找第一个磁盘上真实存在的参考图 ref_stem。
5) 从 text_root 里读出 query (原图) 和 ref_stem 的 caption，组装成 Stable Diffusion 的 prompt。
6) 用 SD img2img 以参考图 ref_img 为输入、prompt 为条件生成新图。
7) 保存：
   - 参考图：{stem}_ref_{ref_stem}.png
   - 生成图：{stem}_gen_from_{ref_stem}.png
   - 二图并排对比：{stem}_ref_gen_side_by_side.png

用法示例（注意括号要转义）：
python kreps2gen_img2img_\(11-26\).py \
  --payload_root data_rep_k32 \
  --stem coco_000002 \
  --adapter_ckpt runs/kreps2clip_exp1/adapter_clip_best.pth \
  --sem_db runs/sem_db/image_sem_db_\(11-22\).pt \
  --image_root data_coco/image \
  --text_root data_coco/text \
  --outdir runs/kreps2gen_img2img_11_26 \
  --topk 5 \
  --gen_rank 1 \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --sd_strength 0.6 \
  --sd_guidance_scale 7.5 \
  --sd_steps 50 \
  --sd_seed 42
"""

import argparse
import json
import base64
import zlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline


# ===========================
#  MLP Adapter 定义 & 加载
# ===========================

# ===========================
#  MLP Adapter 定义 & 加载（兼容训练时的结构）
# ===========================

# ===========================
#  MLP Adapter 定义 & 加载（根据 state_dict 自动推断结构）
# ===========================

class AdapterMLP(nn.Module):
    """简单包一层 nn.Sequential，方便 forward 调用。"""
    def __init__(self, seq: nn.Sequential):
        super().__init__()
        self.net = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_adapter_from_ckpt(ckpt_path: Path, device: torch.device):
    """
    直接看 checkpoint 里 Linear 的权重形状，反推结构：
    - 如果只有 2 个 Linear（常见：Linear -> ReLU -> Dropout -> Linear）
    - 如果有 3 个 Linear（Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear）

    这样就不用再瞎猜 n_layers=1 还是 n_layers=2 了，保证和训练时一模一样。
    """
    blob = torch.load(str(ckpt_path), map_location="cpu")

    # 先把 x_mean / x_std 读出来（有就用，没有就默认）
    if "config" in blob:
        cfg = blob["config"]
        x_mean = float(cfg.get("x_mean", 0.0))
        x_std = float(cfg.get("x_std", 1.0))
    else:
        x_mean = float(blob.get("x_mean", 0.0))
        x_std = float(blob.get("x_std", 1.0))

    # 取出真正的 state_dict
    state_dict = blob.get("state_dict", blob.get("model", {}))
    if not state_dict:
        raise RuntimeError(f"[kreps2clip-infer-11-22] empty state_dict in {ckpt_path}")

    # 找出所有 Linear 层的 index（net.0.weight, net.3.weight 这种）
    weight_keys = [k for k in state_dict.keys() if k.endswith(".weight")]
    lin_indices = sorted({int(k.split(".")[1]) for k in weight_keys})

    print(f"[kreps2clip-infer-11-22] adapter weight keys = {weight_keys}")
    print(f"[kreps2clip-infer-11-22] linear layer indices = {lin_indices}")

    if len(lin_indices) == 2:
        # 结构：Linear -> ReLU -> Dropout -> Linear
        i0, i1 = lin_indices
        w0 = state_dict[f"net.{i0}.weight"]
        w1 = state_dict[f"net.{i1}.weight"]

        d_in = w0.shape[1]
        hidden = w0.shape[0]
        d_out = w1.shape[0]

        seq = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(hidden, d_out),
        )

    elif len(lin_indices) == 3:
        # 备用情况：Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
        i0, i1, i2 = lin_indices
        w0 = state_dict[f"net.{i0}.weight"]
        w1 = state_dict[f"net.{i1}.weight"]
        w2 = state_dict[f"net.{i2}.weight"]

        d_in = w0.shape[1]
        h1 = w0.shape[0]
        h2 = w1.shape[0]
        d_out = w2.shape[0]

        seq = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(h2, d_out),
        )
    else:
        raise RuntimeError(
            f"[kreps2clip-infer-11-22] Unexpected number of linear layers in adapter: "
            f"{len(lin_indices)}, keys={weight_keys}"
        )

    adapter = AdapterMLP(seq)

    # 严格加载，让结构和权重一一对应
    adapter.load_state_dict(state_dict, strict=True)

    adapter.to(device)
    adapter.eval()

    print(f"[kreps2clip-infer-11-22] adapter inferred: d_in={seq[0].in_features}, "
          f"hidden={seq[0].out_features}, "
          f"d_out={seq[-1].out_features}, "
          f"#linears={len(lin_indices)}")
    print(f"[kreps2clip-infer-11-22] x_mean={x_mean:.6f}, x_std={x_std:.6f}")

    d_in = seq[0].in_features
    d_out = seq[-1].out_features
    return adapter, d_in, d_out, x_mean, x_std

# ===========================
#  K-reps payload 读取
# ===========================
def load_payload_kreps(payload_path: Path, max_tokens: int = 32):
    """
    统一读取 K-reps payload，兼容三种情况：
    1) 之前自己存的 float 矩阵：
       - "reps" / "k_reps" / "kreps" / "reps_f"
    2) 简单量化：
       - ("reps_q", "scales") / ("k_reps_q", "k_scales")
    3) KToken-Reps：
       keys = ['kind','version','n_tokens','dim','dtype','encoding','scales','data','metadata']
       - data: 压缩后的 K×dim（通常 int8），可能是 base64(+zlib) 字符串
       - scales: 每 token 的 scale，长度 = K 或 1（可能是 list，也可能是 float list）
    """
    with payload_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    print(f"[kreps2clip-infer-11-22] payload keys = {list(obj.keys())}")

    reps = None  # 最终要得到的 float32 [K, dim]

    # ---------- (A) KToken-Reps 分支 ----------
    if all(k in obj for k in ["data", "scales", "n_tokens", "dim"]):
        n_tokens = int(obj["n_tokens"])
        dim = int(obj["dim"])
        dtype_str = str(obj.get("dtype", "int8")).lower()
        enc_str = str(obj.get("encoding", "")).lower()
        print(
            f"[kreps2clip-infer-11-22] detect KToken-Reps: "
            f"n_tokens={n_tokens}, dim={dim}, dtype={dtype_str}, encoding={enc_str}"
        )

        # ---- 1) 先解码 data ----
        raw_data = obj["data"]
        if isinstance(raw_data, str):
            # 字符串 → base64 → (可选)zlib → bytes
            try:
                # 优先按 encoding 走；如果没写，也默认尝试 base64
                if "base64" in enc_str or enc_str == "" or enc_str == "zlib":
                    b = base64.b64decode(raw_data)
                else:
                    # 万一 encoding 里没写 base64，就直接尝试
                    b = base64.b64decode(raw_data)
            except Exception as e:
                raise ValueError(f"KToken-Reps: failed to base64-decode 'data': {e}")

            if "zlib" in enc_str:
                try:
                    b = zlib.decompress(b)
                except Exception as e:
                    raise ValueError(f"KToken-Reps: failed to zlib-decompress 'data': {e}")

            # bytes -> 数组
            if dtype_str in ("int8", "i1", "byte"):
                np_dtype = np.int8
            elif dtype_str in ("float32", "f4", "float"):
                np_dtype = np.float32
            elif dtype_str in ("float16", "f2", "half"):
                np_dtype = np.float16
            else:
                # 不认识就按 int8 处理
                np_dtype = np.int8

            q = np.frombuffer(b, dtype=np_dtype).astype(np.float32)
            if q.size != n_tokens * dim:
                raise ValueError(
                    f"KToken-Reps data length mismatch after decode: "
                    f"got {q.size}, expected {n_tokens}*{dim}={n_tokens*dim}"
                )
            q = q.reshape(n_tokens, dim)
        else:
            # data 本身已经是 list[list[...]]
            q = np.asarray(raw_data, dtype=np.float32)
            if q.ndim == 1:
                if q.size != n_tokens * dim:
                    raise ValueError(
                        f"KToken-Reps data length mismatch: got {q.size}, "
                        f"expected {n_tokens}*{dim}={n_tokens*dim}"
                    )
                q = q.reshape(n_tokens, dim)
            elif q.ndim == 2 and q.shape != (n_tokens, dim):
                print(
                    f"[kreps2clip-infer-11-22] WARNING: data shape {q.shape} "
                    f"!= (n_tokens, dim)=({n_tokens},{dim})"
                )

        # ---- 2) 再处理 scales ----
        raw_scales = obj["scales"]
        if isinstance(raw_scales, str):
            # scales 也可能被压成 base64(+zlib)
            try:
                b = base64.b64decode(raw_scales)
            except Exception as e:
                raise ValueError(f"KToken-Reps: failed to base64-decode 'scales': {e}")
            if "zlib" in enc_str:
                try:
                    b = zlib.decompress(b)
                except Exception as e:
                    raise ValueError(f"KToken-Reps: failed to zlib-decompress 'scales': {e}")
            s = np.frombuffer(b, dtype=np.float32)
        else:
            s = np.asarray(raw_scales, dtype=np.float32)

        if s.ndim == 1:
            if s.size not in (1, n_tokens):
                raise ValueError(
                    f"KToken-Reps scales length mismatch: got {s.size}, expected 1 or {n_tokens}"
                )
            if s.size == 1:
                s = np.repeat(s, n_tokens)
            s = s[:, None]  # [K,1]
        elif s.ndim == 2:
            if s.shape[0] != n_tokens:
                raise ValueError(
                    f"KToken-Reps scales shape mismatch: {s.shape} vs n_tokens={n_tokens}"
                )
            if s.shape[1] != 1:
                # 万一弄成 [K,dim] 了，就按 L2 折成 [K,1]
                if s.shape[1] == dim:
                    s = np.linalg.norm(s, axis=1, keepdims=True)
                else:
                    raise ValueError(
                        f"KToken-Reps scales second dim={s.shape[1]} not 1 or dim={dim}"
                    )
        else:
            raise ValueError(f"KToken-Reps scales has invalid ndim={s.ndim}, expected 1 or 2.")

        reps = q * s  # 反量化，得到 float32 [K, dim]
        print(f"[kreps2clip-infer-11-22] KToken-Reps decoded reps shape = {reps.shape}")

    # ---------- (B) 普通 float 矩阵 ----------
    if reps is None:
        float_keys = ["reps", "k_reps", "kreps", "reps_f"]
        for k in float_keys:
            if k in obj:
                reps = np.asarray(obj[k], dtype=np.float32)
                print(f"[kreps2clip-infer-11-22] use float reps key = '{k}', shape={reps.shape}")
                break

    # ---------- (C) 简单量化 reps_q + scales ----------
    if reps is None:
        quant_pairs = [
            ("reps_q", "scales"),
            ("k_reps_q", "k_scales"),
        ]
        for q_key, s_key in quant_pairs:
            if q_key in obj and s_key in obj:
                q = np.asarray(obj[q_key], dtype=np.float32)
                s = np.asarray(obj[s_key], dtype=np.float32)
                if s.ndim == 1:
                    s = s[:, None]
                reps = q * s
                print(
                    f"[kreps2clip-infer-11-22] use quantized reps: "
                    f"q_key='{q_key}', s_key='{s_key}', q.shape={q.shape}, s.shape={s.shape}"
                )
                break

    # ---------- (D) 自动兜底 ----------
    if reps is None:
        for k, v in obj.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                try:
                    arr = np.asarray(v, dtype=np.float32)
                except Exception:
                    continue
                if arr.ndim == 2 and arr.shape[1] in (256, 512, 768, 1024):
                    reps = arr
                    print(
                        f"[kreps2clip-infer-11-22] auto-detected reps key='{k}', shape={arr.shape}"
                    )
                    break

    if reps is None:
        raise KeyError(
            f"Unrecognized payload format in {payload_path}. "
            f"Got keys={list(obj.keys())}. "
            f"Cannot decode reps."
        )

    # ---------- 截断到前 max_tokens ----------
    K_raw, d_token = reps.shape
    if max_tokens is not None and K_raw > max_tokens:
        reps = reps[:max_tokens, :]
    K_eff = reps.shape[0]

    x_flat = reps.reshape(-1).astype(np.float32)  # [K_eff * d_token]
    return x_flat, K_raw, K_eff, d_token


# ===========================
#  语义数据库 & 文本/图像 IO
# ===========================

def load_sem_db(sem_db_path: Path):
    """
    读取 image_sem_db_(11-22).pt:
      keys = ['stems', 'embeds', 'clip_model']
    返回：
      embs: Tensor [N, D]
      stems: list[str]
    """
    db = torch.load(str(sem_db_path), map_location="cpu")
    keys = list(db.keys())
    print(f"[kreps2clip-infer-11-22] sem_db keys = {keys}")

    if "embeds" not in db or "stems" not in db:
        raise KeyError(
            f"Expect 'embeds' and 'stems' in sem_db. Got keys={keys}"
        )

    embs = db["embeds"]
    stems = db["stems"]

    if isinstance(embs, np.ndarray):
        embs = torch.from_numpy(embs)
    embs = embs.float()

    # stems 一般是 list[str]，保险一点转一下
    stems = list(stems)

    print(f"[kreps2clip-infer-11-22] sem_db: N={embs.shape[0]}, D={embs.shape[1]}")
    return embs, stems


def find_image_path(image_root: Path, stem: str) -> Path:
    """
    在 image_root 下找对应 stem 的图像。
    1) 先尝试严格匹配：{stem}.jpg / .jpeg / .png
    2) 如果没有，再用通配匹配：{stem}_*.jpg / .jpeg / .png
       适配你这种 ccoo_000002_0001.jpg 的命名。
    """
    # 1) 精确匹配
    candidates = [
        image_root / f"{stem}.jpg",
        image_root / f"{stem}.jpeg",
        image_root / f"{stem}.png",
    ]
    for p in candidates:
        if p.exists():
            return p

    # 2) 通配：coco_000002_0001.jpg 这类
    pattern_candidates = [
        f"{stem}_*.jpg",
        f"{stem}_*.jpeg",
        f"{stem}_*.png",
    ]
    for pat in pattern_candidates:
        matches = sorted(image_root.glob(pat))
        if matches:
            print(f"[kreps2clip-infer-11-22] fallback glob for stem={stem}, pattern={pat} -> {matches[0]}")
            return matches[0]

    # 3) 还是没找到就报错
    tried = [str(c) for c in candidates] + [
        str(image_root / pat) for pat in pattern_candidates
    ]
    raise FileNotFoundError(
        f"cannot find image for stem={stem} under {image_root}. "
        f"tried exact + glob: {tried}"
    )


def load_caption(text_root: Path, stem: str) -> str:
    """
    从 text_root/{stem}.txt 读 caption。
    多行则用 ' | ' 拼起来。
    """
    path = text_root / f"{stem}.txt"
    if not path.exists():
        return "<no_caption_file>"

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            return "<empty_caption_file>"
        return " | ".join(lines)
    except Exception as e:
        return f"<error_reading_caption: {e}>"


def make_side_by_side(imgs: List[Image.Image], padding: int = 8,
                      bg_color=(255, 255, 255)) -> Image.Image:
    """
    把多张图横向拼在一起。
    """
    if len(imgs) == 0:
        raise ValueError("make_side_by_side got empty imgs list")

    widths, heights = zip(*(im.size for im in imgs))
    max_h = max(heights)
    total_w = sum(widths) + padding * (len(imgs) - 1)

    canvas = Image.new("RGB", (total_w, max_h), bg_color)
    x = 0
    for im in imgs:
        y = (max_h - im.size[1]) // 2
        canvas.paste(im, (x, y))
        x += im.size[0] + padding
    return canvas


# ===========================
#  主流程
# ===========================

def main():
    parser = argparse.ArgumentParser("kreps2gen-img2img-11-26")
    parser.add_argument("--payload_root", type=str, required=True)
    parser.add_argument("--stem", type=str, required=True,
                        help="e.g. coco_000002")

    parser.add_argument("--adapter_ckpt", type=str, required=True,
                        help="runs/kreps2clip_exp1/adapter_clip_best.pth")
    parser.add_argument("--sem_db", type=str, required=True,
                        help="runs/sem_db/image_sem_db_(11-22).pt")

    parser.add_argument("--image_root", type=str, required=True,
                        help="e.g. data_coco/image")
    parser.add_argument("--text_root", type=str, required=True,
                        help="e.g. data_coco/text")
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--topk", type=int, default=5,
                        help="top-k retrieval for candidate reference images")
    parser.add_argument("--gen_rank", type=int, default=1,
                        help="preferred rank (1-based) in top-k to use as ref image; "
                             "如果该 rank 没有图，会尝试其他 rank")

    parser.add_argument("--sd_model", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--sd_strength", type=float, default=0.6)
    parser.add_argument("--sd_guidance_scale", type=float, default=7.5)
    parser.add_argument("--sd_steps", type=int, default=50)
    parser.add_argument("--sd_seed", type=int, default=42)

    parser.add_argument("--max_tokens", type=int, default=32,
                        help="使用前 K 个语义 token（必须和训练 adapter 时一致）")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    payload_root = (root / args.payload_root).resolve()
    adapter_ckpt = (root / args.adapter_ckpt).resolve()
    sem_db_path = (root / args.sem_db).resolve()
    image_root = (root / args.image_root).resolve()
    text_root = (root / args.text_root).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    payload_path = payload_root / f"{args.stem}_payload.json"

    print(f"[kreps2clip-infer-11-22] payload = {payload_path}")
    print(f"[kreps2clip-infer-11-22] adapter_ckpt = {adapter_ckpt}")
    print(f"[kreps2clip-infer-11-22] sem_db = {sem_db_path}")
    print(f"[kreps2clip-infer-11-22] image_root = {image_root}")
    print(f"[kreps2clip-infer-11-22] text_root = {text_root}")
    print(f"[kreps2clip-infer-11-22] outdir = {outdir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[kreps2clip-infer-11-22] device = {device}")

    # 1) 加载 adapter
    adapter, d_in, d_out, x_mean, x_std = build_adapter_from_ckpt(adapter_ckpt, device)

    # 2) 加载 sem_db
    db_embs, db_stems = load_sem_db(sem_db_path)
    db_embs = F.normalize(db_embs, dim=-1)  # [N, D] on CPU

    # 3) 加载 payload
    if not payload_path.exists():
        raise FileNotFoundError(f"payload not found: {payload_path}")
    x_flat, K_raw, K_eff, d_token = load_payload_kreps(payload_path, max_tokens=args.max_tokens)
    print(f"[kreps2clip-infer-11-22] payload reps shape = ({K_eff}, {d_token})")
    print(f"[kreps2clip-infer-11-22] K_raw={K_raw}, K_eff(used)={K_eff}, d_in_expected={d_in}")

    if x_flat.shape[0] != d_in:
        raise ValueError(
            f"flattened payload dim mismatch: got {x_flat.shape[0]}, expected {d_in}. "
            f"Check max_tokens and token dim."
        )

    x = torch.from_numpy(x_flat).to(device)
    # 标准化，同训练阶段
    x = (x - x_mean) / (x_std + 1e-8)

    with torch.no_grad():
        z = adapter(x)  # [D]
        z = F.normalize(z, dim=-1)  # [D]

    # 4) 和 sem_db 做相似度，top-k
    sims = torch.matmul(z.cpu(), db_embs.T).squeeze(0)  # [N]
    topk = min(args.topk, sims.numel())
    vals, idxs = torch.topk(sims, k=topk, dim=0)

    top_stems = [db_stems[i] for i in idxs.tolist()]
    print("[kreps2clip-infer-11-22] top-k retrieval:")
    for rank, (score, stem) in enumerate(zip(vals.tolist(), top_stems), start=1):
        print(f"[kreps2clip-infer-11-22]   #{rank}: stem={stem}  sim={score:.4f}")

    # 5) 从 top-k 中选择一个真实存在的参考图
    chosen_stem = None
    chosen_path = None

    candidate_indices = list(range(len(top_stems)))
    preferred_idx = args.gen_rank - 1
    if 0 <= preferred_idx < len(top_stems):
        candidate_indices = [preferred_idx] + [i for i in candidate_indices if i != preferred_idx]

    for i in candidate_indices:
        stem_i = top_stems[i]
        try:
            img_path_i = find_image_path(image_root, stem_i)
            chosen_stem = stem_i
            chosen_path = img_path_i
            print(f"[kreps2clip-infer-11-22] choose ref_stem={stem_i} at rank={i+1}, img={img_path_i}")
            break
        except FileNotFoundError as e:
            print(f"[kreps2clip-infer-11-22] skip rank {i+1} stem={stem_i}: {e}")

    # 如果 top-k 里面都没找到图片，就退回用原图作为参考图
    if chosen_path is None:
        try:
            ref_stem = args.stem
            ref_img_path = find_image_path(image_root, args.stem)
            print(
                f"[kreps2clip-infer-11-22] no usable image in top-k, "
                f"fallback to original image: stem={ref_stem}, img={ref_img_path}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "[kreps2clip-infer-11-22] no usable reference image found in top-k, "
                "and original image is also missing. "
                "check your image_root or rebuild sem_db."
            )
    else:
        ref_stem = chosen_stem
        ref_img_path = chosen_path


    # 原图（ground truth）也顺便找一下（能找到的话）
    try:
        orig_img_path = find_image_path(image_root, args.stem)
    except FileNotFoundError:
        orig_img_path = None

    # 6) 读取 query caption & ref caption，组 prompt
    query_caption = load_caption(text_root, args.stem)
    ref_caption = load_caption(text_root, ref_stem)

    print(f"[kreps2clip-infer-11-22] query_caption ({args.stem}): {query_caption}")
    print(f"[kreps2clip-infer-11-22] ref_caption   ({ref_stem}): {ref_caption}")

    caption_pieces = []
    if query_caption and not query_caption.startswith("<"):
        caption_pieces.append(query_caption)
    if ref_caption and not ref_caption.startswith("<") and ref_stem != args.stem:
        caption_pieces.append(ref_caption)

    if caption_pieces:
        prompt = "High quality photo. " + " ".join(caption_pieces)
    else:
        prompt = "High quality photo of the same scene."

    print(f"[kreps2clip-infer-11-22] SD prompt = {prompt}")

    # 7) 载入 SD img2img pipeline
    print(f"[kreps2clip-infer-11-22] loading SD img2img pipeline: {args.sd_model}")
    sd_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=sd_dtype,
    ).to(device)

    # 8) 读参考图，跑 img2img
    ref_img = Image.open(ref_img_path).convert("RGB")

    # 可选：控制生成随机性
    generator = None
    if args.sd_seed is not None and args.sd_seed >= 0:
        generator = torch.Generator(device=device).manual_seed(args.sd_seed)

    print("[kreps2clip-infer-11-22] running img2img generation ...")
    result = pipe(
        prompt=prompt,
        image=ref_img,
        strength=args.sd_strength,
        guidance_scale=args.sd_guidance_scale,
        num_inference_steps=args.sd_steps,
        generator=generator,
    )
    gen_img = result.images[0]

    # 9) 保存单图 + 对比图
    ref_out = outdir / f"{args.stem}_ref_{ref_stem}.png"
    gen_out = outdir / f"{args.stem}_gen_from_{ref_stem}.png"
    ref_img.save(ref_out)
    gen_img.save(gen_out)

    print(f"[kreps2clip-infer-11-22] saved ref image    -> {ref_out}")
    print(f"[kreps2clip-infer-11-22] saved gen image    -> {gen_out}")

    side_imgs: List[Image.Image] = [ref_img, gen_img]
    if orig_img_path is not None:
        orig_img = Image.open(orig_img_path).convert("RGB")
        side_imgs = [orig_img, ref_img, gen_img]

    side = make_side_by_side(side_imgs, padding=8)
    side_out = outdir / f"{args.stem}_ref_gen_side_by_side.png"
    side.save(side_out)
    print(f"[kreps2clip-infer-11-22] saved side-by-side -> {side_out}")


if __name__ == "__main__":
    main()

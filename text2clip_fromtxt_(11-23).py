#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
text2clip_fromtxt_(11-23).py

- 从 data/text/{stem}.txt 读 caption 作为查询
- 用 CLIP 文本编码成 embedding
- 在 image_sem_db_(11-22).pt 里做 TEXT→IMAGE 检索
- 对每个 top-k 结果再去 data/text/{stem}.txt 读 caption 打印出来
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def load_caption_from_txt(text_root: Path, stem: str) -> str:
    txt_path = text_root / f"{stem}.txt"
    if not txt_path.exists():
        return "<no_caption_file>"

    try:
        with txt_path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            return "<empty_caption_file>"
        # 你可以改成只取第一行，这里把所有非空行用 " | " 拼起来
        return " | ".join(lines)
    except Exception as e:
        return f"<error_reading_caption: {e}>"


def load_db(sem_db_path: Path):
    db = torch.load(str(sem_db_path), map_location="cpu")
    print(f"[text-from-txt-11-23] db keys = {list(db.keys())}")

    # 尝试多种可能的 key 名

    emb_key_candidates = ["embeds", "embs", "image_embs", "clip_embs", "reps", "image_reps"]
    stem_key_candidates = ["stems", "image_stems", "names"]

    emb_key = None
    for k in emb_key_candidates:
        if k in db:
            emb_key = k
            break

    stem_key = None
    for k in stem_key_candidates:
        if k in db:
            stem_key = k
            break

    if emb_key is None or stem_key is None:
        raise KeyError(
            f"Cannot find embedding/stem keys in db. "
            f"Got keys={list(db.keys())}, "
            f"tried emb_keys={emb_key_candidates}, stem_keys={stem_key_candidates}"
        )

    embs = db[emb_key]
    stems = db[stem_key]
    return embs, stems


def main():
    parser = argparse.ArgumentParser("text-from-txt-11-23")
    parser.add_argument("--stem", type=str, required=True,
                        help="e.g. coco_000002")
    parser.add_argument("--text_root", type=str, required=True,
                        help="root dir of txt captions, e.g. data/text")
    parser.add_argument("--sem_db", type=str, required=True,
                        help="path to image_sem_db_(11-22).pt")
    parser.add_argument("--clip_model", type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    text_root = (root / args.text_root).resolve()
    sem_db_path = (root / args.sem_db).resolve()

    print(f"[text-from-txt-11-23] stem = {args.stem}")
    print(f"[text-from-txt-11-23] text_root = {text_root}")
    print(f"[text-from-txt-11-23] sem_db = {sem_db_path}")
    print(f"[text-from-txt-11-23] topk = {args.topk}")
    print("")

    # 1) 读取查询 caption
    print("[text-from-txt-11-23] ====== QUERY CAPTION FROM TXT ======")
    query_caption = load_caption_from_txt(text_root, args.stem)
    print(f"[text-from-txt-11-23] {query_caption}")
    print("")

    # 2) 载入语义数据库（自动识别 embs/stems 的 key）
    embs, stems = load_db(sem_db_path)
    print(
        f"[text-from-txt-11-23] sem_db loaded: N={embs.shape[0]}, D={embs.shape[1]}"
    )

    # 3) 载入 CLIP 文本编码器
    print("[text-from-txt-11-23] loading CLIP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()

    # 4) 编码查询文本
    with torch.no_grad():
        inputs = processor(
            text=[query_caption],
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_feat = model.get_text_features(**inputs)  # [1, D]
        text_feat = F.normalize(text_feat, dim=-1)

    # 5) 归一化 DB embedding 并做相似度
    embs = F.normalize(embs, dim=-1)
    sims = torch.matmul(text_feat.cpu(), embs.T).squeeze(0)  # [N]

    topk = min(args.topk, sims.numel())
    vals, idxs = torch.topk(sims, k=topk, dim=0)

    print("\n[text-from-txt-11-23] ====== TOP-K TEXT→IMAGE RETRIEVAL ======")
    for rank, (score, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        stem = stems[idx]
        cap = load_caption_from_txt(text_root, stem)
        print(f"[text-from-txt-11-23] #{rank}: stem={stem:>9}   sim={score:.4f}")
        print(f"[text-from-txt-11-23]     caption: {cap}")
        print("")


if __name__ == "__main__":
    main()

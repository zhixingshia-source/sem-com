#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# text2clip_retrieval_(11-23).py
#   手工输入一句 caption，用 CLIP 做 text→image 检索，
#   并在有 text_root 时，把 top-k 的 caption 一起打印出来。

import argparse
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor


def load_db(path: Path):
    print(f"[text-retrieval-11-23] sem_db = {path}")
    db = torch.load(str(path), map_location="cpu")

    keys = list(db.keys())
    print(f"[text-retrieval-11-23] db keys = {keys}")

    emb_keys = ["embs", "image_embs", "clip_embs", "reps", "image_reps", "embeds"]
    stem_keys = ["stems", "image_stems", "names"]

    emb_key = next((k for k in emb_keys if k in db), None)
    stem_key = next((k for k in stem_keys if k in db), None)

    if emb_key is None or stem_key is None:
        raise KeyError(
            f"Cannot find embedding/stem keys in db. "
            f"Got keys={keys}, tried emb_keys={emb_keys}, stem_keys={stem_keys}"
        )

    embs = torch.as_tensor(db[emb_key], dtype=torch.float32)
    stems = list(db[stem_key])

    if embs.ndim != 2:
        raise ValueError(f"embs should be 2D [N, D], got shape={embs.shape}")

    N, D = embs.shape
    print(f"[text-retrieval-11-23] sem_db loaded: N={N}, D={D}")
    return embs, stems, db.get("clip_model", None)


def load_caption_from_txt(text_root: Path, stem: str) -> str:
    """
    从 data/text/coco_xxxxx.txt 读 caption（取第一行非空）。
    """
    txt_path = text_root / f"{stem}.txt"
    if not txt_path.exists():
        return "<caption_txt_not_found>"

    try:
        with txt_path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
    except Exception as e:
        return f"<caption_read_error: {e}>"

    lines = [ln for ln in lines if ln]
    if not lines:
        return "<empty_caption>"

    # 这里取第一行，如果你想要全部，可以改成 " ".join(lines)
    return lines[0]


def main():
    parser = argparse.ArgumentParser(
        description="Text → CLIP → image retrieval (11-23, print captions if possible)"
    )
    parser.add_argument(
        "--caption",
        type=str,
        required=True,
        help="query caption string",
    )
    parser.add_argument(
        "--sem_db",
        type=str,
        required=True,
        help="path to semantic db .pt (with stems + CLIP embeddings)",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HF id of CLIP model",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="top-k retrieved images to show",
    )
    parser.add_argument(
        "--text_root",
        type=str,
        default=None,
        help="optional: root of text files (e.g. data/text) to print captions for retrieved stems",
    )

    args = parser.parse_args()

    sem_db_path = Path(args.sem_db).expanduser().resolve()
    text_root = Path(args.text_root).expanduser().resolve() if args.text_root else None

    print(f"[text-retrieval-11-23] caption_query = {args.caption!r}")
    print(f"[text-retrieval-11-23] topk = {args.topk}")
    if text_root is not None:
        print(f"[text-retrieval-11-23] text_root = {text_root}")

    # 1) 加载 DB
    db_embs, stems, db_clip_model = load_db(sem_db_path)

    # 2) 加载 CLIP
    clip_model_name = args.clip_model
    print(f"[text-retrieval-11-23] loading CLIP model: {clip_model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    # 3) 编码 query 文本
    inputs = processor(
        text=[args.caption],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)  # [1, D]

    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # 4) 归一化 DB，算相似度
    db_embs = db_embs.to(device)
    db_embs = db_embs / db_embs.norm(dim=-1, keepdim=True)

    sims = (text_emb @ db_embs.T).squeeze(0)  # [N]
    topk = min(args.topk, sims.numel())
    sim_vals, sim_idx = torch.topk(sims, k=topk, dim=0)

    print("\n[text-retrieval-11-23] ====== TOP-K TEXT → IMAGE RETRIEVAL ======")
    for rank in range(topk):
        idx = sim_idx[rank].item()
        stem = stems[idx]
        sim_val = sim_vals[rank].item()
        print(f"[text-retrieval-11-23] #{rank+1}: stem={stem:<10} sim={sim_val:.4f}")

        # 如果给了 text_root，就尝试打印该 stem 对应的 caption
        if text_root is not None:
            cap = load_caption_from_txt(text_root, stem)
            print(f"[text-retrieval-11-23]     caption: {cap}")
        else:
            print(f"[text-retrieval-11-23]     caption: <no_caption_available (no text_root)>")

        print("")


if __name__ == "__main__":
    main()

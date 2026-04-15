#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch CLIP text-to-image retrieval baseline.

This baseline reads captions from text_root/{stem}.txt, encodes them with the
same CLIP model used to build the image semantic DB, and evaluates retrieval
against sem_db["stems"] / sem_db["embeds"].

Default test split:
  text_root/*.txt intersected with stems contained in sem_db.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def log(*items):
    print("[clip-text-baseline-eval]", *items, flush=True)


def load_caption(text_root: Path, stem: str, mode: str = "first") -> str:
    path = text_root / f"{stem}.txt"
    if not path.exists():
        return ""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    if mode == "all":
        return " ".join(lines)
    return lines[0]


def load_stems(text_root: Path, stems_file: Optional[Path], db_stems: List[str]) -> List[str]:
    db_set = set(db_stems)
    if stems_file is not None:
        stems = []
        for line in stems_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            stems.append(s)
    else:
        stems = sorted(p.stem for p in text_root.glob("*.txt"))
    stems = [s for s in stems if s in db_set and (text_root / f"{s}.txt").exists()]
    return stems


def average_precision_single_relevant(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def encode_text_batch(
    captions: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        z = model.get_text_features(**inputs)
        z = F.normalize(z.float(), dim=-1)
    return z.cpu()


def main():
    parser = argparse.ArgumentParser("batch CLIP text retrieval baseline")
    parser.add_argument("--text_root", type=str, required=True, help="Directory containing {stem}.txt captions")
    parser.add_argument("--sem_db", type=str, required=True, help="image semantic DB .pt")
    parser.add_argument("--outdir", type=str, required=True, help="Directory for metrics.json and per_sample_results.jsonl")
    parser.add_argument("--clip_model", type=str, default=None, help="CLIP model name/path. Defaults to sem_db['clip_model'] if present")
    parser.add_argument("--stems_file", type=str, default=None, help="Optional text file, one stem per line")
    parser.add_argument("--caption_mode", type=str, default="first", choices=["first", "all"], help="Use first non-empty caption line or concatenate all lines")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10, help="Store top-k predictions per sample")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples, 0 means all")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    text_root = Path(args.text_root).resolve()
    sem_db_path = Path(args.sem_db).resolve()
    outdir = Path(args.outdir).resolve()
    stems_file = Path(args.stems_file).resolve() if args.stems_file else None
    outdir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    db = torch.load(str(sem_db_path), map_location="cpu")
    db_stems = list(db["stems"])
    embeds_db = F.normalize(db["embeds"].float(), dim=-1)
    clip_model_name = args.clip_model or db.get("clip_model", "openai/clip-vit-large-patch14")

    stems = load_stems(text_root, stems_file, db_stems)
    if args.limit and args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError("No evaluable stems found. Check text_root, stems_file, and sem_db.")

    log(f"text_root = {text_root}")
    log(f"sem_db = {sem_db_path}")
    log(f"outdir = {outdir}")
    log(f"clip_model = {clip_model_name}")
    log(f"device = {device}")
    log(f"num_eval_samples = {len(stems)}")
    log(f"caption_mode = {args.caption_mode}")
    log(f"topk = {args.topk}")

    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    stem_to_db_index: Dict[str, int] = {s: i for i, s in enumerate(db_stems)}
    result_path = outdir / "per_sample_results.jsonl"
    metrics_path = outdir / "metrics.json"

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    ap_sum = 0.0
    evaluated = 0
    skipped = 0

    with result_path.open("w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(stems), args.batch_size), desc="clip-text-eval"):
            batch_stems = stems[start : start + args.batch_size]
            captions = [load_caption(text_root, s, mode=args.caption_mode) for s in batch_stems]
            valid = [(s, c) for s, c in zip(batch_stems, captions) if c]
            skipped += len(batch_stems) - len(valid)
            if not valid:
                continue

            valid_stems = [x[0] for x in valid]
            valid_caps = [x[1] for x in valid]
            text_embs = encode_text_batch(valid_caps, model, processor, device)
            sims_batch = text_embs @ embeds_db.T

            for row_idx, stem in enumerate(valid_stems):
                gt_idx = stem_to_db_index.get(stem)
                if gt_idx is None:
                    skipped += 1
                    continue

                sims = sims_batch[row_idx]
                gt_sim = float(sims[gt_idx].item())
                rank = int((sims > sims[gt_idx]).sum().item()) + 1

                k = min(max(args.topk, 10), sims.numel())
                vals, idxs = torch.topk(sims, k=k)
                idx_list = idxs.tolist()
                val_list = vals.tolist()

                top_predictions = [
                    {
                        "rank": i + 1,
                        "stem": db_stems[idx],
                        "sim": float(val),
                    }
                    for i, (idx, val) in enumerate(zip(idx_list[: args.topk], val_list[: args.topk]))
                ]

                hit1 = rank <= 1
                hit5 = rank <= 5
                hit10 = rank <= 10
                ap = average_precision_single_relevant(rank)

                hits_at_1 += int(hit1)
                hits_at_5 += int(hit5)
                hits_at_10 += int(hit10)
                ap_sum += ap
                evaluated += 1

                out = {
                    "stem": stem,
                    "caption": valid_caps[row_idx],
                    "rank": rank,
                    "ap": ap,
                    "hit@1": hit1,
                    "hit@5": hit5,
                    "hit@10": hit10,
                    "gt_sim": gt_sim,
                    "top_predictions": top_predictions,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if evaluated == 0:
        raise RuntimeError("No samples were evaluated successfully.")

    metrics = {
        "num_samples": evaluated,
        "num_skipped": skipped,
        "recall@1": hits_at_1 / evaluated,
        "recall@5": hits_at_5 / evaluated,
        "recall@10": hits_at_10 / evaluated,
        "mAP": ap_sum / evaluated,
        "map_definition": "single relevant item per query; AP = 1 / full-database rank of the ground-truth stem",
        "text_root": str(text_root),
        "sem_db": str(sem_db_path),
        "clip_model": clip_model_name,
        "stems_file": str(stems_file) if stems_file else None,
        "caption_mode": args.caption_mode,
        "topk_saved_per_sample": int(args.topk),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Recall@1  = {metrics['recall@1']:.4f}")
    log(f"Recall@5  = {metrics['recall@5']:.4f}")
    log(f"Recall@10 = {metrics['recall@10']:.4f}")
    log(f"mAP       = {metrics['mAP']:.4f}")
    log(f"saved metrics -> {metrics_path}")
    log(f"saved per-sample results -> {result_path}")


if __name__ == "__main__":
    main()

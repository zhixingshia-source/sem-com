#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation for payload-based image retrieval.

This script reuses the single-sample retrieval helpers from
`05_run_image_retrieval.py` and evaluates all payloads, or a provided stem list.

Default test split:
  payload_root/*_payload.json intersected with stems contained in sem_db.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def log(*items):
    print("[batch-retrieval-eval]", *items, flush=True)


def load_infer_helpers():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "05_run_image_retrieval.py",
        here.parent / "kreps2sem_infer_(11-22).py",
    ]
    helper_path = next((p for p in candidates if p.exists()), None)
    if helper_path is None:
        raise FileNotFoundError("Could not find 05_run_image_retrieval.py or kreps2sem_infer_(11-22).py")

    spec = importlib.util.spec_from_file_location("payload_retrieval_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, helper_path


def load_stems(payload_root: Path, stems_file: Optional[Path], db_stems: List[str]) -> List[str]:
    db_set = set(db_stems)

    if stems_file is not None:
        stems = []
        for line in stems_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            stems.append(s)
    else:
        stems = [
            p.name[: -len("_payload.json")]
            for p in sorted(payload_root.glob("*_payload.json"))
            if p.name.endswith("_payload.json")
        ]

    stems = [s for s in stems if (payload_root / f"{s}_payload.json").exists()]
    stems = [s for s in stems if s in db_set]
    return stems


def average_precision_single_relevant(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def main():
    parser = argparse.ArgumentParser("batch payload image retrieval evaluation")
    parser.add_argument("--payload_root", type=str, required=True, help="Directory containing *_payload.json")
    parser.add_argument("--adapter_ckpt", type=str, required=True, help="adapter_clip_best.pth from adapter training")
    parser.add_argument("--sem_db", type=str, required=True, help="image semantic DB .pt")
    parser.add_argument("--outdir", type=str, required=True, help="Directory for metrics.json and per_sample_results.jsonl")
    parser.add_argument("--stems_file", type=str, default=None, help="Optional text file, one stem per line")
    parser.add_argument("--topk", type=int, default=10, help="Store top-k predictions per sample")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples, 0 means all")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    helpers, helper_path = load_infer_helpers()
    log(f"reusing helpers from {helper_path}")

    payload_root = Path(args.payload_root).resolve()
    adapter_ckpt = Path(args.adapter_ckpt).resolve()
    sem_db_path = Path(args.sem_db).resolve()
    outdir = Path(args.outdir).resolve()
    stems_file = Path(args.stems_file).resolve() if args.stems_file else None
    outdir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    log(f"payload_root = {payload_root}")
    log(f"adapter_ckpt = {adapter_ckpt}")
    log(f"sem_db = {sem_db_path}")
    log(f"outdir = {outdir}")
    log(f"device = {device}")

    adapter, d_in, d_out, x_mean, x_std = helpers.build_adapter_from_ckpt(adapter_ckpt, device)

    db = torch.load(str(sem_db_path), map_location="cpu")
    db_stems = list(db["stems"])
    embeds_db = db["embeds"].float()
    embeds_db = F.normalize(embeds_db, dim=-1)
    if embeds_db.shape[1] != d_out:
        raise RuntimeError(f"dimension mismatch: adapter d_out={d_out} vs sem_db D={embeds_db.shape[1]}")

    stem_to_db_index: Dict[str, int] = {s: i for i, s in enumerate(db_stems)}
    stems = load_stems(payload_root, stems_file, db_stems)
    if args.limit and args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError("No evaluable stems found. Check payload_root, stems_file, and sem_db.")

    log(f"num_eval_samples = {len(stems)}")
    log(f"topk = {args.topk}")

    result_path = outdir / "per_sample_results.jsonl"
    metrics_path = outdir / "metrics.json"

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    ap_sum = 0.0
    evaluated = 0
    skipped = 0

    with result_path.open("w", encoding="utf-8") as f:
        for stem in tqdm(stems, desc="retrieval-eval"):
            payload_path = payload_root / f"{stem}_payload.json"
            gt_idx = stem_to_db_index.get(stem)
            if gt_idx is None or not payload_path.exists():
                skipped += 1
                continue

            try:
                reps = helpers.load_kreps_json(payload_path)
                x_flat, k_raw, k_eff = helpers.make_input_from_reps(reps, d_in=d_in)
                x_norm = (x_flat - x_mean) / (x_std + 1e-6)
                x = torch.from_numpy(x_norm).unsqueeze(0).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    z_pred = adapter(x)
                    z_pred = F.normalize(z_pred, dim=-1)

                sims = (z_pred.cpu() @ embeds_db.T).squeeze(0)
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

                row = {
                    "stem": stem,
                    "rank": rank,
                    "ap": ap,
                    "hit@1": hit1,
                    "hit@5": hit5,
                    "hit@10": hit10,
                    "gt_sim": gt_sim,
                    "payload_path": str(payload_path),
                    "k_raw": int(k_raw),
                    "k_eff": int(k_eff),
                    "top_predictions": top_predictions,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                skipped += 1
                row = {
                    "stem": stem,
                    "error": str(exc),
                    "payload_path": str(payload_path),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
        "payload_root": str(payload_root),
        "adapter_ckpt": str(adapter_ckpt),
        "sem_db": str(sem_db_path),
        "stems_file": str(stems_file) if stems_file else None,
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

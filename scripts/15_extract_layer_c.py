"""Layer C：CLAP 预训练嵌入提取。

    uv run python scripts/15_extract_layer_c.py --dataset reg_emo_mix --batch 16
"""

from __future__ import annotations

import argparse
import time

import librosa
import numpy as np
import pandas as pd

from soundml import features_c
from soundml.features_a import TARGET_LUFS
from soundml.provenance import REPO_ROOT, RunRecord

import pyloudnorm

_meter = None


def load_norm(path: str) -> np.ndarray:
    """48 kHz 单声道、居中 30 s、归一到 −23 LUFS —— 与 Layer A/B 口径一致。"""
    global _meter
    y, _ = librosa.load(path, sr=features_c.FS, mono=True)
    n = int(features_c.MAX_DURATION_S * features_c.FS)
    if y.size > n:
        s = (y.size - n) // 2
        y = y[s:s + n]
    if _meter is None:
        _meter = pyloudnorm.Meter(features_c.FS)
    if y.size >= int(0.4 * features_c.FS):
        lu = _meter.integrated_loudness(y)
        if np.isfinite(lu):
            y = y * 10.0 ** ((TARGET_LUFS - lu) / 20.0)
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="如 reg_emo_mix / reg_deam")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    labels = pd.read_csv(REPO_ROOT / "data" / f"{args.dataset}.csv")
    out = REPO_ROOT / "data" / "features" / f"layer_c_{args.dataset}.parquet"

    with RunRecord(f"extract_layer_c_{args.dataset}", seed=0,
                   params={"dataset": args.dataset, "model": features_c.MODEL_ID,
                           "extractor_version": features_c.EXTRACTOR_VERSION,
                           "fs": features_c.FS, "target_lufs": TARGET_LUFS,
                           "note": "笔记 §2.3 明确此层为对照组，非主力"}) as run:
        embs, ids, failed = [], [], []
        t0 = time.perf_counter()
        rows = list(labels.itertuples(index=False))
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            waves, keep = [], []
            for r in chunk:
                try:
                    waves.append(load_norm(r.path)); keep.append(r.clip_id)
                except Exception as e:
                    failed.append({"clip_id": r.clip_id, "error": str(e)[:120]})
            if waves:
                embs.append(features_c.embed_batch(waves)); ids.extend(keep)
            done = min(i + args.batch, len(rows))
            if done % 160 == 0 or done == len(rows):
                rate = done / (time.perf_counter() - t0)
                print(f"  {done}/{len(rows)}  ({rate:.1f} clip/s)", flush=True)

        E = np.vstack(embs)
        feats = pd.DataFrame(E, columns=features_c.column_names(E.shape[1]))
        feats.insert(0, "clip_id", ids)
        merged = labels.merge(feats, on="clip_id", validate="one_to_one")
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out, index=False)

        run.log_metric("n_clips", len(merged))
        run.log_metric("dim", int(E.shape[1]))
        run.log_metric("n_failed", len(failed))
        run.log_metric("elapsed_s", round(time.perf_counter() - t0, 1))
        run.note("output", out.relative_to(REPO_ROOT).as_posix())
        print(f"\nrun_id: {run.run_id}\n维度 {E.shape[1]}  样本 {len(merged)}  失败 {len(failed)}")


if __name__ == "__main__":
    main()

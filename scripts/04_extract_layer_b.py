"""Step 3: 提取 Layer B 心理声学特征（多进程）。

    uv run python scripts/04_extract_layer_b.py --dataset deam
    uv run python scripts/04_extract_layer_b.py --dataset deam --limit 3   # 试跑
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from joblib import Parallel, delayed

from soundml import features_b
from soundml.provenance import REPO_ROOT, RunRecord


def _one(clip_id: str, path: str) -> dict:
    try:
        return {"clip_id": clip_id, **features_b.extract(path)}
    except Exception as exc:
        return {"clip_id": clip_id, "__error__": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["esc50", "deam", "pmemo", "emo"], required=True)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 段，用于试跑")
    args = ap.parse_args()

    labels_path = REPO_ROOT / "data" / f"labels_{args.dataset}.csv"
    labels = pd.read_csv(labels_path)
    if args.limit:
        labels = labels.head(args.limit)

    out = REPO_ROOT / "data" / "features" / f"layer_b_{args.dataset}.parquet"

    with RunRecord(
        f"extract_layer_b_{args.dataset}",
        seed=0,
        params={
            "dataset": args.dataset,
            "extractor_version": features_b.EXTRACTOR_VERSION,
            "fs": features_b.FS,
            "max_duration_s": features_b.MAX_DURATION_S,
            "roughness_duration_s": features_b.ROUGHNESS_DURATION_S,
            "seg_s": features_b.SEG_S,
            "target_lufs": features_b.TARGET_LUFS,
            "reference_spl_db": features_b.REFERENCE_SPL_DB,
            "n_jobs": args.n_jobs,
            "limit": args.limit,
        },
    ) as run:
        run.add_input_snapshot("labels", [labels_path])

        t0 = time.perf_counter()
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_one)(r.clip_id, r.path) for r in labels.itertuples(index=False)
        )
        elapsed = time.perf_counter() - t0

        failed = [r for r in results if "__error__" in r]
        ok = [r for r in results if "__error__" not in r]
        feats = pd.DataFrame(ok)

        if args.limit:  # 试跑不落盘，只打印
            print(feats.T)
            run.log_metric("trial_only", True)
        else:
            merged = labels.merge(feats, on="clip_id", validate="one_to_one")
            out.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(out, index=False)
            run.note("output", out.relative_to(REPO_ROOT).as_posix())

        run.log_metric("n_ok", len(ok))
        run.log_metric("n_features", feats.shape[1] - 1 if len(feats) else 0)
        run.log_metric("n_failed", len(failed))
        run.log_metric("failed", failed[:10])
        run.log_metric("elapsed_s", round(elapsed, 1))

        print(f"\nrun_id   : {run.run_id}")
        print(f"特征维度 : {feats.shape[1] - 1 if len(feats) else 0}   成功 {len(ok)}   失败 {len(failed)}")
        print(f"耗时     : {elapsed:.0f}s ({len(labels) / max(elapsed, 1e-9):.2f} clip/s)")
        for f in failed[:5]:
            print("  FAIL", f["clip_id"], f["__error__"])


if __name__ == "__main__":
    main()

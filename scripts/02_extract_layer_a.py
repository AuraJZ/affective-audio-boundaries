"""Step 1b / 2b: 提取 Layer A 特征，写 parquet。

    uv run python scripts/02_extract_layer_a.py --dataset deam
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from soundml import features_a
from soundml.provenance import REPO_ROOT, RunRecord


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="分类用数据集名（esc50/deam/pmemo/emo），或回归标签文件主干如 reg_deam")
    ap.add_argument("--normalize-loudness", action="store_true",
                    help="提特征前把每段音频归一到 -23 LUFS（响度混淆判别实验）")
    ap.add_argument("--duration", type=float, default=None,
                    help="覆盖居中截窗时长（秒）。用于时长混淆对照："
                         "把音乐语料截到与环境声语料相同的 6 s，排除「域之墙其实是时长之墙」")
    args = ap.parse_args()

    if args.duration is not None:
        features_a.MAX_DURATION_S = args.duration

    # 以 reg_ 开头的视为回归标签文件，否则按 labels_<name>.csv 查找
    if args.dataset.startswith("reg_"):
        labels_path = REPO_ROOT / "data" / f"{args.dataset}.csv"
        stem = args.dataset
    else:
        labels_path = REPO_ROOT / "data" / f"labels_{args.dataset}.csv"
        stem = args.dataset
    tag = f"{stem}_ln" if args.normalize_loudness else stem
    if args.duration is not None:
        tag += f"_{int(args.duration)}s"
    out = REPO_ROOT / "data" / "features" / f"layer_a_{tag}.parquet"
    labels = pd.read_csv(labels_path)

    with RunRecord(
        f"extract_layer_a_{tag}",
        seed=0,
        params={
            "dataset": args.dataset,
            "normalize_loudness": args.normalize_loudness,
            "target_lufs": features_a.TARGET_LUFS if args.normalize_loudness else None,
            "extractor_version": features_a.EXTRACTOR_VERSION,
            "sr": features_a.SR,
            "n_fft": features_a.N_FFT,
            "hop": features_a.HOP,
            "max_duration_s": features_a.MAX_DURATION_S,
            "slope_band_hz": features_a.SLOPE_BAND,
            "low_band_hz": features_a.LOW_BAND,
            "harsh_band_hz": features_a.HARSH_BAND,
        },
    ) as run:
        run.add_input_snapshot("labels", [labels_path])

        rows, failed = [], []
        t0 = time.perf_counter()
        for i, rec in enumerate(labels.itertuples(index=False), start=1):
            try:
                rows.append({
                    "clip_id": rec.clip_id,
                    **features_a.extract(rec.path, normalize_loudness=args.normalize_loudness),
                })
            except Exception as exc:  # 单个坏文件不该中断整批
                failed.append({"clip_id": rec.clip_id, "error": f"{type(exc).__name__}: {exc}"})
            if i % 100 == 0 or i == len(labels):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(labels)}  ({rate:.1f} clip/s, 失败 {len(failed)})", flush=True)

        feats = pd.DataFrame(rows)
        merged = labels.merge(feats, on="clip_id", validate="one_to_one")
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out, index=False)

        run.log_metric("n_clips", len(merged))
        run.log_metric("n_features", feats.shape[1] - 1)
        run.log_metric("n_failed", len(failed))
        run.log_metric("failed", failed[:20])
        run.log_metric("elapsed_s", round(time.perf_counter() - t0, 1))
        run.log_metric("n_nan_cells", int(feats.isna().sum().sum()))
        run.note("output", out.relative_to(REPO_ROOT).as_posix())

        print(f"\nrun_id   : {run.run_id}")
        print(f"特征维度 : {feats.shape[1] - 1}   样本 {len(merged)}   失败 {len(failed)}")
        print(f"输出     : {out}")


if __name__ == "__main__":
    main()

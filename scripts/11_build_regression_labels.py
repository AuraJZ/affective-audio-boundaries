"""回归标签构造 —— 用上全部样本，不丢中间地带。

    uv run python scripts/11_build_regression_labels.py --dataset deam
    uv run python scripts/11_build_regression_labels.py --dataset emo --include-mixes
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soundml import labels as L
from soundml.provenance import REPO_ROOT, RunRecord

RAW = REPO_ROOT / "data" / "raw"
ROOTS = {
    "deam": RAW / "DEAM",
    "pmemo": RAW / "PMEmo" / "PMEmo2019",
    "emo": RAW / "EmoSoundscapes" / "Emo-Soundscapes",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(L.REG_BUILDERS), required=True)
    ap.add_argument("--include-mixes", action="store_true",
                    help="仅 emo：纳入 613 段人工混音")
    ap.add_argument("--no-std-filter", action="store_true",
                    help="不做标注分歧过滤，用满全部标注")
    args = ap.parse_args()

    kw = {}
    if args.dataset == "emo":
        kw["include_mixes"] = args.include_mixes
    elif args.no_std_filter:
        kw["std_cap"] = None

    tag = args.dataset + ("_mix" if args.include_mixes else "") + ("_nofilter" if args.no_std_filter else "")
    out = REPO_ROOT / "data" / f"reg_{tag}.csv"

    with RunRecord(f"build_reg_labels_{tag}", seed=0,
                   params={"dataset": args.dataset, "task": "regression",
                           "target": "arousal (−1..1)", **kw}) as run:
        df = L.REG_BUILDERS[args.dataset](ROOTS[args.dataset], **kw)
        run.add_input_snapshot("audio", [Path(p) for p in df["path"]])
        df.to_csv(out, index=False, encoding="utf-8")

        run.log_metric("n", len(df))
        run.log_metric("n_groups", int(df["group"].nunique()))
        run.log_metric("arousal_stats", {
            "min": float(df.arousal.min()), "max": float(df.arousal.max()),
            "mean": float(df.arousal.mean()), "std": float(df.arousal.std())})
        run.note("output", out.relative_to(REPO_ROOT).as_posix())

        print(f"run_id : {run.run_id}")
        print(f"样本   : {len(df)}   分组 {df['group'].nunique()}")
        print(f"arousal: [{df.arousal.min():.3f}, {df.arousal.max():.3f}]  "
              f"均值 {df.arousal.mean():+.3f}  标准差 {df.arousal.std():.3f}")
        print(f"输出   : {out}")


if __name__ == "__main__":
    main()

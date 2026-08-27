"""Step 1a / 2a: 构造标签表，并留一条 run record。

    uv run python scripts/01_build_labels.py --dataset deam
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soundml import labels as L
from soundml.provenance import REPO_ROOT, RunRecord

RAW = REPO_ROOT / "data" / "raw"

ROOTS = {
    "esc50": RAW / "ESC-50",
    "deam": RAW / "DEAM",
    "pmemo": RAW / "PMEmo" / "PMEmo2019",
    "emo": RAW / "EmoSoundscapes" / "Emo-Soundscapes",
}

CRITERIA = {
    "esc50": {
        "label_type": "semantic_proxy",
        "positive_categories": L.POSITIVE_CATEGORIES,
        "negative_categories": L.NEGATIVE_CATEGORIES,
        "role": "smoke_test",
    },
    "deam": {
        "label_type": "arousal_annotation",
        "scale": "1-9",
        "arousal_low": L.DEAM_AROUSAL_LOW,
        "arousal_high": L.DEAM_AROUSAL_HIGH,
        "valence_floor": L.DEAM_VALENCE_FLOOR,
        "std_cap": L.DEAM_STD_CAP,
        "role": "train",
    },
    "pmemo": {
        "label_type": "arousal_annotation",
        "scale": "0-1",
        "arousal_low": L.PMEMO_AROUSAL_LOW,
        "arousal_high": L.PMEMO_AROUSAL_HIGH,
        "valence_floor": L.PMEMO_VALENCE_FLOOR,
        "std_cap": L.PMEMO_STD_CAP,
        "role": "independent_test",
    },
    "emo": {
        "label_type": "arousal_annotation",
        "scale": "-1..1",
        "domain": "environmental_soundscape",
        "arousal_low": L.EMO_AROUSAL_LOW,
        "arousal_high": L.EMO_AROUSAL_HIGH,
        "valence_floor": L.EMO_VALENCE_FLOOR,
        "std_cap": None,  # 数据集不提供逐条标注分歧
        "subset": "600 原始录音（丢弃 613 段人工混音以避免泄漏）",
        "role": "train_environmental_domain",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(L.BUILDERS), required=True)
    args = ap.parse_args()

    builder, pretty = L.BUILDERS[args.dataset]
    out = REPO_ROOT / "data" / f"labels_{args.dataset}.csv"

    with RunRecord(
        f"build_labels_{args.dataset}",
        seed=0,
        params={"dataset": pretty, "criteria": CRITERIA[args.dataset]},
    ) as run:
        df = builder(ROOTS[args.dataset])
        run.add_input_snapshot("audio", [Path(p) for p in df["path"]])
        df.to_csv(out, index=False, encoding="utf-8")

        run.log_metric("n_total", len(df))
        run.log_metric("n_positive", int(df["label"].sum()))
        run.log_metric("n_negative", int((1 - df["label"]).sum()))
        run.log_metric("n_groups", int(df["group"].nunique()))
        run.note("output", out.relative_to(REPO_ROOT).as_posix())
        if args.dataset == "esc50":
            run.note("warning", "语义类别代理标签，非实测助眠效应；AUC 无科学意义")

        print(f"run_id : {run.run_id}")
        print(f"样本   : {len(df)}  (正 {df['label'].sum()} / 负 {(1 - df['label']).sum()})")
        print(f"分组   : {df['group'].nunique()}")
        print(f"输出   : {out}")


if __name__ == "__main__":
    main()

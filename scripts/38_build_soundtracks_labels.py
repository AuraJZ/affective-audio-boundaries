"""构造 Soundtracks (Eerola & Vuoskoski 2011) 回归标签 —— 第三个独立评分语料。

    .venv/Scripts/python scripts/38_build_soundtracks_labels.py

## 为什么加这个语料

此前的评分侧结论建立在 DEAM / PMEmo / Emo-Soundscapes 上。Soundtracks Set1
是**第四个、且来源完全独立**的语料：360 段电影配乐节选，评分来自与本项目
其他语料毫无交集的评分研究，且**同时给出维度（valence/energy/tension）与
离散（anger/fear/happy/sad/tender）两套标注** —— 后者是前三个语料都没有的。

它还有一个别处没有的性质：**同一批音频被 ds002721 的 31 名受试听过并采集了
脑电**。因此无论生理侧结论如何，这批音频都是「评分 ↔ 声学」与
「评分 ↔ 生理」两条线唯一的公共锚点。

## 坐标对齐

原始评分为 1–9 李克特量表。为与既有 `reg_*` 语料（[-1, 1]）可比，
线性映射 `(x - 5) / 4`。`energy` 对应既有语料的 `arousal`，
`valence` 同名，`tension` 无对应项，单独保留。

映射是线性的，不改变任何秩相关；本项目的评分侧统计量全部基于 Spearman，
因此该变换对结论无影响，只为量纲可比。
"""

from __future__ import annotations

import pandas as pd

from soundml.provenance import REPO_ROOT, RunRecord

AUD = REPO_ROOT / "data" / "raw" / "soundtracks"
MP3 = AUD / "Set1"
RATINGS = AUD / "mean_ratings_set1.csv"
OUT = REPO_ROOT / "data" / "reg_soundtracks.csv"

DIMS = ["valence", "energy", "tension"]
DISCRETE = ["anger", "fear", "happy", "sad", "tender"]


def main() -> None:
    with RunRecord(
        "build_soundtracks_labels", seed=0,
        params={"source": "Eerola & Vuoskoski (2011) Soundtracks Set1",
                "osf": "https://osf.io/p6vkg/",
                "licence": "academic research use (see OSF readme)",
                "scale": "1-9 Likert → (x-5)/4 → [-1, 1]",
                "n_expected": 360},
    ) as run:
        run.add_input_snapshot("ratings", [RATINGS])

        R = pd.read_csv(RATINGS, encoding="utf-8-sig")
        R.columns = [c.strip().lower() for c in R.columns]
        R["number"] = R["number"].astype(str).str.zfill(3)

        rows, missing = [], []
        for rec in R.itertuples(index=False):
            p = MP3 / f"{rec.number}.mp3"
            if not p.exists():
                missing.append(rec.number)
                continue
            d = {"clip_id": f"st_{rec.number}", "path": str(p),
                 "stim": int(rec.number),
                 "group": f"st_{rec.number}", "source": "Soundtracks",
                 "category": "music", "target_emotion": rec.target}
            for c in DIMS + DISCRETE:
                d[c] = (float(getattr(rec, c)) - 5.0) / 4.0
            # 与既有语料的列名对齐：energy 即唤醒轴
            d["arousal"] = d["energy"]
            rows.append(d)

        D = pd.DataFrame(rows)
        D.to_csv(OUT, index=False)

        run.log_metric("n_clips", int(len(D)))
        run.log_metric("n_missing_audio", len(missing))
        run.log_metric("target_emotion_counts",
                       {str(k): int(v) for k, v in
                        D.target_emotion.value_counts().items()})
        for c in DIMS:
            run.log_metric(f"{c}_range",
                           [float(D[c].min()), float(D[c].max()), float(D[c].std())])
        run.note("output", str(OUT))

        print(f"→ {OUT}   {len(D)} 段（缺音频 {len(missing)}）")
        print("\n目标情绪分布：")
        for k, v in D.target_emotion.value_counts().items():
            print(f"  {k:12s} {v}")
        print("\n维度评分（映射到 [-1,1] 后）：")
        print(D[DIMS + DISCRETE].describe().loc[["mean", "std", "min", "max"]]
              .round(3).to_string())


if __name__ == "__main__":
    main()

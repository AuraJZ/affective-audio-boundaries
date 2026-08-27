"""主观标注的信度天花板 —— 把我们自己的方法学发现用在自己的头号数字上。

    uv run python scripts/20_label_ceiling.py

## 为什么必须做

报告 B2 的结论是：**说「X 预测不了 Y」之前，必须先测 Y 的信度**。
我们对 EDA 做了（天花板 0.155），却**从未对主观标注做过**。

于是 ρ = 0.736（DEAM 手工特征）/ 0.773（CLAP）到底是好是坏，我们并不知道：

- 若评分者间信度约 0.78 → 已贴天花板，再优化模型是在榨噪声
- 若信度约 0.95 → 还有实质空间，值得继续投入表示与模型

这个数决定后续所有 in-silico 工作的方向，所以排在最前。

## 方法

DEAM 提供逐评分者的 song-level 打分（workerID, SongId, Valence, Arousal）。

1. **分半信度**：把评分者随机分两半 → 各自求每首歌的均值 → 两半相关
   → Spearman-Brown 校正到全体评分者。与 EDA 分析用同一方法，保证可比。
2. **衰减校正**：ρ_true = ρ_observed / sqrt(reliability)，
   给出模型与「无噪声真值」的相关。
3. 在**分析子集**（std ≤ 1.8 过滤后的 1233 首）上单独再算一次——
   模型是在这个子集上评估的，天花板也必须在同一子集上取。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

RATER_DIR = (REPO_ROOT / "data" / "raw" / "DEAM" / "annotations"
             / "annotations per each rater" / "song_level")
REG_LABELS = REPO_ROOT / "data" / "reg_deam.csv"

N_SPLITS = 200
# 我们已报告的模型成绩（组内 GroupKFold-5，Spearman ρ）
OBSERVED = {"Hand-crafted + XGBoost": 0.736, "CLAP + ridge probe": 0.773}


def load_raters() -> pd.DataFrame:
    """两个文件 schema 不同，需归一化。

    songs 1–2000 : workerID, SongId, Valence, Arousal
    songs 2000+  : SongId, WorkerId, Arousal_Average, Valence_Average, …
    """
    rename = {"workerid": "workerID", "songid": "SongId",
              "arousal": "Arousal", "valence": "Valence",
              "arousal_average": "Arousal", "valence_average": "Valence"}
    frames = []
    for f in sorted(RATER_DIR.glob("*.csv")):
        d = pd.read_csv(f, skipinitialspace=True)
        d.columns = [rename.get(c.strip().lower(), c.strip()) for c in d.columns]
        keep = ["workerID", "SongId", "Arousal", "Valence"]
        missing = [c for c in keep if c not in d.columns]
        if missing:
            raise ValueError(f"{f.name} 缺列 {missing}；实际列 {list(d.columns)}")
        frames.append(d[keep])
    return pd.concat(frames, ignore_index=True)


def split_half(df: pd.DataFrame, col: str, rng, n_splits: int) -> dict:
    """评分者随机分半 → 各自的歌曲均值 → 相关 → Spearman-Brown 校正。"""
    workers = df["workerID"].unique()
    raw = []
    for _ in range(n_splits):
        perm = rng.permutation(workers)
        a = set(perm[: len(perm) // 2])
        ga = df[df.workerID.isin(a)].groupby("SongId")[col].mean()
        gb = df[~df.workerID.isin(a)].groupby("SongId")[col].mean()
        common = ga.index.intersection(gb.index)
        if len(common) > 50:
            r = spearmanr(ga.loc[common], gb.loc[common]).statistic
            if np.isfinite(r):
                raw.append(float(r))
    raw = np.array(raw)
    sb = 2 * raw / (1 + raw)          # Spearman-Brown 校正到全体
    return {"raw_mean": float(raw.mean()),
            "spearman_brown": float(sb.mean()),
            "sb_p10": float(np.percentile(sb, 10)),
            "sb_p90": float(np.percentile(sb, 90)),
            "n_splits": int(raw.size)}


def main() -> None:
    df = load_raters()
    rng = np.random.default_rng(SEED)

    per_song = df.groupby("SongId").size()
    per_worker = df.groupby("workerID").size()

    # 分析子集：模型实际评估用的 1233 首
    subset_ids = set(
        pd.read_csv(REG_LABELS)["clip_id"].str.removeprefix("deam_").astype(int))
    df_sub = df[df["SongId"].isin(subset_ids)]

    with RunRecord(
        "label_reliability_ceiling", seed=SEED,
        params={"corpus": "DEAM", "n_splits": N_SPLITS,
                "method": "rater split-half + Spearman-Brown",
                "note": "与 EDA 信度分析同法，保证两者可比"},
    ) as run:
        run.log_metric("n_ratings", int(len(df)))
        run.log_metric("n_workers", int(df["workerID"].nunique()))
        run.log_metric("n_songs", int(df["SongId"].nunique()))
        run.log_metric("ratings_per_song",
                       {"median": float(per_song.median()),
                        "p10": float(per_song.quantile(0.1)),
                        "p90": float(per_song.quantile(0.9))})
        run.log_metric("ratings_per_worker",
                       {"median": float(per_worker.median()),
                        "max": int(per_worker.max())})

        ceil_all = split_half(df, "Arousal", rng, N_SPLITS)
        ceil_sub = split_half(df_sub, "Arousal", rng, N_SPLITS)
        ceil_val = split_half(df, "Valence", rng, N_SPLITS)
        run.log_metric("ceiling_arousal_all_songs", ceil_all)
        run.log_metric("ceiling_arousal_analysis_subset", ceil_sub)
        run.log_metric("ceiling_valence_all_songs", ceil_val)

        # 衰减校正：模型与「无噪声真值」的相关
        R = ceil_sub["spearman_brown"]
        corrected = {k: {"observed": v,
                         "pct_of_ceiling": v / R,
                         "attenuation_corrected": v / np.sqrt(R)}
                     for k, v in OBSERVED.items()}
        run.log_metric("model_vs_ceiling", corrected)

        headroom = R - max(OBSERVED.values())
        run.log_metric("headroom_to_ceiling", float(headroom))
        run.log_metric("at_ceiling", bool(headroom < 0.05))

        # ---------- 打印 ----------
        print("=" * 72)
        print(f"run_id: {run.run_id}")
        print(f"{len(df):,} 条打分   {df['workerID'].nunique():,} 名评分者   "
              f"{df['SongId'].nunique():,} 首歌")
        print(f"每首歌中位打分数 {per_song.median():.0f} "
              f"(p10 {per_song.quantile(0.1):.0f} – p90 {per_song.quantile(0.9):.0f})")

        print("\n--- 标注信度天花板（评分者分半 + Spearman-Brown）---")
        for name, c in (("全部 1802 首", ceil_all),
                        ("分析子集 1233 首", ceil_sub),
                        ("Valence 对照", ceil_val)):
            print(f"  {name:<20} 分半 ρ = {c['raw_mean']:.3f}   "
                  f"校正后 = {c['spearman_brown']:.3f}  "
                  f"[{c['sb_p10']:.3f}, {c['sb_p90']:.3f}]")

        print(f"\n--- 模型成绩 vs 天花板（{R:.3f}）---")
        for k, v in corrected.items():
            print(f"  {k:<26} ρ = {v['observed']:.3f}   "
                  f"占天花板 {v['pct_of_ceiling']:.0%}   "
                  f"衰减校正后 {v['attenuation_corrected']:.3f}")

        print(f"\n剩余空间 = {headroom:+.3f}")
        if headroom < 0.05:
            print("→ **已贴天花板**：继续优化模型是在榨标注噪声。")
            print("   后续应转向「扩大覆盖」而非「提升精度」。")
        else:
            print("→ 仍有实质空间，值得继续投入表示与模型。")


if __name__ == "__main__":
    main()

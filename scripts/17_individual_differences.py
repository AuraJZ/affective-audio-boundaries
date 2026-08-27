"""PLAN III-3：个体差异 —— 规则对每个人都一样吗？

    uv run python scripts/17_individual_differences.py

## 为什么重要

笔记 §8.2 提出了这个风险却从未实现：

> 嗅觉受体差异已大，听觉/音乐偏好差异**更大**。

而笔记 §9.2 的 Layer 3「用户分型 → 声音子库」整层，
**前提就是个体间存在可量化的差异**。若不同人的声学响应几乎一致，
那一层是多余的；若差异很大，那一层是必需的，且需要知道差异有多大。

PMEmo 的 EDA 是**逐被试**记录的，可以直接量化：
对每名被试单独算「声学特征 ↔ 其生理唤醒」的关系，看被试间散布多大。

## 指标

1. **被试内一致性**：每名被试的 EDA 与群体平均 EDA 的相关（split-half 思路）
2. **规则的个体间方差**：域无关特征对每名被试的效应（Spearman ρ）的分布
3. **被试可解释方差占比**：混合效应视角下，多少方差来自「谁在听」
   而非「听什么」
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer

from soundml.eda import BASELINE_FRAC, FS, HIGHPASS_HZ, _clip_features
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED, feature_cols

EDA_DIR = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019" / "EDA"
FEAT = REPO_ROOT / "data" / "features" / "layer_a_reg_pmemo_ln.parquet"
INVARIANT = ["onset_env_dmean", "spec_contrast3_dmean", "onset_rate",
             "spec_flatness_std", "onset_env_mean", "spec_flatness_p50", "mfcc1_std"]
MIN_CLIPS_PER_SUBJECT = 15


def per_subject_table() -> pd.DataFrame:
    """逐 (被试, 音频) 的生理唤醒值。"""
    rows = []
    for f in sorted(EDA_DIR.glob("*_EDA.csv")):
        mid = f.stem.split("_")[0]
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        t = df.iloc[:, 0].to_numpy(dtype=np.float64)
        for subj in df.columns[1:]:
            tonic, phasic = _clip_features(t, df[subj].to_numpy())
            if np.isfinite(tonic) and np.isfinite(phasic):
                rows.append((mid, str(subj), tonic, phasic))
    R = pd.DataFrame(rows, columns=["music_id", "subject", "tonic", "phasic"])
    for c in ("tonic", "phasic"):
        g = R.groupby("subject")[c]
        R[f"z_{c}"] = (R[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    R["eda"] = R[["z_tonic", "z_phasic"]].mean(axis=1)
    return R.dropna(subset=["eda"])


def main() -> None:
    print("读取逐被试 EDA…", flush=True)
    R = per_subject_table()
    feats = pd.read_parquet(FEAT)
    feats["music_id"] = feats["clip_id"].str.removeprefix("pmemo_")
    cols = [c for c in feature_cols(feats) if c != "music_id"]

    F = feats[["music_id", *cols]].copy()
    X = SimpleImputer(strategy="median").fit_transform(F[cols].to_numpy(dtype=np.float64))
    F.loc[:, cols] = X
    M = R.merge(F, on="music_id", how="inner")

    counts = M.groupby("subject").size()
    keep = counts[counts >= MIN_CLIPS_PER_SUBJECT].index
    M = M[M["subject"].isin(keep)]

    with RunRecord("individual_differences", seed=SEED,
                   params={"min_clips_per_subject": MIN_CLIPS_PER_SUBJECT,
                           "n_subjects_kept": int(len(keep)),
                           "invariant_features": INVARIANT}) as run:
        run.log_metric("n_subject_clip_pairs", int(len(M)))
        run.log_metric("n_subjects", int(M["subject"].nunique()))
        run.log_metric("n_clips", int(M["music_id"].nunique()))
        run.log_metric("clips_per_subject",
                       {"median": float(counts[keep].median()),
                        "min": int(counts[keep].min()), "max": int(counts[keep].max())})

        # ---------- 0. 分半信度：EDA 聚合量的可预测性上限 ⭐ ----------
        # 这是解读「声学无法预测 EDA」的前提对照。
        # 若聚合量自身信度接近 0，则任何预测器都到不了，ρ≈0 无信息量；
        # 若信度高，则声学预测失败是实质发现。
        rng = np.random.default_rng(SEED)
        halves = []
        for _ in range(50):
            subs = M["subject"].unique()
            perm = rng.permutation(subs)
            a, b = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
            ga = M[M.subject.isin(a)].groupby("music_id")["eda"].mean()
            gb = M[M.subject.isin(b)].groupby("music_id")["eda"].mean()
            common = ga.index.intersection(gb.index)
            if len(common) > 30:
                r = spearmanr(ga.loc[common], gb.loc[common]).statistic
                if np.isfinite(r):
                    halves.append(float(r))
        halves = np.array(halves)
        # Spearman-Brown 校正到全样本
        sb = 2 * halves / (1 + halves)
        run.log_metric("split_half_reliability",
                       {"raw_mean": float(halves.mean()),
                        "spearman_brown": float(np.mean(sb)),
                        "p10": float(np.percentile(halves, 10)),
                        "p90": float(np.percentile(halves, 90))})

        # ---------- 1. 被试与群体均值的一致性 ----------
        pop = M.groupby("music_id")["eda"].mean()
        agree = []
        for s, g in M.groupby("subject"):
            common = g["music_id"]
            if len(common) >= MIN_CLIPS_PER_SUBJECT:
                r = spearmanr(g["eda"], pop.loc[common]).statistic
                if np.isfinite(r):
                    agree.append(float(r))
        agree = np.array(agree)
        run.log_metric("subject_vs_population_rho",
                       {"mean": float(agree.mean()), "median": float(np.median(agree)),
                        "p10": float(np.percentile(agree, 10)),
                        "p90": float(np.percentile(agree, 90)),
                        "frac_negative": float((agree < 0).mean())})

        # ---------- 2. 域无关特征的个体间效应散布 ----------
        rows = []
        for f in INVARIANT:
            if f not in M:
                continue
            rhos = []
            for s, g in M.groupby("subject"):
                if g[f].nunique() > 3:
                    r = spearmanr(g[f], g["eda"]).statistic
                    if np.isfinite(r):
                        rhos.append(float(r))
            rhos = np.array(rhos)
            if rhos.size:
                rows.append({"feature": f, "n_subjects": int(rhos.size),
                             "mean_rho": float(rhos.mean()),
                             "sd_rho": float(rhos.std()),
                             "frac_same_sign_as_mean": float(
                                 (np.sign(rhos) == np.sign(rhos.mean())).mean()),
                             "p10": float(np.percentile(rhos, 10)),
                             "p90": float(np.percentile(rhos, 90))})
        D = pd.DataFrame(rows)
        D.to_csv(run.artifact_path("per_subject_effects.csv"), index=False, encoding="utf-8")
        run.log_metric("invariant_feature_individual_spread", D.round(4).to_dict("records"))

        # ---------- 3. 方差分解：谁在听 vs 听什么 ----------
        gm = M["eda"].mean()
        ss_tot = float(((M["eda"] - gm) ** 2).sum())
        ss_subj = float(M.groupby("subject")["eda"].transform("mean").sub(gm).pow(2).sum())
        ss_clip = float(M.groupby("music_id")["eda"].transform("mean").sub(gm).pow(2).sum())
        run.log_metric("variance_decomposition",
                       {"subject_share": ss_subj / ss_tot, "clip_share": ss_clip / ss_tot,
                        "residual_share": 1 - (ss_subj + ss_clip) / ss_tot})

        # ---------- 打印 ----------
        print("\n" + "=" * 70)
        print(f"run_id: {run.run_id}")
        print(f"被试 {M['subject'].nunique()} 名 × 音频 {M['music_id'].nunique()} 段 "
              f"= {len(M)} 条记录（每名被试至少 {MIN_CLIPS_PER_SUBJECT} 段）\n")

        rel = run.meta["metrics"]["split_half_reliability"]
        print("--- 0. EDA 聚合量的分半信度（预测上限）⭐ ---")
        print(f"  分半 ρ = {rel['raw_mean']:+.3f}   "
              f"Spearman-Brown 校正 = {rel['spearman_brown']:+.3f}")
        print(f"  → 任何预测器对 EDA 的 ρ 都不可能稳定超过约 {abs(rel['spearman_brown']):.2f}")

        print("\n--- 1. 单个被试 vs 群体平均的一致性（Spearman ρ）---")
        print(f"  均值 {agree.mean():+.3f}   中位 {np.median(agree):+.3f}   "
              f"p10 {np.percentile(agree, 10):+.3f}   p90 {np.percentile(agree, 90):+.3f}")
        print(f"  与群体**反向**的被试占比: {(agree < 0).mean():.1%}")

        print("\n--- 2. 域无关特征对不同被试的效应散布 ---")
        print(f"{'特征':<24}{'均值ρ':>9}{'个体间SD':>10}{'同号率':>9}{'p10':>8}{'p90':>8}")
        for r in D.itertuples():
            print(f"{r.feature:<24}{r.mean_rho:>+9.3f}{r.sd_rho:>10.3f}"
                  f"{r.frac_same_sign_as_mean:>8.0%}{r.p10:>+8.3f}{r.p90:>+8.3f}")

        vd = run.meta["metrics"]["variance_decomposition"]
        print("\n--- 3. 生理唤醒的方差来自哪里 ---")
        print(f"  「谁在听」（被试）: {vd['subject_share']:.1%}")
        print(f"  「听什么」（音频）: {vd['clip_share']:.1%}")
        print(f"  残差             : {vd['residual_share']:.1%}")


if __name__ == "__main__":
    main()

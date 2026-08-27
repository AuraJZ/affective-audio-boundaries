"""PLAN III-1：用 PMEmo 皮电验证主观 arousal 学到的规律。

    uv run python scripts/13_eda_validation.py

## 要回答的三个问题

1. **主观 arousal 与生理唤醒本身有多一致？**
   若两者相关很弱，说明它们是不同构念，后续解释都要小心。

2. **用声学特征预测生理唤醒，能到什么程度？**
   直接建模 EDA，与建模主观打分对比。

3. **同一批声学特征，对两种标签的作用方向是否一致？**
   这是核心：若一致，说明我们学到的是「声音如何影响人」；
   若不一致，说明我们学到的只是「人如何描述声音」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from soundml.eda import load_eda_arousal
from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import N_SPLITS, SEED, build_regressors, feature_cols, reg_score

EDA_DIR = REPO_ROOT / "data" / "raw" / "PMEmo" / "PMEmo2019" / "EDA"
FEAT = REPO_ROOT / "data" / "features" / "layer_a_reg_pmemo_ln.parquet"


def cohens_d_by_median(x: np.ndarray, y: np.ndarray) -> float:
    """按 y 的中位数二分，返回 x 的标准化均差（高 y 组 − 低 y 组）。"""
    hi, lo = x[y >= np.median(y)], x[y < np.median(y)]
    if len(hi) < 2 or len(lo) < 2:
        return np.nan
    s = np.sqrt(((len(hi) - 1) * hi.var(ddof=1) + (len(lo) - 1) * lo.var(ddof=1))
                / (len(hi) + len(lo) - 2))
    return float((hi.mean() - lo.mean()) / s) if s > 0 else np.nan


def main() -> None:
    print("读取 794 个 EDA 文件…", flush=True)
    eda = load_eda_arousal(EDA_DIR)
    print(f"  得到 {len(eda)} 段音频的生理指标", flush=True)

    feats = pd.read_parquet(FEAT)
    feats["music_id"] = feats["clip_id"].str.removeprefix("pmemo_")
    df = feats.merge(eda, on="music_id", how="inner")

    cols = [c for c in feature_cols(feats) if c != "music_id"]
    X = SimpleImputer(strategy="median").fit_transform(df[cols].to_numpy(dtype=np.float64))
    y_subj = df["arousal"].to_numpy()
    y_eda = df["eda_arousal"].to_numpy()
    groups = df["group"].to_numpy()

    with RunRecord(
        "eda_validation", seed=SEED,
        params={"n_matched": len(df), "n_features": len(cols),
                "eda_index": "被试内 z 标准化后的 tonic_rise 与 phasic 均值",
                "question": "主观 arousal 学到的规律能否预测生理唤醒"},
    ) as run:
        run.log_metric("n_matched_clips", len(df))
        run.log_metric("n_subjects_median", float(df["n_subjects"].median()))

        # ---------- Q1：两种标签本身的一致性 ----------
        rho_labels = spearmanr(y_subj, y_eda).statistic
        run.log_metric("subjective_vs_eda_spearman", float(rho_labels))
        run.log_metric("subjective_vs_tonic", float(spearmanr(y_subj, df["eda_tonic"]).statistic))
        run.log_metric("subjective_vs_phasic", float(spearmanr(y_subj, df["eda_phasic"]).statistic))

        # ---------- Q2：分别建模两种标签 ----------
        cv = GroupKFold(n_splits=N_SPLITS)
        folds = list(cv.split(X, y_subj, groups))
        perf = {}
        for tgt_name, tgt in (("subjective", y_subj), ("eda", y_eda)):
            for algo in ("RF", "XGBoost", "ElasticNet"):
                preds = np.zeros(len(tgt))
                for tr, te in folds:
                    m = build_regressors()[algo]
                    m.fit(X[tr], tgt[tr])
                    preds[te] = m.predict(X[te])
                perf[f"{algo}|{tgt_name}"] = reg_score(tgt, preds)
        run.log_metric("model_performance", perf)

        # ---------- Q2b：交叉预测 ----------
        cross = {}
        for algo in ("RF", "XGBoost"):
            preds = np.zeros(len(y_subj))
            for tr, te in folds:
                m = build_regressors()[algo]
                m.fit(X[tr], y_subj[tr])          # 训练用主观
                preds[te] = m.predict(X[te])
            cross[f"{algo}|subj_model_on_eda"] = reg_score(y_eda, preds)  # 评价用生理
        run.log_metric("cross_target", cross)

        # ---------- Q3：逐特征方向一致性 ----------
        rows = []
        for j, c in enumerate(cols):
            rows.append({
                "feature": c,
                "d_subjective": cohens_d_by_median(X[:, j], y_subj),
                "d_eda": cohens_d_by_median(X[:, j], y_eda),
                "rho_subjective": float(spearmanr(X[:, j], y_subj).statistic),
                "rho_eda": float(spearmanr(X[:, j], y_eda).statistic),
            })
        D = pd.DataFrame(rows)
        D["same_direction"] = np.sign(D["d_subjective"]) == np.sign(D["d_eda"])
        D["min_abs_d"] = D[["d_subjective", "d_eda"]].abs().min(axis=1)
        D.sort_values("min_abs_d", ascending=False).to_csv(
            run.artifact_path("feature_direction_subj_vs_eda.csv"), index=False, encoding="utf-8")

        agree_all = float(D["same_direction"].mean())
        strong = D[D["d_subjective"].abs() > 0.3]
        agree_strong = float(strong["same_direction"].mean()) if len(strong) else np.nan
        run.log_metric("direction_agreement_all", agree_all)
        run.log_metric("direction_agreement_strong_subjective", agree_strong)
        run.log_metric("n_strong", int(len(strong)))

        # 域无关核心特征单独看
        INV = ["onset_env_dmean", "spec_contrast3_dmean", "onset_rate", "spec_flatness_std",
               "onset_env_mean", "spec_flatness_p50", "mfcc1_std"]
        sub = D[D["feature"].isin(INV)]
        run.log_metric("invariant_feature_directions",
                       sub.set_index("feature")[["d_subjective", "d_eda", "same_direction"]]
                       .round(3).to_dict("index"))

        # ---------- 打印 ----------
        print("\n" + "=" * 72)
        print(f"run_id: {run.run_id}")
        print(f"匹配到 {len(df)} 段音频，中位被试数 {df['n_subjects'].median():.0f}\n")

        print("--- Q1：主观 arousal vs 生理唤醒 ---")
        print(f"  综合 EDA 指标  ρ = {rho_labels:+.3f}")
        print(f"  仅 tonic       ρ = {spearmanr(y_subj, df['eda_tonic']).statistic:+.3f}")
        print(f"  仅 phasic      ρ = {spearmanr(y_subj, df['eda_phasic']).statistic:+.3f}")

        print("\n--- Q2：声学特征分别预测两种标签（组内 CV）---")
        P = pd.DataFrame(perf).T[["spearman", "r2", "mae"]]
        print(P.round(3).to_string())

        print("\n--- Q2b：主观模型直接预测生理唤醒 ---")
        for k, v in cross.items():
            print(f"  {k:<28} ρ = {v['spearman']:+.3f}")

        print("\n--- Q3：逐特征方向一致性 ---")
        print(f"  全部 {len(D)} 个特征        : {agree_all:.1%}")
        print(f"  主观上强效应的 {len(strong)} 个 : {agree_strong:.1%}")

        print("\n--- 域无关核心特征在两种标签下的方向 ---")
        print(f"{'特征':<24}{'主观 d':>10}{'生理 d':>10}  一致")
        for r in sub.itertuples():
            mark = "✓" if r.same_direction else "✗"
            print(f"{r.feature:<24}{r.d_subjective:>10.3f}{r.d_eda:>10.3f}  {mark}")


if __name__ == "__main__":
    main()

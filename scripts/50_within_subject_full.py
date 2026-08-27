"""被试内分析（156 维完整特征集）+ 量化「校正与否」造成的差别。

    .venv/Scripts/python scripts/50_within_subject_full.py

## 起因

用户问：用同一个数据库的其他文献为什么能得到阳性结果？

三个可能原因，本脚本查其中两个。

### 原因一：问的不是同一个问题（`45` 已部分回答）

| | 问的是 | 本项目 | 文献 |
|---|---|---|---|
| 跨受试刺激层面信度 | 不同的人对同一段音乐反应是否一致 | ICC ≤ 0.092 | 几乎不做 |
| **被试内**解码 | 一个人的脑电能否追踪他自己报告的状态 | **本脚本** | 标准做法 |

### 原因二：多重比较校正

`48` 的 156 维刺激层面结果里，**11 个 p<0.05（未校正）**，
BH-FDR 后 **0 个**。按逐个检验不校正的惯例，这就是一篇有阳性发现的论文。
本脚本对被试内结果同样并列报告两种口径。

### 原因三（未查，见 §限制）：伪迹处理

**我没有做 ICA 去眼电。** 试次级带功率与相干都包含眨眼与肌电，
这会抬高噪声、**压低**所有效应量 —— 方向上不利于我，是真缺口。

## 与 `45` 的差别

`45` 只用了 26 维带功率。但 `48` 显示**相干才是最强的族**
（ICC 0.092 vs 带功率 0.040）。因此被试内分析必须同样补齐到 156 维，
否则又是「用不含目标效应的特征集去否定该效应」。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
QUESTIONS = ["pleasant", "energetic", "tense", "angry",
             "afraid", "happy", "sad", "tender"]
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
MIN_TRIALS = 15


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """控制 z（试次序号）后的 x–y 秩相关 —— 同向疲劳漂移会伪造相关。"""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    A = np.c_[np.ones_like(rz), rz]

    def resid(a):
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]

    ex, ey = resid(rx), resid(ry)
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def bh(p: np.ndarray) -> np.ndarray:
    o = rankdata(p, method="ordinal")
    return np.minimum(1.0, p * len(p) / o)


def main() -> None:
    import importlib
    m45 = importlib.import_module("45_within_subject_decoding")

    A = pd.read_parquet(FEAT / "ds002721_trials_raw.parquet")
    B = pd.read_parquet(FEAT / "ds002721_connectivity.parquet")
    band = [c for c in A.columns if any(c.startswith(b) for b in BANDS) or c == "faa"]
    conn = [c for c in B.columns if c.startswith(("coh_", "icoh_", "asym_"))]
    D = A[["subject", "run", "trial", "stim"] + band].merge(
        B[["subject", "run", "trial"] + conn], on=["subject", "run", "trial"])
    D = D.merge(m45.reextract_ratings(), on=["subject", "run", "trial"], how="left")
    D["trial_index"] = D.groupby("subject").cumcount()
    feats = band + conn
    fam = {c: ("band power" if c in band else
               "coherence" if c.startswith("coh_") else
               "imaginary coherence" if c.startswith("icoh_") else "asymmetry")
           for c in feats}

    with RunRecord(
        "within_subject_full", seed=SEED,
        params={"n_features": len(feats), "n_questions": len(QUESTIONS),
                "n_combinations": len(feats) * len(QUESTIONS),
                "min_trials": MIN_TRIALS,
                "control": "偏掉试次序号",
                "question": "文献的阳性来自哪里：不同的问题，还是不同的校正口径",
                "known_gap": "未做 ICA 去眼电 —— 会压低所有效应量，方向不利于本结论"},
    ) as run:
        rows = []
        for (sub,), g in D.groupby(["subject"]):
            for q in QUESTIONS:
                s = g.dropna(subset=[q])
                if len(s) < MIN_TRIALS or s[q].nunique() < 3:
                    continue
                yv, tv = s[q].to_numpy(dtype=float), s.trial_index.to_numpy()
                for c in feats:
                    xv = s[c].to_numpy(dtype=float)
                    if not np.isfinite(xv).all():
                        continue
                    rows.append({"subject": sub, "question": q, "feature": c,
                                 "family": fam[c],
                                 "rho": partial_spearman(xv, yv, tv)})
            print(f"  {sub}", flush=True)
        C = pd.DataFrame(rows).dropna(subset=["rho"])

        # 组水平：每个「维度 × 特征」组合上，跨受试的 ρ 是否偏离 0
        res = []
        for (q, c), g in C.groupby(["question", "feature"]):
            v = g.rho.to_numpy()
            if len(v) >= 10:
                st = wilcoxon(v)
                res.append({"question": q, "feature": c,
                            "family": g.family.iloc[0], "n_subjects": len(v),
                            "mean_rho": float(v.mean()),
                            "frac_same_sign": float(max((v > 0).mean(),
                                                        (v < 0).mean())),
                            "p": float(st.pvalue)})
        R = pd.DataFrame(res)
        R["q"] = bh(R.p.to_numpy())
        R = R.sort_values("p")

        n_unc = int((R.p < 0.05).sum())
        n_fdr = int((R.q < 0.05).sum())
        exp_unc = 0.05 * len(R)
        run.log_metric("n_combinations", int(len(R)))
        run.log_metric("n_uncorrected_p05", n_unc)
        run.log_metric("n_expected_by_chance", float(exp_unc))
        run.log_metric("n_fdr05", n_fdr)
        run.log_metric("max_abs_mean_rho", float(R.mean_rho.abs().max()))
        run.log_metric("by_family",
                       R.groupby("family").agg(
                           n=("p", "size"),
                           n_unc=("p", lambda s: int((s < 0.05).sum())),
                           max_abs_rho=("mean_rho", lambda s: float(s.abs().max()))
                       ).to_dict("index"))

        print("\n" + "=" * 84)
        print(f"run_id: {run.run_id}   受试 {C.subject.nunique()}   "
              f"组合 {len(R)}（{len(feats)} 特征 × {C.question.nunique()} 维度）")

        print("\n--- 两种报告口径的差别 ---")
        print(f"  未校正 p<0.05      **{n_unc}** 个    "
              f"（纯随机期望 {exp_unc:.0f} 个）")
        print(f"  BH-FDR   q<0.05    **{n_fdr}** 个")
        print(f"  最大 |ρ均值|        {R.mean_rho.abs().max():.3f}")

        print("\n--- 未校正下最强的 12 个（这就是常规口径会报告的东西）---")
        for r_ in R.head(12).itertuples():
            print(f"  {r_.question:10s} {r_.feature:20s} {r_.family:20s} "
                  f"ρ={r_.mean_rho:+.3f}  同号 {100*r_.frac_same_sign:.0f}%  "
                  f"p={r_.p:.4f}  q={r_.q:.3f}")

        print("\n--- 分特征族 ---")
        for f_, g in R.groupby("family"):
            print(f"  {f_:22s} n={len(g):4d}   未校正 p<0.05 {int((g.p<0.05).sum()):3d}"
                  f"   期望 {0.05*len(g):5.1f}   最大 |ρ| {g.mean_rho.abs().max():.3f}")

        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "ds002721_within_subject_full.csv", index=False)

        print("\n" + "=" * 84)
        if n_unc > exp_unc * 1.5 and n_fdr == 0:
            print("→ 未校正下的「阳性」数量高于随机期望，但无一通过 FDR。")
            print("  **文献与本项目的差别主要在报告口径，不在数据。**")
        elif n_fdr:
            print(f"→ {n_fdr} 个组合通过 FDR —— 存在稳健的被试内关系，须报告。")
        else:
            print("→ 未校正下的阳性数量与随机期望相当 —— 无证据支持被试内关系。")
        print("\n⚠️ 未做 ICA 去眼电。伪迹会压低所有效应量，"
              "**方向上不利于本结论**，须在报告中声明。")


if __name__ == "__main__":
    main()

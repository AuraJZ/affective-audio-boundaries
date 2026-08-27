"""同一批数据，两种分析口径 —— 「阳性」能凭分析选择造出多少。

    .venv/Scripts/python scripts/51_pooled_vs_subject_aware.py

## 问题

用同一数据库的文献报告了阳性效应，而本项目的被试内分析精确落在随机期望上
（1248 个组合，未校正 p<0.05 有 59 个，期望 62 个）。差别从哪来？

## 本脚本比较三种口径，数据、特征、对比完全相同，只改统计单元

| 口径 | 统计单元 | 常见于 |
|---|---|---|
| **A 合并试次** | 1,240 个试次视作独立 | 早期 EEG 文献的常见做法 |
| **B 被试内中位分割 + 合并** | 同上，但分割阈用被试自身中位数 | 控制了个体量表用法，仍视试次独立 |
| **C 被试感知** | 先算每人的效应，再做组水平检验 | 本项目 `50` 用的口径 |

口径 A 把 31 名受试的 1,240 个试次当独立样本，**自由度虚高约 40 倍**。
这是重复测量数据的经典陷阱：同一人的 40 个试次彼此相关，
把它们当独立样本会让标准误缩小到不该有的程度。

## 这不是对任何具体文献的指控

我没有拿到 Daly et al. (2014) 的方法学细节（该文非开放获取）。
本脚本只回答一个可验证的问题：**在这批数据上，仅仅改变统计单元，
能造出多少「显著」结果。** 结论对任何使用重复测量设计的分析都成立。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
QUESTIONS = ["pleasant", "energetic", "tense", "angry",
             "afraid", "happy", "sad", "tender"]
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
MIN_PER_GROUP = 5


def bh(p: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, p * len(p) / rankdata(p, method="ordinal"))


def main() -> None:
    import importlib
    m45 = importlib.import_module("45_within_subject_decoding")

    A = pd.read_parquet(FEAT / "ds002721_trials_raw.parquet")
    B = pd.read_parquet(FEAT / "ds002721_connectivity.parquet")
    band = [c for c in A.columns if any(c.startswith(b) for b in BANDS) or c == "faa"]
    conn = [c for c in B.columns if c.startswith(("coh_", "icoh_", "asym_"))]
    D = A[["subject", "run", "trial"] + band].merge(
        B[["subject", "run", "trial"] + conn], on=["subject", "run", "trial"])
    D = D.merge(m45.reextract_ratings(), on=["subject", "run", "trial"], how="left")
    feats = band + conn

    with RunRecord(
        "pooled_vs_subject_aware", seed=SEED,
        params={"n_features": len(feats), "n_questions": len(QUESTIONS),
                "n_combinations": len(feats) * len(QUESTIONS),
                "comparison": ["A 合并试次（视作独立）",
                               "B 被试内中位分割后合并",
                               "C 被试感知（先个体后组）"],
                "point": "数据/特征/对比不变，只改统计单元"},
    ) as run:
        rows = []
        for q in QUESTIONS:
            S = D.dropna(subset=[q]).copy()
            if len(S) < 200:
                continue
            # 全局中位分割（口径 A）与被试内中位分割（口径 B）
            S["hi_global"] = S[q] > S[q].median()
            S["hi_within"] = S[q] > S.groupby("subject")[q].transform("median")

            for c in feats:
                v = S[c].to_numpy(dtype=float)
                if not np.isfinite(v).all():
                    continue
                out = {"question": q, "feature": c}

                for tag, col in (("A", "hi_global"), ("B", "hi_within")):
                    g1, g0 = v[S[col].to_numpy()], v[~S[col].to_numpy()]
                    out[f"p_{tag}"] = (float(mannwhitneyu(g1, g0).pvalue)
                                       if min(len(g1), len(g0)) > 20 else np.nan)
                    sd = np.sqrt((g1.var(ddof=1) + g0.var(ddof=1)) / 2)
                    out[f"d_{tag}"] = float((g1.mean() - g0.mean()) / sd) if sd > 0 else np.nan

                # 口径 C：先算每人的组间差，再做组水平检验
                per = []
                for _, g in S.groupby("subject"):
                    m = g["hi_within"].to_numpy()
                    x = g[c].to_numpy(dtype=float)
                    if m.sum() >= MIN_PER_GROUP and (~m).sum() >= MIN_PER_GROUP:
                        per.append(float(np.mean(x[m]) - np.mean(x[~m])))
                if len(per) >= 10:
                    out["p_C"] = float(wilcoxon(per).pvalue)
                    out["n_subjects_C"] = len(per)
                else:
                    out["p_C"], out["n_subjects_C"] = np.nan, len(per)
                rows.append(out)
            print(f"  {q}", flush=True)

        R = pd.DataFrame(rows)
        for tag in "ABC":
            m = R[f"p_{tag}"].notna()
            R.loc[m, f"q_{tag}"] = bh(R.loc[m, f"p_{tag}"].to_numpy())

        n = int(R.p_A.notna().sum())
        summary = {}
        for tag, name in (("A", "合并试次（视作独立）"),
                          ("B", "被试内中位分割后合并"),
                          ("C", "被试感知")):
            p, qq = R[f"p_{tag}"], R[f"q_{tag}"]
            summary[name] = {
                "n_tested": int(p.notna().sum()),
                "p<0.05": int((p < 0.05).sum()),
                "p<0.001": int((p < 0.001).sum()),
                "FDR q<0.05": int((qq < 0.05).sum()),
                "min_p": float(p.min()),
            }
        run.log_metric("summary", summary)
        run.log_metric("chance_expectation_p05", 0.05 * n)

        print("\n" + "=" * 86)
        print(f"run_id: {run.run_id}   {n} 个「维度 × 特征」组合   "
              f"试次 {len(D):,}   受试 {D.subject.nunique()}")
        print(f"\n{'口径':<26}{'p<0.05':>9}{'p<0.001':>10}{'FDR<0.05':>11}"
              f"{'最小 p':>12}")
        print(f"{'（纯随机期望）':<24}{0.05*n:>9.0f}{0.001*n:>10.0f}"
              f"{0:>11}{'—':>12}")
        for name, s in summary.items():
            print(f"{name:<24}{s['p<0.05']:>9}{s['p<0.001']:>10}"
                  f"{s['FDR q<0.05']:>11}{s['min_p']:>12.2e}")

        print("\n--- 口径 A 下最「显著」的 8 个（效应量同时给出）---")
        for r_ in R.nsmallest(8, "p_A").itertuples():
            print(f"  {r_.question:10s} {r_.feature:20s} "
                  f"p_A={r_.p_A:.2e}  d={r_.d_A:+.3f}   "
                  f"→ 同一组合 p_C={r_.p_C:.3f}")

        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 "ds002721_pooled_vs_subject.csv", index=False)

        # 判决按「未校正阳性数 vs 随机期望」给，不按 FDR 计数 ——
        # 三种口径的 FDR 都是 0，只看 FDR 会把 105 vs 49 vs 62 这个
        # 真正的发现整个漏掉。
        print("\n" + "=" * 86)
        exp = 0.05 * n
        a = summary["合并试次（视作独立）"]["p<0.05"]
        b = summary["被试内中位分割后合并"]["p<0.05"]
        c = summary["被试感知"]["p<0.05"]
        run.log_metric("inflation_ratio_A_over_chance", a / exp)
        run.log_metric("nominal_positives", {"A": a, "B": b, "C": c,
                                             "chance": exp})
        print(f"→ 未校正阳性：A {a}   B {b}   C {c}   随机期望 {exp:.0f}")
        if a > exp * 1.3 and b <= exp * 1.1:
            print(f"\n  **A 与 B 的唯一差别是分割阈：全局中位数 vs 被试自身中位数。**")
            print(f"  用全局阈时阳性 {a} 个（{a/exp:.1f}× 随机），"
                  f"改用被试内阈后掉到 {b} 个。")
            print("  混杂来源明确：评分习惯偏高的人同时具有不同的脑电基线，")
            print("  **受试间差异被当成了试次间效应**。")
        if summary["合并试次（视作独立）"]["FDR q<0.05"] == 0 and \
           summary["被试感知"]["FDR q<0.05"] == 0:
            print("\n  但三种口径**都没有**产生 FDR 显著的结果 ——")
            print("  仅靠改变统计单元，复现不出文献报告的阳性。")
            print("  尚未排除：情绪类别（而非中位分割）的对比、"
                  "源空间或其他预处理、**以及本项目未做 ICA 去眼电**。")
        print("\n  效应量：口径 A 下最「显著」的组合 d 约 0.14–0.27，"
              "组间分布重叠约 90%。")
        print("  p 值小是因为 n 大，不是因为效应大。")


if __name__ == "__main__":
    main()

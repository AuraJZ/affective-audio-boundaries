"""逐个体解码：**这个人**的脑电能否预测**他自己**的状态。

    .venv/Scripts/python scripts/53_per_subject_decoding.py

## 我此前问错了问题 🔴

`45` / `50` 名义上是「被试内」分析，但最终都汇总到一个**组水平** Wilcoxon，
问的是「这个关系在人群里是否一致」。**那个问题失败，不代表个体问题失败** ——
若 31 个人各自都有稳定关系，但关系的形式因人而异（甚至方向相反），
组平均必然为零。而这正是按人校准所要的情形。

本项目的产品前提是**按人校准**，因此正确的问题是：

> 有多少个体，其脑电能以超过**该个体自身零分布**的水平预测其自评状态？

## 与组水平检验的关键差别

| | 组水平（`50`） | 逐个体（本脚本） |
|---|---|---|
| 零分布 | 跨受试 ρ 的均值是否偏离 0 | **每个人自己的打乱标签零分布** |
| 关系方向因人而异 | 相互抵消 → 阴性 | **各自计数 → 仍可为阳性** |
| 结论形式 | 「存在通用关系」 | 「N/31 人可校准」 |

## 逐个体阳性对照 ⭐

沿用本项目的纪律：**先证明该个体的数据能测出已知效应**。

个体级的阳性对照取 **试次序号**（session 内的时间）：脑电必然随时间漂移
（阻抗、疲劳、警觉度），因此每个人的脑电都应能显著预测试次序号。
**某个体连这个都测不出 → 该个体数据不可用，其阴性不可解读。**

这是把「阳性对照先行」从数据集层面下沉到个体层面。

## 统计

- 特征 156 维，试次 ~30 → 折内 StandardScaler → PCA(8) → RidgeCV
- 5 折 CV，指标 Spearman ρ(预测, 实际)
- **零分布：该个体自己打乱标签，跑完全相同的流程 200 次**
  （CV 与 PCA 都在零分布里重跑，因此过拟合被零分布吸收）
- 逐个体 p；再对全部「个体 × 维度」做 BH-FDR
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soundml.provenance import REPO_ROOT, RunRecord
from soundml.regression import SEED

warnings.filterwarnings("ignore")

FEAT = REPO_ROOT / "data" / "features"
QUESTIONS = ["pleasant", "energetic", "tense", "angry",
             "afraid", "happy", "sad", "tender"]
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
MIN_TRIALS = 20
N_FOLDS = 5
N_PERM = 200
N_PC = 8


def bh(p: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, p * len(p) / rankdata(p, method="ordinal"))


def cv_rho(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """5 折 CV 的 ρ(预测, 实际)。PCA 在折内拟合，不泄漏。"""
    pred = np.empty(len(y))
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=seed).split(X):
        if np.std(y[tr]) < 1e-9:
            return np.nan
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(N_PC, len(tr) - 1, X.shape[1])),
            RidgeCV(alphas=np.logspace(-2, 4, 15)))
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
    r = spearmanr(pred, y).statistic
    return float(r) if np.isfinite(r) else np.nan


def per_subject_test(X: np.ndarray, y: np.ndarray,
                     rng: np.random.Generator) -> tuple[float, float, float]:
    """返回 (观测 ρ, 逐个体置换 p, 零分布均值)。"""
    obs = cv_rho(X, y, SEED)
    if not np.isfinite(obs):
        return np.nan, np.nan, np.nan
    null = []
    for i in range(N_PERM):
        r = cv_rho(X, rng.permutation(y), SEED)
        if np.isfinite(r):
            null.append(r)
    if len(null) < N_PERM // 2:
        return obs, np.nan, np.nan
    null = np.asarray(null)
    return obs, float((np.sum(null >= obs) + 1) / (len(null) + 1)), float(null.mean())


def main() -> None:
    import argparse
    import importlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--ica", action="store_true",
                    help="改用 ICA 去眼电后的特征（52 产出），其余完全不变")
    args = ap.parse_args()
    m45 = importlib.import_module("45_within_subject_decoding")

    suf = "_ica" if args.ica else ""
    A = pd.read_parquet(FEAT / f"ds002721_trials{suf or '_raw'}.parquet")
    B = pd.read_parquet(FEAT / f"ds002721_connectivity{suf}.parquet")
    band = [c for c in A.columns if any(c.startswith(b) for b in BANDS) or c == "faa"]
    conn = [c for c in B.columns if c.startswith(("coh_", "icoh_", "asym_"))]
    D = A[["subject", "run", "trial"] + band].merge(
        B[["subject", "run", "trial"] + conn], on=["subject", "run", "trial"])
    D = D.merge(m45.reextract_ratings(), on=["subject", "run", "trial"], how="left")
    D["trial_index"] = D.groupby("subject").cumcount()
    feats = band + conn

    with RunRecord(
        "per_subject_decoding" + ("_ica" if args.ica else ""), seed=SEED,
        params={"ica_cleaned": args.ica, "n_features": len(feats), "n_pc": N_PC, "n_folds": N_FOLDS,
                "n_perm_per_subject": N_PERM, "min_trials": MIN_TRIALS,
                "question": "有多少个体的脑电能超过其自身零分布地预测自评状态",
                "positive_control": "逐个体：脑电预测试次序号（时间漂移必然存在）",
                "why": "产品前提是按人校准；组水平检验回答的是另一个问题"},
    ) as run:
        rng = np.random.default_rng(SEED)
        rows, ctrl = [], []

        for si, (sub, g) in enumerate(D.groupby("subject"), 1):
            Xall = np.nan_to_num(g[feats].to_numpy(dtype=float), nan=0.0)

            # ---- 逐个体阳性对照：预测试次序号 ----
            o, p, nm = per_subject_test(Xall, g.trial_index.to_numpy(dtype=float), rng)
            ctrl.append({"subject": sub, "n_trials": len(g), "rho": o,
                         "p": p, "null_mean": nm})

            for q in QUESTIONS:
                s = g.dropna(subset=[q])
                if len(s) < MIN_TRIALS or s[q].nunique() < 3:
                    continue
                X = np.nan_to_num(s[feats].to_numpy(dtype=float), nan=0.0)
                o, p, nm = per_subject_test(X, s[q].to_numpy(dtype=float), rng)
                rows.append({"subject": sub, "question": q, "n": len(s),
                             "rho": o, "p": p, "null_mean": nm})
            print(f"  {si}/{D.subject.nunique()}  {sub}", flush=True)

        R = pd.DataFrame(rows).dropna(subset=["p"])
        C = pd.DataFrame(ctrl).dropna(subset=["p"])
        R["q"] = bh(R.p.to_numpy())

        # ---------- 阳性对照 ----------
        ctrl_ok = C[C.p < 0.05].subject.tolist()
        run.log_metric("per_subject_positive_control",
                       {"n_subjects": int(len(C)), "n_passed": len(ctrl_ok),
                        "mean_rho": float(C.rho.mean()),
                        "mean_null": float(C.null_mean.mean())})

        # ---------- 主结果 ----------
        n_sub = R.subject.nunique()
        n_tests = len(R)
        n_sig = int((R.p < 0.05).sum())
        exp = 0.05 * n_tests
        bt = binomtest(n_sig, n_tests, 0.05, alternative="greater")
        subs_any = R[R.p < 0.05].subject.nunique()
        run.log_metric("results", {
            "n_tests": n_tests, "n_subjects": int(n_sub),
            "n_p05": n_sig, "expected_by_chance": float(exp),
            "binomial_p": float(bt.pvalue),
            "n_subjects_with_any_p05": int(subs_any),
            "expected_subjects_any": float(n_sub * (1 - 0.95 ** 8)),
            "n_fdr05": int((R.q < 0.05).sum()),
            "max_rho": float(R.rho.max())})

        print("\n" + "=" * 86)
        print(f"run_id: {run.run_id}   受试 {n_sub}   检验 {n_tests}"
              f"（个体 × 维度），每个带 {N_PERM} 次个体自身置换")

        print("\n--- 逐个体阳性对照：脑电预测试次序号 ---")
        print(f"  {len(C)} 名受试   通过（p<0.05）**{len(ctrl_ok)}**   "
              f"实测 ρ 均值 {C.rho.mean():+.3f}   零分布 {C.null_mean.mean():+.3f}")
        if len(ctrl_ok) < len(C) * 0.5:
            print("  ⚠️ 过半个体连时间漂移都测不出 —— 其阴性结果不可解读。")

        print("\n--- 主结果：个体能否预测自身状态 ---")
        print(f"  p<0.05 的「个体 × 维度」   **{n_sig}** / {n_tests}"
              f"   随机期望 {exp:.1f}   二项检验 p={bt.pvalue:.4f}")
        print(f"  至少一个维度显著的个体     **{subs_any}** / {n_sub}"
              f"   随机期望 {n_sub*(1-0.95**8):.1f}")
        print(f"  BH-FDR q<0.05             **{int((R.q < 0.05).sum())}**")
        print(f"  最强 ρ                    {R.rho.max():+.3f}")

        print("\n--- 逐维度：多少个体显著 ---")
        for q in QUESTIONS:
            g = R[R.question == q]
            if len(g):
                print(f"  {q:10s} {int((g.p<0.05).sum()):2d}/{len(g):2d} 名受试   "
                      f"最强 ρ {g.rho.max():+.3f}   ρ 中位 {g.rho.median():+.3f}")

        top = R.nsmallest(12, "p")
        print("\n--- 最强的 12 个「个体 × 维度」---")
        for r_ in top.itertuples():
            mark = "✅" if r_.subject in ctrl_ok else "⚠️对照未过"
            print(f"  {r_.subject} {r_.question:10s} n={r_.n:2d}  "
                  f"ρ={r_.rho:+.3f}（零分布 {r_.null_mean:+.3f}）  "
                  f"p={r_.p:.4f}  q={r_.q:.3f}  {mark}")

        R.to_csv(REPO_ROOT / "reports" / "source_data" /
                 f"ds002721_per_subject_decoding{suf}.csv", index=False)
        C.to_csv(REPO_ROOT / "reports" / "source_data" /
                 f"ds002721_per_subject_control{suf}.csv", index=False)

        print("\n" + "=" * 86)
        if bt.pvalue < 0.05 and subs_any > n_sub * (1 - 0.95 ** 8) * 1.5:
            print(f"→ **显著个体数高于随机期望**（{n_sig} vs {exp:.1f}，"
                  f"二项 p={bt.pvalue:.4f}）。")
            print(f"  {subs_any}/{n_sub} 名受试至少有一个维度可解码 —— "
                  "**按人校准在这批数据上有支持**。")
            print("  须同时报告：组水平检验为阴性，因此关系的形式因人而异。")
        else:
            print(f"→ 显著个体数与随机期望相当（{n_sig} vs {exp:.1f}，"
                  f"二项 p={bt.pvalue:.4f}）。")
            print("  在本数据集的条件下（~30 试次/人，12 s 片段，单次暴露），")
            print("  **个体层面同样没有可检出的脑电→自评关系**。")
            print("  这不排除更长暴露、更多试次或个体内重复设计下存在关系。")


if __name__ == "__main__":
    main()
